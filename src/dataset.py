import glob
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from augmentations import train_transforms, test_transforms


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class AlaskaDataset(Dataset):
    def __init__(self, img_dir, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.img_dir = img_dir
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        image = cv2.imread(f'{self.img_dir}/{kind}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transforms:
            image = self.transforms(image=image)['image']

        target = onehot(4, label)
        return image, target

    def __len__(self):
        return len(self.image_names)


class AlaskaInferenceDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        super().__init__()
        self.image_paths = image_paths
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transforms:
            image = self.transforms(image=image)['image']

        return image_path.split('/')[-1], image

    def __len__(self):
        return len(self.image_paths)


def get_train_valid_datasets(config):
    df = pd.read_csv(config.folds_df_path)
    train_df = df[df['fold'] != config.fold]
    valid_df = df[df['fold'] == config.fold]

    train_dataset = AlaskaDataset(
        img_dir=config.img_dir,
        kinds=train_df.kind.values,
        image_names=train_df.image_name.values,
        labels=train_df.label.values,
        transforms=train_transforms,
    )

    valid_dataset = AlaskaDataset(
        img_dir=config.img_dir,
        kinds=valid_df.kind.values,
        image_names=valid_df.image_name.values,
        labels=valid_df.label.values,
        transforms=test_transforms,
    )

    return train_dataset, valid_dataset


def get_test_dataset(config):
    test_images = glob.glob(f'{config.img_dir}/Test/*.jpg')
    return AlaskaInferenceDataset(
        image_paths=test_images,
        transforms=test_transforms,
    )
