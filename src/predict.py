import numpy as np
import pandas as pd
import ttach as tta
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import get_test_dataset
from utils import set_seed, set_cuda_device


def predict(loader, model, device):
    model.eval()
    result = {'Id': [], 'Label': []}
    for image_names, images in tqdm(loader):
        labels = []
        for transformer in tta.aliases.d4_transform():
            augmented_image = transformer.augment_image(images)
            with torch.no_grad():
                y_pred = model(augmented_image.to(device))
                y_pred = 1 - F.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]
            labels.append(y_pred[:, None])
        y_pred = np.concatenate(labels, axis=1)
        result['Id'].extend(image_names)
        result['Label'].append(y_pred)
    result['Label'] = np.concatenate(result['Label']).mean(1)
    submission = pd.DataFrame(result)
    submission = submission.fillna(0.5)
    submission.to_csv(config.submission_path, index=False)


if __name__ == '__main__':
    set_seed(config.seed)
    set_cuda_device(config.gpu)
    model = torch.load(f'model/{config.workdir}/best_model.pth').to(config.device)
    test_dataset = get_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    predict(test_loader, model, config.device)
