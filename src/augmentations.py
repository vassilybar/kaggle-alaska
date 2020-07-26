import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    A.Normalize(),
    ToTensorV2(),
])

test_transforms = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])
