# ALASKA2 Image Steganalysis

- This is my code for [ALASKA2 Image Steganalysis](https://www.kaggle.com/c/alaska2-image-steganalysis) where our team ended up at [24th place](https://www.kaggle.com/c/alaska2-image-steganalysis/leaderboard) in the competition.

## Solution Overview

- Ensemble of EfficientNet-B2, EfficientNet-B3 and EfficientNet-B4 models trained on 4 classes
- Augmentations: HorizontalFlip and RandomRotate90
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Scheduler: CosineAnnealingWarmUpRestarts
- NVIDIA Apex for mixed precision training
- Test-time augmentations: D4

## How to run

### Preprocessing

~~~
$ python src/generate_folds.py
~~~

[generate_folds.py](src/generate_folds.py) generates pandas dataframe with 5 folds for cross validation. 

### Training 

~~~
$ python src/train.py
~~~

[train.py](src/train.py) trains EfficientNet-B4 with parameters from [config.py](src/config.py)


### Predicting 

~~~
$ python src/predict.py
~~~

[predict.py](src/predict.py) makes predictions for test data using best model checkpoint