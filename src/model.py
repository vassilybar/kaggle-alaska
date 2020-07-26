import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def get_model(model_name, out_features):
    model = EfficientNet.from_pretrained(model_name)
    model._fc = nn.Linear(model._fc.in_features, out_features)
    return model
