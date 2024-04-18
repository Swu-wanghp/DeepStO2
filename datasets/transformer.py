import random

from torchvision import transforms


def transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])
