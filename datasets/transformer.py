import random

from torchvision import transforms

# 转成Tensor
def transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])
