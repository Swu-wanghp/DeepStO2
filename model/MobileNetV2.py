import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Conv2dNormActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=6):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.conv = nn.Sequential(
            Conv2dNormActivation(in_channels, hidden_dim, kernel_size=1),
            Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2, self).__init__()
        self.features = nn.Sequential(
            Conv2dNormActivation(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            InvertedResidual(32, 16, stride=1, expansion_factor=1),
            InvertedResidual(16, 24, stride=2),
            # InvertedResidual(24, 24, stride=1),
            # InvertedResidual(24, 32, stride=2),
            # InvertedResidual(32, 32, stride=1),
            # InvertedResidual(32, 64, stride=2),
            # InvertedResidual(64, 64, stride=1),
            # InvertedResidual(64, 96, stride=1),
            # InvertedResidual(96, 96, stride=1),
            # InvertedResidual(96, 160, stride=2),
            # InvertedResidual(160, 160, stride=1),
            # InvertedResidual(160, 320, stride=1),
            Conv2dNormActivation(24, 1280, kernel_size=1),
        )
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

