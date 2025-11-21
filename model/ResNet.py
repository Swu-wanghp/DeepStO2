"""
Author: yida
Time is: 2022/1/7 19:49
this Code: 重新实现真正的Resnet18, 同torch官方实现的model
不能重复的原因:
1.没有按照指定方法初始化参数
2.BN层指定初始化准确率也能提升1-2%
结果:现在能和官方的model获得相同准确率
很值得参考的博客https://blog.csdn.net/weixin_44331304/article/details/106127552?spm=1001.2014.3001.5501
"""
import os

import torch
import torch.nn as nn
from torchvision import models

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BasicBlock(nn.Module):
    def __init__(self, in_channel, s):
        """
        基础模块, 共有两种形态, 1.s=1输入输出维度相同时 2.s=2特征图大小缩小一倍, 维度扩充一倍
        :param in_channel: 输入通道数维度
        :param s: s=1 不缩小 s=2 缩小尺度
        """
        super(BasicBlock, self).__init__()
        self.s = s
        self.conv1 = nn.Conv2d(in_channel, in_channel * s, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel * s)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel * s, in_channel * s, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel * s)
        if self.s == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * s, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(in_channel * s)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.s == 2:  # 缩小
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, n_class, zero_init_residual=True):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=1),
            BasicBlock(in_channel=64, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channel=128, s=2),
            BasicBlock(in_channel=256, s=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channel=256, s=2),
            BasicBlock(in_channel=512, s=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_class)

        # 初始化参数 -> 影响准确率 7%
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 初始化BasicBlock -> 影响准确率 1-2%
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# class ModifiedResNet18(nn.Module):
#     def __init__(self, pretrained=True, in_channels=1, num_classes=1, freeze_conv=True):
#         """
#         初始化 ModifiedResNet18 模型。

#         参数:
#         - pretrained (bool): 是否加载预训练权重，默认为 True。
#         - in_channels (int): 输入图像的通道数，默认为 1（单通道）。
#         - num_classes (int): 输出类别数，默认为 1（二分类）。
#         - freeze_conv (bool): 是否冻结卷积层参数，默认为 True。
#         """
#         super(ModifiedResNet18, self).__init__()

#         # 加载预训练的 ResNet18 模型
#         self.model = models.resnet18(pretrained=pretrained)

#         # 修改第一层卷积层以适应单通道输入
#         original_conv1 = self.model.conv1  # 原始的第一层卷积层
#         self.model.conv1 = nn.Conv2d(
#             in_channels=in_channels,  # 修改输入通道数
#             out_channels=original_conv1.out_channels,
#             kernel_size=original_conv1.kernel_size,
#             stride=original_conv1.stride,
#             padding=original_conv1.padding,
#             bias=original_conv1.bias
#         )

#         # 初始化新卷积层的权重
#         with torch.no_grad():
#             self.model.conv1.weight[:, 0, :, :] = original_conv1.weight.mean(dim=1)

#         # 修改最后一层全连接层以适应二分类任务
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#         # 冻结卷积层参数
#         if freeze_conv:
#             for name, param in self.model.named_parameters():
#                 if "fc" not in name:  # 不冻结全连接层
#                     param.requires_grad = False

#     def forward(self, x):
#         """
#         定义前向传播。

#         参数:
#         - x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)。

#         返回:
#         - torch.Tensor: 输出张量，形状为 (batch_size, num_classes)。
#         """
#         return self.model(x)
    
    
class ModifiedResNet18(nn.Module):
    def __init__(self, pretrained=True, in_channels=1, num_classes=1):
        """
        初始化 ModifiedResNet18 模型。

        参数:
        - pretrained (bool): 是否加载预训练权重，默认为 True。
        - in_channels (int): 输入图像的通道数，默认为 1（单通道）。
        - num_classes (int): 输出类别数，默认为 1（二分类）。
        """
        super(ModifiedResNet18, self).__init__()

        # 加载预训练的 ResNet18 模型
        self.model = models.resnet18(pretrained=pretrained)

        # 修改第一层卷积层以适应单通道输入
        original_conv1 = self.model.conv1  # 原始的第一层卷积层
        self.model.conv1 = nn.Conv2d(
            in_channels=in_channels,  # 修改输入通道数
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )

        # 初始化新卷积层的权重
        with torch.no_grad():
            self.model.conv1.weight[:, 0, :, :] = original_conv1.weight.mean(dim=1)

        # 修改最后一层全连接层以适应二分类任务
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        定义前向传播。

        参数:
        - x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)。

        返回:
        - torch.Tensor: 输出张量，形状为 (batch_size, num_classes)。
        """
        return self.model(x)
    
# if __name__ == '__main__':
#     inputs = torch.rand((4, 1, 481, 411))
#     model = ResNet18(n_class=2)
#     print(model)
#     outputs = model(inputs)
#     print(outputs.shape)

if __name__ == "__main__":
    # 创建模型实例
    model = ModifiedResNet18(in_channels=1, num_classes=1)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 打印模型结构
    print(model)

    # 示例输入
    example_input = torch.randn(1, 1, 224, 224).to(device)  # 单通道输入，形状为 (batch_size, channels, height, width)
    output = model(example_input)
    print("输出形状:", output.shape)

