import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_mask(H, W, top_lef, bottom_right):
    """
        生成矩形掩码矩阵
        H: 高度 (rows)
        W: 宽度 (cols)
        top_left: (y1, x1) 左上角
        bottom_right: (y2, x2) 右下角
    """
    mask = torch.zeros((H, W), dtype=torch.float32)
    y1, x1 = top_lef
    y2, x2 = bottom_right

    # 限制在图像范围内，避免越界
    y1, x1 = max(0, y1), max(0, x1)
    y2, x2 = min(H - 1, y2), min(W - 1, x2)

    mask[y1:y2 + 1, x1:x2 + 1] = 1.0
    return mask

class RegionFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):

        mask = mask.unsqueeze(0).unsqueeze(0)

        # 下采样 mask 到特征图大小
        mask_down = F.interpolate(mask, size=x.shape[2:], mode='nearest')  # [1,1,H/8,W/8]
        feat = x * mask_down  # 掩码作用

        return feat

# 区域卷积 + 金字塔池化
# class RegionConvSPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.sequence_basic = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),  # Conv1
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # 下采样一半
#
#             # 16, 32
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Conv2
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             # 32, 64
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv3
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         # 多尺度池化层 (SPP) 1，2，4
#         self.pool_sizes = [1, 2, 4]
#         # 64
#         self.fc = nn.Linear(64 * sum([p * p for p in self.pool_sizes]), out_channels)
#
#     def forward(self, x, mask):
#         feat = self.sequence_basic(x)
#         # mask 扩展并下采样到 CNN 输出大小
#         mask = mask.unsqueeze(0).unsqueeze(0)
#
#         # 下采样 mask 到特征图大小
#         mask_down = F.interpolate(mask, size=feat.shape[2:], mode='nearest')  # [1,1,H/8,W/8]
#         feat = feat * mask_down  # 掩码作用
#
#         # Spatial Pyramid Pooling
#         spp_features = []
#         for p in self.pool_sizes:
#             pooled = F.adaptive_avg_pool2d(feat, output_size=(p, p))
#             spp_features.append(pooled.view(feat.size(0), -1))
#
#         spp_out = torch.cat(spp_features, dim=1)
#
#         out = self.fc(spp_out)  # 映射到 out_channels
#
#         return out

class RegionConvSPP(nn.Module):
    def __init__(self, in_channels, out_channels, depth=3):
        super().__init__()
        channels = [16, 32, 64, 128, 256]
        layers = []
        for i in range(depth):
            cin = in_channels if i == 0 else channels[i - 1]
            cout = channels[i]
            layers += [
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]
        self.sequence_basic = nn.Sequential(*layers)

        self.pool_sizes = [1, 2, 4]
        self.fc = nn.Linear(channels[depth - 1] * sum([p * p for p in self.pool_sizes]), out_channels)

    def forward(self, x, mask):
        feat = self.sequence_basic(x)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask_down = F.interpolate(mask, size=feat.shape[2:], mode='nearest')
        feat = feat * mask_down

        spp_features = []
        for p in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(feat, output_size=(p, p))
            spp_features.append(pooled.view(feat.size(0), -1))
        spp_out = torch.cat(spp_features, dim=1)
        out = self.fc(spp_out)
        return out

class RegionConvSPP_Fuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 三层卷积通道设计
        self.channels = [16, 32, 64]

        # ===== 卷积层堆叠 =====
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[2], 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # ===== 融合后再做 SPP =====
        self.pool_sizes = [1, 2, 4]
        # 拼接后通道为 16 + 32 + 64 = 112
        fusion_channels = sum(self.channels)
        self.fc = nn.Linear(fusion_channels * sum([p * p for p in self.pool_sizes]), out_channels)

    def forward(self, x, mask):
        # 原始掩码下采样
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # 三层特征
        feat1 = self.conv1(x)
        mask1 = F.interpolate(mask, size=feat1.shape[2:], mode='nearest')
        feat1 = feat1 * mask1

        feat2 = self.conv2(feat1)
        mask2 = F.interpolate(mask, size=feat2.shape[2:], mode='nearest')
        feat2 = feat2 * mask2

        feat3 = self.conv3(feat2)
        mask3 = F.interpolate(mask, size=feat3.shape[2:], mode='nearest')
        feat3 = feat3 * mask3

        # ===== 多尺度特征融合 =====
        # 融合策略1：上采样到同一分辨率后拼接
        feat2_up = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        feat3_up = F.interpolate(feat3, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        feat_fused = torch.cat([feat1, feat2_up, feat3_up], dim=1)  # [B, 112, H, W]

        # ===== SPP层 =====
        spp_features = []
        for p in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(feat_fused, output_size=(p, p))
            spp_features.append(pooled.view(x.size(0), -1))
        spp_out = torch.cat(spp_features, dim=1)

        out = self.fc(spp_out)
        return out


if __name__ == "__main__":
    # 模拟数据
    inputData = torch.randn(size=(32, 1, 481, 411))

    # 区域定义以及掩码定义
    H, W = 481, 411
    mask_forehead = generate_mask(H, W, (20, 125), (135, 308))
    mask_left_cheek = generate_mask(H, W, (208, 51), (356, 146))
    mask_right_cheek = generate_mask(H, W, (208, 266), (356, 361))
    mask_nose = generate_mask(H, W, (207, 161), (304, 251))
    mask_jaw = generate_mask(H, W, (406, 146), (446, 266))

    # 特征提取与SPP
    region_extractor = RegionConvSPP(in_channels=1, out_channels=64, depth=3)
    region_feats_list = []

    feat_forhead = region_extractor(inputData, mask_forehead)
    feat_left_cheek = region_extractor(inputData, mask_left_cheek)
    feat_right_cheek = region_extractor(inputData, mask_right_cheek)
    feat_nose = region_extractor(inputData, mask_nose)
    feat_jaw = region_extractor(inputData, mask_jaw)

    region_feats = torch.stack(
        [feat_forhead, feat_left_cheek, feat_right_cheek, feat_nose, feat_jaw],
        dim=1
    )

    print(region_feats.size())
    print("you have done a good job!")
