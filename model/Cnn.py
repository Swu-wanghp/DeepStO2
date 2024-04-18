import torch
import torch.nn as nn
import torch.nn.functional as F


# class Cnn6(nn.Module):
#
#     def __init__(self):
#         super(Cnn6, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
#         # self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
#
#         # self.fc1 = nn.Linear(143016, 120)
#         self.fc2 = nn.Linear(292536, 256)
#         self.fc3 = nn.Linear(in_features=256, out_features=2)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square you can only specify a single number
#         # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         # 特征图转换为一个１维的向量
#         x = x.view(-1, self.num_flat_features(x))
#         # x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features

class Cnn6(nn.Module):

    def __init__(self):
        super(Cnn6, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)

        self.fc1 = nn.Linear(143016, 120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=2)

        # 使用 Xavier 初始化方法对线性层的参数进行随机初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 特征图转换为一个１维的向量
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features





if __name__ == "__main__":
    inputTest = torch.rand((4, 1, 481, 411))
    model = Cnn6()
    output = model(inputTest)
    print("output = {0}, size = {1}".format(output, output.shape))
