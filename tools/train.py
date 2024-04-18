import os
import tqdm
import time
import shutil
import torch

import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from utils import parser
from model import Cnn6
from model import MobileNetV2
from datasets import StO2Dataset, transformer
from torch.utils.data import Dataset, DataLoader


# 进行LOSO训练与测试
def train(agrs):
    # 用于计算准确率平均值
    averageAll = 0
    # 对每一个被试进行训练
    for p in range(args.sub_num):

        # 初始化变量， 暂时存储最大值，训练损失函数
        temp_max = 0
        train_loss = 0

        print("当前遍历第{}个人".format(p + 1))
        train_dataset = StO2Dataset(agrs.root_path, train=True, test_subnum=p + 1, transform=transformer.transform())
        test_datast = StO2Dataset(agrs.root_path, train=False, test_subnum=p + 1, transform=transformer.transform())
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batchsize, shuffle=True)
        val_loader = DataLoader(dataset=test_datast, batch_size=args.test_batchsize, shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Cnn6()
        model.to(device)
        model.zero_grad()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.9)

        # 每个被试下epoch
        for k in range(args.train_epoch):
            model.train()
            for batch in train_loader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                images = images.to(torch.float)
                labels = torch.squeeze(labels, dim=1)
                optimizer.zero_grad()
                preds = model(images)
                loss = F.cross_entropy(preds, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print("epoch: {} | loss = {}".format(k + 1, loss))

                train_loss += loss.item()

            # 每次运行完，进行一次验证
            model.eval()
            with torch.no_grad():
                preds, labels = [], []
                for batchTest in val_loader:
                    imagesTest, labelsTest = batchTest
                    imagesTest = imagesTest.to(torch.float)
                    labelsTest = torch.squeeze(labelsTest, dim=1)

                    output = model(imagesTest)
                    predict = torch.max(output, dim=1)[1]

                    labels.append(labelsTest)
                    preds.append(predict)
                labels = torch.cat(labels, dim=0)
                preds = torch.cat(preds, dim=0)
                labels = labels.cpu().numpy()
                preds = preds.cpu().numpy()
                eval_result = (np.sum(labels == preds)) / len(labels)
                print("第{}个被试的第{}周期下的预测结果为{}".format(p + 1, k + 1, eval_result))
                # 在一个epoch下，如果acc值大于当前的，则保存该模型
                filename = str(p + 1) + str(k+1) + ".pth"
                save_path = os.path.join(args.output, filename)
                # os.makedirs(save_path, exist_ok=True)
                # model.module用于判断是否使用分布式的
                model_to_save = (model.module if hasattr(model, "module") else model)
                if eval_result > temp_max:
                    temp_max = eval_result
                    torch.save(model_to_save.state_dict(), save_path)

        print("第{}个被试作为预测情况下，所有训练的最大值为{}".format(p + 1, temp_max))
        averageAll += temp_max
        print("所有被试的真实和欺骗识别率为{}".format(averageAll))

        # 用于输出标签，从而验证结果
        # print("labels = ", labels)
        # print("preds = ", preds)
        # print("results = ", eval_result)

    # 输出LOSO的均值
    print("eval_result = ", averageAll / agrs.sub_num)


if __name__ == "__main__":
    args = parser.parse_args()

    # 初始化训练的GPU设备

    # 可以开始打印一下主要的参数信息

    train(args)
