import torch
import os
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat, savemat
# import datasets.transformer as p
import datasets.transformer as transform
from math import isnan
import h5py


class StO2Dataset(Dataset):

    # 数据集初始化操作
    def __init__(self, root_path, train=True, test_subnum=0, transform=None):

        self.root_path = root_path
        self.train = train
        self.test_subnum = test_subnum
        self.transform = transform

        self.path_deception_baseline = []
        self.path_deception = []
        self.path_truth_baseline = []
        self.path_truth = []
        self.deception_label = []
        self.truth_label = []

        # 存储一下中间变量的
        self.path_all_temp = []
        self.path_baseline_all_temp = []
        self.label_all_temp = []

        # 存储最终变量的
        self.path_all = []
        self.path_baseline_all = []
        self.label_all = []

        self.path_all_train = []
        self.path_baseline_all_train = []
        self.label_all_train = []

        self.path_all_test = []
        self.path_baseline_all_test = []
        self.label_all_test = []

        # sub_dir 用于是几个主文件夹
        sub_dir = os.listdir(self.root_path)

        path_deception_baseline = self.root_path + '/' + sub_dir[0]
        path_deception = self.root_path + '/' + sub_dir[1]
        path_truth_baseline = self.root_path + '/' + sub_dir[2]
        path_truth = self.root_path + '/' + sub_dir[3]

        # 获取子目录下面的所有文件信息
        dir_mat_deception_baseline = os.listdir(path_deception_baseline)
        dir_mat_deception = os.listdir(path_deception)
        dir_mat_truth = os.listdir(path_truth)
        dir_mat_truth_baseline = os.listdir(path_truth_baseline)

        # 减一是有原因的，欺骗和诚实的数据长度是不一样的，但是欺骗与基线、诚实与基线长度是一样的
        # print("mat_path_deception_baseline的长度为： ", len(dir_mat_deception_baseline))
        # print("mat_path_truth_baseline的长度为： ", len(dir_mat_truth_baseline))

        # 对欺骗和欺骗基线下的所有mat矩阵进行处理
        for m in range(len(dir_mat_deception_baseline) - 1):
            # 对每一个文件夹下面所有数据处理
            deception_path = path_deception + "/" + dir_mat_deception[m]
            deception_baseline_path = path_deception_baseline + "/" + dir_mat_deception_baseline[m]

            self.path_deception.append(deception_path)
            self.path_deception_baseline.append(deception_baseline_path)
            self.deception_label.append([1])

        # 对诚实和诚实基线下的所有mat矩阵进行处理
        for m in range(len(dir_mat_truth_baseline) - 1):
            # 对每一个文件夹下面所有数据处理
            truth_path = path_truth + "/" + dir_mat_truth[m]
            truth_baseline_path = path_truth_baseline + "/" + dir_mat_truth_baseline[m]

            self.path_truth.append(truth_path)
            self.path_truth_baseline.append(truth_baseline_path)
            self.truth_label.append([0])

            # truth_mat_value = loadmat(truth_path)['newJregistered']
            # truth_baseline_mat_value = loadmat(truth_baseline_path)['newJregistered']

        # 如果test_subnum为100，表示获取文件夹下所有被试的数据，从而进行K折交叉验证
        if test_subnum == 100:
            self.path_all.extend(self.path_deception + self.path_truth)
            self.path_baseline_all.extend(self.path_deception_baseline + self.path_truth_baseline)
            self.label_all.extend(self.deception_label + self.truth_label)
        # 如果不是100，则根据test_subnum进行数据集划分
        else:
            self.path_all_temp.extend(self.path_deception + self.path_truth)
            self.path_baseline_all_temp.extend(self.path_deception_baseline + self.path_truth_baseline)
            self.label_all_temp.extend(self.deception_label + self.truth_label)

            for i in range(len(self.path_all_temp)):
                path = self.path_all_temp[i]
                file_name = os.path.basename(path)
                file_number = file_name.split("_")[0]
                file_number = int(file_number)

                if file_number != test_subnum:
                    self.path_all_train.append(path)
                    self.path_baseline_all_train.append(self.path_baseline_all_temp[i])
                    self.label_all_train.append(self.label_all_temp[i])
                else:
                    self.path_all_test.append(path)
                    self.path_baseline_all_test.append(self.path_baseline_all_temp[i])
                    self.label_all_test.append(self.label_all_temp[i])

    # 获取每一个元素
    def __getitem__(self, item):

        if self.train:
            with h5py.File(self.path_all_train[item], 'r') as mat_file:
                mat_value = mat_file['newJregistered'][:]
            with h5py.File(self.path_baseline_all_train[item], 'r') as mat_baseline_file:
                mat_baseline_value = mat_baseline_file['newJregistered'][:]

            temp_value = mat_value - mat_baseline_value
            label = self.label_all_train[item]
        else:
            with h5py.File(self.path_all_test[item], 'r') as mat_file:
                mat_value = mat_file['newJregistered'][:]
            with h5py.File(self.path_baseline_all_test[item], 'r') as mat_baseline_file:
                mat_baseline_value = mat_baseline_file['newJregistered'][:]

            temp_value = mat_value - mat_baseline_value
            label = self.label_all_test[item]

        # 去除nan值
        temp = 0
        weight = temp_value.shape[0]
        height = temp_value.shape[1]
        for i in range(0, weight):
            for j in range(0, height):
                # 这个地方进行RIO区域StO2数量统计时需要重新考虑
                temp += 1
                # 是nan值，则不进行相加
                if isnan(temp_value[i, j]):
                    temp_value[i, j] = 0

        # 对原始的输入数据做数据增强以及转变为增量
        if self.transform:
            temp_value = self.transform(temp_value)

        # 标签转tensor，且大小为N
        label = torch.as_tensor(label, dtype=torch.long)

        return temp_value, label

    # 数据集长度
    def __len__(self):

        if self.train:
            return len(self.path_all_train)
        else:
            return len(self.path_all_test)

if __name__ == "__main__":

    root_path = r"D:\Users\12150\PycharmProjects\data-sub1-sub14-all\hd5"
    clsdataset = StO2Dataset(root_path, train=False, test_subnum=1, transform=transform.transform())
    print("len(clsdataset) = ", len(clsdataset))
    for img, label in clsdataset:
        print("img.shape = {}; label = {}".format(img.shape, label))
