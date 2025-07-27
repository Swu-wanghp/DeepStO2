import torch
import os
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat, savemat
import torchvision.transforms as transforms

class StO2Dataset(Dataset):

    def __init__(self, root_path, train=True, test_subnum=0, transform=None):
        # 初始化参数
        self.root_path = root_path
        self.train = train
        self.test_subnum = test_subnum

        # 定义默认转换流程
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

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
        for m in range(len(dir_mat_deception_baseline)):
            # 对每一个文件夹下面所有数据处理
            deception_path = path_deception + "/" + dir_mat_deception[m]
            deception_baseline_path = path_deception_baseline + "/" + dir_mat_deception_baseline[m]

            self.path_deception.append(deception_path)
            self.path_deception_baseline.append(deception_baseline_path)
            self.deception_label.append([1])

        # 对诚实和诚实基线下的所有mat矩阵进行处理
        for m in range(len(dir_mat_truth_baseline)):
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
        # 预拟合Scaler（仅在训练模式）
        # if train and not os.path.exists(self.scaler_path):
        #     self._fit_scaler()

    def __getitem__(self, item):
        # 加载数据
        if self.train:
            # 加载三个区域的数据
            data_paths = self.path_all_train[item * 3: item * 3 + 3]
            baseline_paths = self.path_baseline_all_train[item * 3: item * 3 + 3]
            labels = self.label_all_train[item * 3]

        else:
            data_paths = self.path_all_test[item * 3: item * 3 + 3]
            baseline_paths = self.path_baseline_all_test[item * 3: item * 3 + 3]
            labels = self.label_all_test[item * 3]

        region_tensors = []
        for data_path, baseline_path in zip(data_paths, baseline_paths):
            region = self.load_and_process_region(data_path, baseline_path)
            region_tensor = torch.from_numpy(region).float()
            region_tensors.append(region_tensor)

        label = torch.as_tensor(labels, dtype=torch.long).squeeze()

        return region_tensors, label

    # 加载区域数据
    def load_and_process_region(self, data_path, baseline_path):

        # 读取npy文件
        data = np.load(data_path)
        baseline = np.load(baseline_path)

        # 插值
        temp_value = data - baseline

        # 替换NaN值为全局最小值 （没有NaN值了）
        global_min = np.nanmin(temp_value)
        temp_value[np.isnan(temp_value)] = global_min

        # 归一化 （使用训练拟合的Scalar）
        temp_value = temp_value.astype(np.float32)

        # 应用 transform
        if self.transform:
            # temp_value = (temp_value * 255).astype(np.uint8)  # 缩放到 0~255
            temp_value = self.transform(temp_value)

        return temp_value

    # 返回样本量数据
    def __len__(self):

        if self.train:
            return len(self.path_all_train) // 3
        else:
            return len(self.path_all_test) // 3


if __name__ == "__main__":
    # 定义转换流程
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 创建数据集实例
    root_path = r"D:\论文\StO2数据\最后mat数据\hd5_for_train_npyfile"
    train_dataset = StO2Dataset(
        root_path,
        train=True,
        test_subnum=43,
        transform=None
    )

    test_dataset = StO2Dataset(
        root_path,
        train=False,
        test_subnum=43,
        transform=None
    )

    # 验证输出
    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    for img, label in train_dataset:
        print("Processed image shape:", img[0].shape)  # 额头区域
        print("Processed image shape:", img[1].shape)  # 左脸颊区域
        print("Processed image shape:", img[2].shape)  # 右脸颊区域
        print("Processed image' label shape:", label)  # 额头区域


