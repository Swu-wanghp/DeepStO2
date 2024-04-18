# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import torch.nn.functional as F
import torch
import os
import h5py
import scipy.io
from scipy.io import loadmat

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def convert_mat_to_hdf5(mat_file, hdf5_file):
    # 加载MATLAB文件
    mat_data = loadmat(mat_file)

    # 创建HDF5文件
    with h5py.File(hdf5_file, 'w') as f:
        # 遍历MATLAB数据中的每一个变量并写入HDF5文件
        for var_name in mat_data:
            f.create_dataset(var_name, data=mat_data[var_name])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path_all = []
    path_baseline_all = []
    path_deception_list = []
    path_deception_baseline_list = []
    path_truth_list = []
    path_truth_baseline_list = []

    root_path = r"D:\Users\12150\PycharmProjects\data-sub1-sub14-all\mat"
    sub_dir = os.listdir(root_path)

    path_deception_baseline = root_path + '/' + sub_dir[0]
    path_deception = root_path + '/' + sub_dir[1]
    path_truth_baseline = root_path + '/' + sub_dir[2]
    path_truth = root_path + '/' + sub_dir[3]

    # 获取子目录下面的所有文件信息
    dir_mat_deception_baseline = os.listdir(path_deception_baseline)
    dir_mat_deception = os.listdir(path_deception)
    dir_mat_truth = os.listdir(path_truth)
    dir_mat_truth_baseline = os.listdir(path_truth_baseline)

    # 减一是有原因的，欺骗和诚实的数据长度是不一样的，但是欺骗与基线、诚实与基线长度是一样的
    print("mat_path_deception_baseline的长度为： ", len(dir_mat_deception_baseline))
    print("mat_path_truth_baseline的长度为： ", len(dir_mat_truth_baseline))

    # 对欺骗和欺骗基线下的所有mat矩阵进行处理
    for m in range(len(dir_mat_deception_baseline) - 1):
        # 对每一个文件夹下面所有数据处理
        deception_path = path_deception + "/" + dir_mat_deception[m]
        deception_baseline_path = path_deception_baseline + "/" + dir_mat_deception_baseline[m]

        path_deception_list.append(deception_path)
        path_deception_baseline_list.append(deception_baseline_path)


    # 对诚实和诚实基线下的所有mat矩阵进行处理
    for m in range(len(dir_mat_truth_baseline) - 1):
        # 对每一个文件夹下面所有数据处理
        truth_path = path_truth + "/" + dir_mat_truth[m]
        truth_baseline_path = path_truth_baseline + "/" + dir_mat_truth_baseline[m]

        path_truth_list.append(truth_path)
        path_truth_baseline_list.append(truth_baseline_path)



    # for i in range(len(path_deception_list)):
    #     # 读取MATLAB文件
    #     filename = path_deception_list[i]
    #
    #     # 创建HDF5文件
    #     if filename.endswith('.mat'):
    #         hdf5_file = filename.replace(".mat", ".h5")
    #         convert_mat_to_hdf5(filename, hdf5_file)

    for i in range(len(path_deception_baseline_list)):
        # 读取MATLAB文件
        filename = path_deception_baseline_list[i]

        # 创建HDF5文件
        if filename.endswith('.mat'):
            hdf5_file = filename.replace(".mat", ".h5")
            convert_mat_to_hdf5(filename, hdf5_file)

    for i in range(len(path_truth_list)):
        # 读取MATLAB文件
        filename = path_truth_list[i]

        # 创建HDF5文件
        if filename.endswith('.mat'):
            hdf5_file = filename.replace(".mat", ".h5")
            convert_mat_to_hdf5(filename, hdf5_file)

    for i in range(len(path_truth_baseline_list)):
        # 读取MATLAB文件
        filename = path_truth_baseline_list[i]

        # 创建HDF5文件
        if filename.endswith('.mat'):
            hdf5_file = filename.replace(".mat", ".h5")
            convert_mat_to_hdf5(filename, hdf5_file)


# byteord