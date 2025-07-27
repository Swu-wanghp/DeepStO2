## 对mat数据进行预处理，然后转为.npy数据

import os
import h5py
import shutil
import numpy as np
import matplotlib.pyplot as plt
# 读取一个mat文件，并且分块
def raad_h5_and_divide_block(filePath, left_cheek, right_cheek, forehead):
    with h5py.File(filePath, 'r') as mat_file:
        mat_value = mat_file['newJregistered'][:]

        print(mat_value)

        # 提取各区域坐标
        def extract_block(region):
            top_left = region["Top_left_corner"]
            bottom_right = region["Bottom_right_corner"]
            return mat_value[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]  # 注意先y后x

        # 提取区域
        left_cheek_block = extract_block(left_cheek)
        right_cheek_block = extract_block(right_cheek)
        forehead_block = extract_block(forehead)

        # 构造输出目录
        out_dir = os.path.dirname(filePath)
        base_name = os.path.splitext(os.path.basename(filePath))[0]

        # 保存为npy
        np.save(os.path.join(out_dir, f"{base_name}_left_cheek.npy"), left_cheek_block)
        np.save(os.path.join(out_dir, f"{base_name}_right_cheek.npy"), right_cheek_block)
        np.save(os.path.join(out_dir, f"{base_name}_forehead.npy"), forehead_block)

        print("区域数据已保存为 .npy 文件")

# 可视化npy文件，并且判断数据中是否出现了NaN值
def vis_npy_file(filepath):
    # 读取 npy 文件
    data = np.load(filepath)

    # 打印形状和部分数据
    print("数据形状:", data.shape)
    print("部分内容:", data[:5])  # 打印前5行

    # 检查是否包含 NaN
    has_nan = np.isnan(data).any()
    print("是否包含 NaN:", has_nan)

    if has_nan:
        nan_count = np.isnan(data).sum()
        print("NaN 总数:", nan_count)

# 把新生成的npy文件挑选出来
def move_npy_files(src_root, dst_root):
    # 创建目标根目录（如果不存在）
    os.makedirs(dst_root, exist_ok=True)

    for subfolder in os.listdir(src_root):
        src_subfolder = os.path.join(src_root, subfolder)

        # 只处理文件夹
        if not os.path.isdir(src_subfolder):
            continue

        dst_subfolder = os.path.join(dst_root, subfolder)
        os.makedirs(dst_subfolder, exist_ok=True)

        for file in os.listdir(src_subfolder):
            if file.endswith('.npy'):
                src_file_path = os.path.join(src_subfolder, file)
                dst_file_path = os.path.join(dst_subfolder, file)

                # 移动文件
                shutil.move(src_file_path, dst_file_path)
                print(f"移动文件: {src_file_path} --> {dst_file_path}")

if __name__ == "__main__":

    # 左脸颊，右脸颊，额头 (51, 207), (146, 356) // (266, 207), (361, 356) // (125, 10), (308, 135) 后期要修改
    filepath = r"D:\论文\StO2数据\最后mat数据\DeepStO2\datasets\01_S1_D1.h5"
    left_cheek = {
        "Top_left_corner": [51, 208],
        "Bottom_right_corner": [146, 356]
    }
    right_cheek = {
        "Top_left_corner": [266, 208],
        "Bottom_right_corner": [361, 356]
    }
    forehead = {
        "Top_left_corner": [125, 20],
        "Bottom_right_corner": [308, 135]
    }
    # 一个文件画块
    # raad_h5_and_divide_block(filepath, left_cheek, right_cheek, forehead)

    # 可视化npy文件的
    # npyfilepath = r"D:\论文\StO2数据\最后mat数据\DeepStO2\datasets\01_S1_D1_right_cheek.npy"
    # vis_npy_file(npyfilepath)

    # 根目录 D:\论文\StO2数据\最后mat数据\hd5_for_train
    rootPath = r"D:\论文\StO2数据\最后mat数据\hd5_for_train"
    saverootPath = r"D:\论文\StO2数据\最后mat数据\hd5_for_train_npyfile"

    # subfile = os.listdir(rootPath)
    # print(subfile)
    # for i in range(len(subfile)):
    #     subpath = os.path.join(rootPath, subfile[i])
    #
    #     for file in os.listdir(subpath):
    #         file_path = os.path.join(subpath, file)
    #         if os.path.isfile(file_path):
    #             print("找到文件:", file_path)
    #
    #             raad_h5_and_divide_block(file_path, left_cheek, right_cheek, forehead)


    # 挑选.npy数据
    move_npy_files(rootPath, saverootPath)

