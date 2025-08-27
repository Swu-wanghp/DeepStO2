# import os.path
#
# import numpy as np
#
# path = "D:\论文\StO2数据\最后mat数据\DeepStO2\model"
# savepath = os.path.join(path, "adj.npy")
# # 示例：创建一个包含 1 到 5 的数组
# adj = np.array([[1, 2, 3],
#                 [1, 2, 3],
#                 [1, 2, 3]])
#
# # 将 my_array 保存到名为 'my_array.npy' 的文件中
# np.save(savepath, adj)
#
# data = np.load(savepath)
# print(data)

import scipy.io as scio
import matplotlib.pyplot as plt
dataFile = r'D:\论文\StO2数据\最后mat数据\DeepStO2\datasets\keypoint.mat'
data = scio.loadmat(dataFile)
print("11")

# 打印 key 看看里面有哪些变量
print("Keys in mat file:", data.keys())

# 假设图像数据存储在 'img' 变量里（请把 'img' 替换成你的实际 key）
img = data['keypoint']

print("Shape of img:", img.shape)

print("img.value = ", img)