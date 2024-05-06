from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torchvision

torchvision.models.densenet121()
# 创建一个三维图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维图形
...

# 激活鼠标旋转工具
ax.mouse_init()

# 显示图形
plt.show()
