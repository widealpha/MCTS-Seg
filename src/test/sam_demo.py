import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from segment_anything import SamPredictor, sam_model_registry
from utils.helpers import load_sam, device
matplotlib.use('TkAgg')

# 初始化SAM模型

sam = load_sam()
predictor = SamPredictor(sam)

# 加载图像
image_path = "/home/kmh/mcts/data/ependymoma/raw/train/image/sub002_4.png"
image = plt.imread(image_path)
# 统一转换为RGB格式
if image.ndim == 2:
    # 灰度图转RGB：复制三个通道
    image = np.stack([image] * 3, axis=-1)
elif image.shape[2] == 4:
    # RGBA转RGB：去除alpha通道
    image = image[..., :3]
elif image.shape[2] == 1:
    # 单通道伪灰度图转RGB
    image = np.concatenate([image] * 3, axis=-1)


# 初始化交互界面
fig, ax = plt.subplots()
ax.imshow(image)
points = []  # 存储点击坐标

# 点击事件处理函数
def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    
    # 记录点击坐标（注意坐标转换）
    x, y = int(event.xdata), int(event.ydata)
    points.append((x, y))
    
    # 转换输入格式
    input_point = np.array([[x, y]])
    input_label = np.array([1])  # 前景点
    
    # 生成预测
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # 显示结果（取第一个掩码）
    ax.clear()
    ax.imshow(image)
    ax.scatter(x, y, c='red', marker='*', s=200, edgecolor='white')
    ax.imshow(masks[0], alpha=0.5)
    plt.draw()

# 设置SAM图像编码
predictor.set_image(image)

# 绑定点击事件
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()