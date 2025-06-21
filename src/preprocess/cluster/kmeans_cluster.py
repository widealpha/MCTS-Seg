import cv2
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import time

from src.utils.helpers import set_chinese_font

set_chinese_font()
high_contrast_colors = np.array([
    [255, 0, 0],      # 红
    [0, 255, 0],      # 绿
    [0, 0, 255],      # 蓝
    [255, 255, 255],  # 白
    [255, 255, 0],    # 黄
    [0, 255, 255],    # 青
    [255, 0, 255],    # 紫
    [255, 128, 0],    # 橙
    [0, 0, 0],        # 黑
], dtype=np.uint8)

# 读取图像
img_path = '/home/kmh/ai/MCTS-Seg/data/ISIC2016/raw/test/image/ISIC_0000020.png'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 图像数据预处理
pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 设置聚类数
k = 4
palette = high_contrast_colors[:k]

def cluster_with_time(pixel_values, kmeans):
    start_time = time.time()
    labels = kmeans.fit_predict(pixel_values)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"KMeans聚类耗时: {elapsed:.4f} 秒")
    return labels

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = cluster_with_time(pixel_values, kmeans)
centers = np.uint8(kmeans.cluster_centers_)

# 生成聚类后的图像
segmented_img = palette[labels.flatten()]
segmented_img = segmented_img.reshape(img_rgb.shape)

# 展示原图和聚类结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'KMeans Clustering (k={k})')
plt.imshow(segmented_img)
plt.axis('off')

plt.tight_layout()
plt.show()