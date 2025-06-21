import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float

from src.utils.helpers import set_chinese_font

set_chinese_font()

# 读取图像
img_path = '/home/kmh/ai/MCTS-Seg/data/ISIC2016/raw/test/image/ISIC_0000020.png'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = img_as_float(img_rgb)

# SLIC超像素分割
n_segments = 200  # 超像素数量
compactness = 10  # 紧凑性参数

start_time = time.time()
segments = slic(img_float, n_segments=n_segments, compactness=compactness, start_label=0)
elapsed = time.time() - start_time
print(f"SLIC超像素分割耗时: {elapsed:.4f} 秒")

# 可视化超像素边界
slic_vis = mark_boundaries(img_rgb, segments, color=(1, 0, 0))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'SLIC Superpixels (n={n_segments})')
plt.imshow(slic_vis)
plt.axis('off')

plt.tight_layout()
plt.show()