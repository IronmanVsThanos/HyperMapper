import os
import numpy as np
from PIL import Image
import tqdm
from collections import defaultdict

# 设置文件夹路径
folder_path = '/data/DL/code/Rein-train-depth/data/tsp_back/gtFine/train_ori'

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 限制处理前100张图片
image_files = image_files[:100]

# 用于存储所有像素值的计数
pixel_counts = defaultdict(int)

# 处理每张图片
for img_file in tqdm.tqdm(image_files):
    img_path = os.path.join(folder_path, img_file)
    
    # 读取图片
    img = np.array(Image.open(img_path))
    
    # 统计像素值
    unique, counts = np.unique(img, return_counts=True)
    for val, count in zip(unique, counts):
        pixel_counts[val] += count

# 按像素值排序
sorted_pixels = sorted(pixel_counts.items(), key=lambda x: x[0])

print("\n所有像素值及其出现次数：")
print("像素值: 出现次数")
print("-" * 20)
for val, count in sorted_pixels:
    print(f"{val}: {count}")

print("\n总共有 {len(sorted_pixels)} 个不同的像素值")