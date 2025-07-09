import os
import cv2
import random

img_dir = '/home/yjr/下载/rknn_model_zoo-2.1.0/examples/yolov8/model/tomato'
output_txt = './coco_subset_20.txt'

valid_images = []
for fname in sorted(os.listdir(img_dir)):
    if fname.endswith('.jpg'):
        fpath = os.path.join(img_dir, fname)
        try:
            img = cv2.imread(fpath)
            if img is not None:
                valid_images.append(fpath)
        except:
            continue

print(f"Found {len(valid_images)} valid images.")

# 随机选取前100张（或所有有效图像）
selected = random.sample(valid_images, min(100, len(valid_images)))

with open(output_txt, 'w') as f:
    for path in selected:
        f.write(path + '\n')

print("✅ 写入完成: coco_subset_20.txt")