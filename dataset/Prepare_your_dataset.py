# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# 参数设置
image_dir = "../data/orange/images"
mask_dir = "../data/orange/masks"
save_path = "../data/orange"
img_size = (1024, 1024)
channels = 3
num_classes = 2

train_txt_path = '../data/orange/imageset/train.txt'
test_txt_path = '../data/orange/imageset/test.txt'
val_txt_path = '../data/orange/imageset/test.txt'

if train_txt_path is not None and test_txt_path is not None and val_txt_path is not None:
    def load_txt_list(txt_path):
        with open(txt_path, "r") as f:
            names = [line.strip() for line in f.readlines()]
        imgs = [os.path.join(image_dir, name+".png") for name in names]
        masks = [os.path.join(mask_dir, name+".png") for name in names]
        return imgs, masks

    train_imgs, train_masks = load_txt_list(train_txt_path)
    val_imgs, val_masks = load_txt_list(val_txt_path)
    test_imgs, test_masks = load_txt_list(test_txt_path)
else:
    # 读取所有图像路径
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    assert len(image_paths) == 263, f"图像数量应为263，当前为{len(image_paths)}"
    mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in image_paths]
    # 随机打乱
    indices = np.arange(len(image_paths))
    np.random.seed(42)
    np.random.shuffle(indices)

    image_paths = [image_paths[i] for i in indices]
    mask_paths = [mask_paths[i] for i in indices]

    # 按比例划分：211训练，26验证，26测试
    train_imgs = image_paths[:211]
    train_masks = mask_paths[:211]

    val_imgs = image_paths[211:237]
    val_masks = mask_paths[211:237]

    test_imgs = image_paths[237:]
    test_masks = mask_paths[237:]

# ----------------------
# Step 1：计算图像均值和标准差
# ----------------------
mean = np.zeros(3)
std = np.zeros(3)
pixel_count = 0

print("=> 正在计算图像均值和标准差...")
for path in tqdm(train_imgs, desc="统计训练集像素"):
    img = Image.open(path).convert("RGB").resize(img_size)
    img = np.asarray(img).astype(np.float32) / 255.0
    pixel_count += img.shape[0] * img.shape[1]
    for c in range(3):
        mean[c] += img[:, :, c].sum()
        std[c] += (img[:, :, c] ** 2).sum()

mean /= pixel_count
std = np.sqrt(std / pixel_count - mean ** 2)

print("图像均值:", mean)
print("图像标准差:", std)

# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]
# ----------------------
# Step 2：加载、标准化数据
# ----------------------
def load_dataset(img_paths, mask_paths, mean, std, num_classes):
    images = []
    masks = []
    for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths), desc="加载并标准化"):
        img = Image.open(img_path).convert("RGB").resize(img_size)
        img = np.asarray(img).astype(np.float32) / 255.0
        img = (img - mean) / std
        images.append(img)

        if num_classes == 2:
            mask = Image.open(mask_path).convert("L").resize(img_size)
            mask = np.asarray(mask)
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = Image.open(mask_path).resize(img_size)
            mask = np.asarray(mask)
            unique_vals = np.unique(mask)
            if np.any(unique_vals >= num_classes):
                raise ValueError(f"mask中存在大于等于 num_classes={num_classes} 的值: {unique_vals}")
        masks.append(mask)
    return np.array(images), np.array(masks)

print("=> 加载训练集...")
train_img_np, train_mask_np = load_dataset(train_imgs, train_masks, mean, std, num_classes)
print("=> 加载验证集...")
val_img_np, val_mask_np = load_dataset(val_imgs, val_masks, mean, std, num_classes)
print("=> 加载测试集...")
test_img_np, test_mask_np = load_dataset(test_imgs, test_masks, mean, std, num_classes)

# ----------------------
# Step 3：保存为 .npy 文件
# ----------------------
np.save(os.path.join(save_path, "data_train.npy"), train_img_np)
np.save(os.path.join(save_path, "data_val.npy"), val_img_np)
np.save(os.path.join(save_path, "data_test.npy"), test_img_np)
np.save(os.path.join(save_path, "mask_train.npy"), train_mask_np)
np.save(os.path.join(save_path, "mask_val.npy"), val_mask_np)
np.save(os.path.join(save_path, "mask_test.npy"), test_mask_np)

print("✅ 数据集划分与标准化完成！")
