import numpy as np
import random
from scipy import ndimage
import torch
from torch.utils.data import Dataset, DataLoader

class Resize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, mask):
        img = ndimage.zoom(img, (self.height / img.shape[0], self.width / img.shape[1], 1), order=1)
        mask = ndimage.zoom(mask, (self.height / mask.shape[0], self.width / mask.shape[1], 1), order=0)
        return img, mask

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
        return img, mask

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            img = np.flip(img, axis=0)
            mask = np.flip(mask, axis=0)
        return img, mask

class RandomRotation:
    def __init__(self, p=0.5, degree=(0, 360)):
        self.p = p
        self.degree = degree
    def __call__(self, img, mask):
        if random.random() < self.p:
            angle = random.uniform(*self.degree)
            img = ndimage.rotate(img, angle, reshape=False, order=1, mode='reflect')
            mask = ndimage.rotate(mask, angle, reshape=False, order=0, mode='nearest')
        return img, mask

class ColorAug:
    def __init__(self, brightness=0.3, contrast=0.2, saturation=0.2, p=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            # 亮度
            img = img * (1 + random.uniform(-self.brightness, self.brightness))
            # 对比度
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = (img - mean) * (1 + random.uniform(-self.contrast, self.contrast)) + mean
            # 饱和度 (仅适用于彩色图像)
            gray = img.mean(axis=2, keepdims=True)
            img = (img - gray) * (1 + random.uniform(-self.saturation, self.saturation)) + gray
            img = np.clip(img, 0, 1)
        return img, mask

class OrangeDefectLoader(Dataset):
    def __init__(self, data_dir, train=True, test=False, size=1024, num_classes=2):
        self.data_dir = data_dir
        self.train = train
        self.test = test
        self.size = size
        self.num_classes = num_classes

        if train:
            self.imgs = np.load(f"{data_dir}/data_train.npy")
            self.masks = np.load(f"{data_dir}/mask_train.npy")
        elif test:
            self.imgs = np.load(f"{data_dir}/data_test.npy")
            self.masks = np.load(f"{data_dir}/mask_test.npy")
        else:
            self.imgs = np.load(f"{data_dir}/data_val.npy")
            self.masks = np.load(f"{data_dir}/mask_val.npy")

        # mask: [H, W] -> [H, W, 1]
        if len(self.masks.shape) == 3:
            self.masks = np.expand_dims(self.masks, axis=-1)

        # 设置增强
        if self.train:
            self.aug_transforms = [
                Resize(size, size),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomRotation(p=0.5, degree=(0, 360)),
                ColorAug(brightness=0.3, contrast=0.2, saturation=0.2, p=0.2)
            ]
        else:
            self.aug_transforms = [
                Resize(size, size)
            ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]    # float32, normalized [H, W, 3]
        mask = self.masks[idx]  # uint8, [H, W, 1]

        # Apply transforms
        for aug in self.aug_transforms:
            img, mask = aug(img, mask)

        # 转换为 torch tensor
        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.copy()).squeeze(2).long()  # [H, W]

        # 计算 one-hot 编码 [num_classes, H, W]
        one_hot = torch.nn.functional.one_hot(mask, num_classes=self.num_classes).permute(2, 0, 1).float()

        return img, mask, one_hot


if __name__ == '__main__':
    trainset = OrangeDefectLoader("../data/orange", train=False, test=False, size=224, num_classes=2)
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

    for img, mask, onehot in trainloader:
        print(img.shape, mask.shape, onehot.shape)
        print(mask.max(), mask.min())
        print(onehot.max(), onehot.min())
        break