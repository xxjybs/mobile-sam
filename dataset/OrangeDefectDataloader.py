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


class DefectHoleSimulation:
    """
    数据增强：将大块连续缺陷模拟成许多小缺陷
    对于大块缺陷区域：用正常表皮覆盖，然后挖小孔暴露部分原始缺陷
    对于小缺陷（小孔）：保留原来的缺陷
    
    Args:
        p: 应用此增强的概率
        large_defect_threshold: 大缺陷面积阈值（像素数），超过此值认为是大缺陷
        hole_ratio: 挖孔占大缺陷区域的比例范围 (min, max)
        hole_size_range: 单个小孔的大小范围（像素）(min, max)
        num_holes_range: 挖孔数量范围 (min, max)
    """
    def __init__(self, p=0.3, large_defect_threshold=500, hole_ratio=(0.1, 0.3),
                 hole_size_range=(3, 15), num_holes_range=(5, 20)):
        self.p = p
        self.large_defect_threshold = large_defect_threshold
        self.hole_ratio = hole_ratio
        self.hole_size_range = hole_size_range
        self.num_holes_range = num_holes_range

    def _get_normal_skin_texture(self, img, mask):
        """从图像中提取正常表皮区域的纹理"""
        mask_2d = mask[:, :, 0] if len(mask.shape) == 3 else mask
        normal_mask = (mask_2d == 0)
        
        if not np.any(normal_mask):
            return None
        
        # 获取正常区域的像素
        normal_pixels = img[normal_mask]
        return normal_pixels

    def _fill_with_normal_texture(self, img, defect_mask, normal_pixels):
        """用正常表皮纹理填充缺陷区域"""
        if normal_pixels is None or len(normal_pixels) == 0:
            return img
        
        filled_img = img.copy()
        defect_coords = np.where(defect_mask)
        num_defect_pixels = len(defect_coords[0])
        
        if num_defect_pixels == 0:
            return filled_img
        
        # 随机采样正常像素来填充缺陷区域
        random_indices = np.random.randint(0, len(normal_pixels), num_defect_pixels)
        filled_img[defect_coords[0], defect_coords[1]] = normal_pixels[random_indices]
        
        return filled_img

    def _dig_holes(self, defect_mask, num_holes, hole_size_range):
        """在缺陷区域内挖小孔"""
        holes_mask = np.zeros_like(defect_mask, dtype=bool)
        defect_coords = np.where(defect_mask)
        
        if len(defect_coords[0]) == 0:
            return holes_mask
        
        for _ in range(num_holes):
            # 随机选择一个缺陷区域内的点作为孔中心
            idx = np.random.randint(0, len(defect_coords[0]))
            center_y, center_x = defect_coords[0][idx], defect_coords[1][idx]
            
            # 随机确定孔的大小
            hole_radius = np.random.randint(hole_size_range[0], hole_size_range[1] + 1)
            
            # 创建圆形孔
            y, x = np.ogrid[:defect_mask.shape[0], :defect_mask.shape[1]]
            dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            hole = dist_from_center <= hole_radius
            
            # 只在原缺陷区域内挖孔
            holes_mask |= (hole & defect_mask)
        
        return holes_mask

    def __call__(self, img, mask):
        if random.random() > self.p:
            return img, mask
        
        original_img = img.copy()  # 保存原始图像用于恢复小孔中的缺陷
        img = img.copy()
        mask = mask.copy()
        mask_2d = mask[:, :, 0] if len(mask.shape) == 3 else mask
        
        # 寻找连通的缺陷区域
        defect_binary = (mask_2d > 0).astype(np.int32)
        labeled_array, num_features = ndimage.label(defect_binary)
        
        if num_features == 0:
            return img, mask
        
        # 获取正常表皮纹理
        normal_pixels = self._get_normal_skin_texture(img, mask)
        
        new_mask = np.zeros_like(mask_2d)
        
        for region_id in range(1, num_features + 1):
            region_mask = (labeled_array == region_id)
            region_area = np.sum(region_mask)
            
            if region_area > self.large_defect_threshold:
                # 大缺陷：用正常表皮覆盖，然后挖小孔
                img = self._fill_with_normal_texture(img, region_mask, normal_pixels)
                
                # 计算要挖的孔数量
                num_holes = np.random.randint(self.num_holes_range[0], self.num_holes_range[1] + 1)
                # 根据缺陷区域大小调整孔的数量
                num_holes = min(num_holes, int(region_area / 50))
                num_holes = max(num_holes, 1)
                
                # 挖小孔
                holes_mask = self._dig_holes(region_mask, num_holes, self.hole_size_range)
                
                # 在小孔位置恢复原始缺陷图像
                img[holes_mask] = original_img[holes_mask]
                # 更新mask，只有小孔位置是缺陷
                new_mask[holes_mask] = 1
            else:
                # 小缺陷：保留原来的缺陷
                new_mask[region_mask] = 1
        
        # 更新mask
        if len(mask.shape) == 3:
            mask[:, :, 0] = new_mask
        else:
            mask = new_mask
        
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
                ColorAug(brightness=0.3, contrast=0.2, saturation=0.2, p=0.2),
                DefectHoleSimulation(p=0.3, large_defect_threshold=500)
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