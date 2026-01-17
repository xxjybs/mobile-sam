"""
后处理模块：用于优化分割结果，特别是处理密集连续大块缺陷的闭合区域
Post-processing module: Optimize segmentation results, especially for dense/continuous large defect closed regions
"""

import numpy as np
import cv2
from PIL import Image


def morphological_close(mask, kernel_size=5, iterations=1):
    """
    形态学闭操作：先膨胀后腐蚀，用于填充小孔洞和连接相邻区域
    Morphological closing: dilate then erode, to fill small holes and connect adjacent regions
    
    Args:
        mask: 二值掩码 (H, W)，值为0或255
        kernel_size: 结构元素大小
        iterations: 迭代次数
    
    Returns:
        处理后的掩码
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def fill_holes(mask):
    """
    填充闭合区域内的孔洞
    Fill holes inside closed regions
    
    Args:
        mask: 二值掩码 (H, W)，值为0或255
    
    Returns:
        填充后的掩码
    """
    # 使用floodFill从边界开始填充背景
    # Use floodFill to fill background starting from boundary
    mask_copy = mask.copy()
    h, w = mask.shape
    
    # 创建一个比原图大2个像素的掩码
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # 从左上角开始填充背景
    cv2.floodFill(mask_copy, flood_mask, (0, 0), 255)
    
    # 反转填充后的图像
    mask_inv = cv2.bitwise_not(mask_copy)
    
    # 与原图做或运算，得到填充孔洞后的结果
    filled = cv2.bitwise_or(mask, mask_inv)
    
    return filled


def remove_small_regions(mask, min_area=100, connectivity=8):
    """
    移除小于指定面积的区域（噪声过滤）
    Remove regions smaller than specified area (noise filtering)
    
    Args:
        mask: 二值掩码 (H, W)，值为0或255
        min_area: 最小区域面积阈值
        connectivity: 连通性 (4 or 8)
    
    Returns:
        过滤后的掩码
    """
    # 确保掩码是二值的
    binary = (mask > 127).astype(np.uint8) * 255
    
    # 查找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )
    
    # 创建输出掩码
    result = np.zeros_like(mask)
    
    # 跳过背景(label 0)，保留面积大于阈值的区域
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            result[labels == i] = 255
    
    return result


def connect_nearby_regions(mask, max_gap=10):
    """
    连接相邻的区域（使用膨胀-腐蚀序列）
    Connect nearby regions using dilation-erosion sequence
    
    Args:
        mask: 二值掩码 (H, W)，值为0或255
        max_gap: 最大间隙距离（像素）
    
    Returns:
        连接后的掩码
    """
    kernel_size = max_gap + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 先膨胀连接相邻区域
    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    # 再腐蚀恢复原始大小（但保持连接）
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # 使用闭操作进一步平滑
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return closed


def apply_postprocess(mask, 
                      enable_closing=True,
                      closing_kernel_size=5,
                      closing_iterations=1,
                      enable_hole_fill=True,
                      enable_small_region_removal=True,
                      min_region_area=100,
                      enable_region_connection=False,
                      connection_max_gap=10):
    """
    应用后处理流水线
    Apply post-processing pipeline
    
    Args:
        mask: 输入掩码，可以是numpy数组(H,W)或PIL Image
        enable_closing: 是否启用形态学闭操作
        closing_kernel_size: 闭操作核大小
        closing_iterations: 闭操作迭代次数
        enable_hole_fill: 是否启用孔洞填充
        enable_small_region_removal: 是否启用小区域移除
        min_region_area: 最小区域面积阈值
        enable_region_connection: 是否启用区域连接
        connection_max_gap: 区域连接的最大间隙
    
    Returns:
        处理后的掩码（与输入类型相同）
    """
    # 处理输入类型
    is_pil = isinstance(mask, Image.Image)
    if is_pil:
        mask_np = np.array(mask.convert('L'))
    else:
        mask_np = mask.copy()
    
    # 确保是二值图像
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    
    # 1. 形态学闭操作 - 填充小孔洞
    if enable_closing:
        mask_np = morphological_close(mask_np, closing_kernel_size, closing_iterations)
    
    # 2. 填充闭合区域内的孔洞
    if enable_hole_fill:
        mask_np = fill_holes(mask_np)
    
    # 3. 连接相邻区域
    if enable_region_connection:
        mask_np = connect_nearby_regions(mask_np, connection_max_gap)
    
    # 4. 移除小区域（噪声过滤）
    if enable_small_region_removal:
        mask_np = remove_small_regions(mask_np, min_region_area)
    
    # 返回与输入相同的类型
    if is_pil:
        return Image.fromarray(mask_np)
    return mask_np


class PostProcessor:
    """
    后处理器类，便于配置和复用
    Post-processor class for easy configuration and reuse
    """
    
    def __init__(self,
                 enable_closing=True,
                 closing_kernel_size=5,
                 closing_iterations=1,
                 enable_hole_fill=True,
                 enable_small_region_removal=True,
                 min_region_area=100,
                 enable_region_connection=False,
                 connection_max_gap=10):
        """
        初始化后处理器参数
        Initialize post-processor parameters
        """
        self.enable_closing = enable_closing
        self.closing_kernel_size = closing_kernel_size
        self.closing_iterations = closing_iterations
        self.enable_hole_fill = enable_hole_fill
        self.enable_small_region_removal = enable_small_region_removal
        self.min_region_area = min_region_area
        self.enable_region_connection = enable_region_connection
        self.connection_max_gap = connection_max_gap
    
    def __call__(self, mask):
        """
        应用后处理
        Apply post-processing
        """
        return apply_postprocess(
            mask,
            enable_closing=self.enable_closing,
            closing_kernel_size=self.closing_kernel_size,
            closing_iterations=self.closing_iterations,
            enable_hole_fill=self.enable_hole_fill,
            enable_small_region_removal=self.enable_small_region_removal,
            min_region_area=self.min_region_area,
            enable_region_connection=self.enable_region_connection,
            connection_max_gap=self.connection_max_gap
        )
    
    @classmethod
    def for_large_defects(cls):
        """
        预设配置：针对大块密集缺陷优化
        Preset configuration: optimized for large dense defects
        """
        return cls(
            enable_closing=True,
            closing_kernel_size=7,
            closing_iterations=2,
            enable_hole_fill=True,
            enable_small_region_removal=True,
            min_region_area=200,
            enable_region_connection=True,
            connection_max_gap=15
        )
    
    @classmethod
    def for_edge_defects(cls):
        """
        预设配置：针对边缘小缺陷（保守处理，减少形态学操作）
        Preset configuration: for small edge defects (conservative, less morphological operations)
        """
        return cls(
            enable_closing=True,
            closing_kernel_size=3,
            closing_iterations=1,
            enable_hole_fill=False,
            enable_small_region_removal=True,
            min_region_area=50,
            enable_region_connection=False,
            connection_max_gap=5
        )
    
    @classmethod
    def balanced(cls):
        """
        预设配置：平衡配置，适用于混合缺陷场景
        Preset configuration: balanced for mixed defect scenarios
        """
        return cls(
            enable_closing=True,
            closing_kernel_size=5,
            closing_iterations=1,
            enable_hole_fill=True,
            enable_small_region_removal=True,
            min_region_area=100,
            enable_region_connection=False,
            connection_max_gap=10
        )
    
    @classmethod
    def conservative(cls):
        """
        预设配置：保守配置，专注于填充内部孔洞，不扩展边界
        Preset configuration: conservative, focus on filling internal holes only, 
        without expanding boundaries or connecting regions.
        This aims to improve IoU by filling holes in large defects while preserving 
        the original edge detection accuracy.
        """
        return cls(
            enable_closing=True,
            closing_kernel_size=3,          # 小核大小，减少边界变化
            closing_iterations=1,            # 单次迭代
            enable_hole_fill=True,           # 填充闭合区域内孔洞（关键功能）
            enable_small_region_removal=False,  # 不移除小区域，保留边缘小缺陷
            min_region_area=50,
            enable_region_connection=False,  # 不连接相邻区域
            connection_max_gap=5
        )
    
    @classmethod
    def minimal(cls):
        """
        预设配置：最小化处理，仅填充闭合区域内的孔洞
        Preset configuration: minimal processing, only fill holes inside closed regions.
        Best for preserving original edge detection while improving large defect fill.
        """
        return cls(
            enable_closing=False,            # 不使用形态学闭操作
            closing_kernel_size=3,
            closing_iterations=1,
            enable_hole_fill=True,           # 仅填充内部孔洞
            enable_small_region_removal=False,  # 保留所有区域
            min_region_area=50,
            enable_region_connection=False,
            connection_max_gap=5
        )


if __name__ == '__main__':
    # 测试代码
    # Test code
    import matplotlib.pyplot as plt
    
    # 创建测试掩码：有孔洞和分散区域
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    
    # 添加一个有孔洞的大区域
    cv2.rectangle(test_mask, (50, 50), (150, 150), 255, -1)
    cv2.rectangle(test_mask, (80, 80), (120, 120), 0, -1)  # 孔洞
    
    # 添加一些分散的小区域
    cv2.circle(test_mask, (180, 100), 15, 255, -1)
    cv2.circle(test_mask, (200, 120), 10, 255, -1)
    
    # 添加一些噪声
    cv2.circle(test_mask, (30, 200), 5, 255, -1)
    cv2.circle(test_mask, (220, 200), 3, 255, -1)
    
    # 应用后处理
    processor = PostProcessor.for_large_defects()
    processed = processor(test_mask)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(test_mask, cmap='gray')
    axes[0].set_title('Original Mask')
    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title('After Post-processing')
    plt.tight_layout()
    plt.savefig('/tmp/postprocess_test.png')
    print("Test saved to /tmp/postprocess_test.png")
