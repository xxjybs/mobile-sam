"""
脐橙缺陷分割 - 缺陷热力图可视化工具
用于对比预测掩码与真实掩码，生成热力图和差异分析图
"""

import os
import argparse

def load_image_safe(image_path):
    """安全加载图像"""
    from PIL import Image
    import numpy as np
    
    if not os.path.exists(image_path):
        print(f"警告: 文件不存在 {image_path}")
        return None
    try:
        img = Image.open(image_path).convert('L')
        return np.array(img)
    except Exception as e:
        print(f"加载图像失败 {image_path}: {e}")
        return None

def load_original_image(data_dir, img_name, test=True):
    """加载原始图像（从numpy文件）"""
    import numpy as np
    
    try:
        # 尝试从测试集加载
        if test:
            imgs = np.load(f"{data_dir}/data_test.npy")
            # 加载测试图像列表
            test_list_file = f"{data_dir}/imageset/test.txt"
            if os.path.exists(test_list_file):
                with open(test_list_file, "r") as f:
                    img_names = [line.strip() for line in f.readlines()]
                if img_name in img_names:
                    idx = img_names.index(img_name)
                    if idx < len(imgs):
                        return imgs[idx]
        return None
    except Exception as e:
        print(f"加载原始图像失败: {e}")
        return None

def compute_prediction_confidence(model, data_dir, img_name, device, img_size=256):
    """使用模型计算预测置信度"""
    import numpy as np
    import torch
    import torch.nn.functional as F
    
    try:
        # 加载原始图像
        imgs = np.load(f"{data_dir}/data_test.npy")
        test_list_file = f"{data_dir}/imageset/test.txt"
        
        if not os.path.exists(test_list_file):
            print(f"测试列表文件不存在: {test_list_file}")
            return None
            
        with open(test_list_file, "r") as f:
            img_names = [line.strip() for line in f.readlines()]
        
        if img_name not in img_names:
            print(f"图像 {img_name} 不在测试列表中")
            return None
            
        idx = img_names.index(img_name)
        if idx >= len(imgs):
            print(f"索引 {idx} 超出范围")
            return None
            
        # 准备输入
        img = imgs[idx]  # [H, W, 3]
        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).float()
        
        # 调整大小
        if img_tensor.size(2) != img_size or img_tensor.size(3) != img_size:
            img_tensor = F.interpolate(img_tensor, size=img_size, mode='bilinear', align_corners=False)
        
        img_tensor = img_tensor.to(device)
        
        # 预测
        model.eval()
        with torch.no_grad():
            logit = model(img_tensor)
            if isinstance(logit, (list, tuple)):
                logit = logit[0]
            # 使用sigmoid获取概率
            confidence = torch.sigmoid(logit[0, 0]).cpu().numpy()
        
        return confidence
    except Exception as e:
        print(f"计算预测置信度失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_font_properties():
    """
    获取可用的中文字体配置
    Returns font properties or None if Chinese fonts are not available
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # Try to find Chinese fonts
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'WenQuanYi Zen Hei', 
                        'Noto Sans CJK', 'Droid Sans Fallback']
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in chinese_fonts:
            if font in available_fonts:
                return font
        
        # No Chinese font found, return None to use default
        return None
    except:
        return None

def visualize_defect_heatmap(pred_path, gt_path, output_path, 
                            original_img=None, confidence_map=None,
                            img_name="image"):
    """
    生成缺陷热力图可视化
    
    参数:
        pred_path: 预测掩码路径
        gt_path: 真实掩码路径
        output_path: 输出图像路径
        original_img: 原始图像（可选）
        confidence_map: 预测置信度图（可选）
        img_name: 图像名称
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # 加载预测和真实掩码
    pred_mask = load_image_safe(pred_path)
    gt_mask = load_image_safe(gt_path)
    
    if pred_mask is None or gt_mask is None:
        print("无法加载掩码文件，跳过可视化")
        return
    
    # 调整尺寸以匹配
    if pred_mask.shape != gt_mask.shape:
        gt_img = Image.fromarray(gt_mask)
        gt_img = gt_img.resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
        gt_mask = np.array(gt_img)
    
    # 二值化掩码
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    # 计算差异图
    # True Positive (TP): 两者都是1
    tp = np.logical_and(pred_binary == 1, gt_binary == 1)
    # False Positive (FP): 预测为1但真实为0
    fp = np.logical_and(pred_binary == 1, gt_binary == 0)
    # False Negative (FN): 预测为0但真实为1
    fn = np.logical_and(pred_binary == 0, gt_binary == 1)
    # True Negative (TN): 两者都是0
    tn = np.logical_and(pred_binary == 0, gt_binary == 0)
    
    # 计算IoU
    intersection = tp.sum()
    union = (pred_binary | gt_binary).sum()
    iou = intersection / union if union > 0 else 0
    
    # 计算精确率和召回率
    precision = tp.sum() / (tp.sum() + fp.sum()) if (tp.sum() + fp.sum()) > 0 else 0
    recall = tp.sum() / (tp.sum() + fn.sum()) if (tp.sum() + fn.sum()) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 创建彩色差异图
    diff_map = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
    diff_map[tp] = [0, 255, 0]    # 绿色: True Positive (正确预测的缺陷)
    diff_map[fp] = [255, 0, 0]    # 红色: False Positive (误报)
    diff_map[fn] = [255, 255, 0]  # 黄色: False Negative (漏报)
    
    # 确定子图布局
    has_original = original_img is not None
    has_confidence = confidence_map is not None
    
    if has_original and has_confidence:
        n_rows, n_cols = 2, 3
    elif has_original or has_confidence:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 2, 2
    
    # 创建图形
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    
    # Get font properties for Chinese text
    chinese_font = get_font_properties()
    font_props = {'fontproperties': chinese_font} if chinese_font else {}
    
    plot_idx = 1
    
    # 原始图像
    if has_original:
        plt.subplot(n_rows, n_cols, plot_idx)
        if len(original_img.shape) == 3:
            plt.imshow(original_img)
        else:
            plt.imshow(original_img, cmap='gray')
        plt.title('原始图像 (Original Image)', fontsize=14, **font_props)
        plt.axis('off')
        plot_idx += 1
    
    # 预测掩码
    plt.subplot(n_rows, n_cols, plot_idx)
    plt.imshow(pred_mask, cmap='gray', vmin=0, vmax=255)
    plt.title('预测掩码 (Predicted Mask)', fontsize=14, **font_props)
    plt.axis('off')
    plot_idx += 1
    
    # 真实掩码
    plt.subplot(n_rows, n_cols, plot_idx)
    plt.imshow(gt_mask, cmap='gray', vmin=0, vmax=255)
    plt.title('真实掩码 (Ground Truth)', fontsize=14, **font_props)
    plt.axis('off')
    plot_idx += 1
    
    # 差异图
    plt.subplot(n_rows, n_cols, plot_idx)
    plt.imshow(diff_map)
    plt.title(f'差异分析 (Difference)\n绿色=正确 红色=误报 黄色=漏报', 
              fontsize=12, **font_props)
    plt.axis('off')
    plot_idx += 1
    
    # 预测置信度热力图
    if has_confidence:
        plt.subplot(n_rows, n_cols, plot_idx)
        im = plt.imshow(confidence_map, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('预测置信度热力图 (Confidence)', fontsize=14, **font_props)
        plt.axis('off')
        plot_idx += 1
    
    # 叠加可视化 (如果有原始图像)
    if has_original:
        plt.subplot(n_rows, n_cols, plot_idx)
        if len(original_img.shape) == 3:
            base_img = original_img.copy()
        else:
            base_img = np.stack([original_img] * 3, axis=-1)
        
        # 调整尺寸匹配
        if base_img.shape[:2] != diff_map.shape[:2]:
            try:
                from scipy import ndimage
                base_img = ndimage.zoom(base_img, 
                                       (diff_map.shape[0] / base_img.shape[0],
                                        diff_map.shape[1] / base_img.shape[1], 1), 
                                       order=1)
            except ImportError:
                # Fallback to PIL if scipy is not available
                from PIL import Image as PILImage
                base_img_pil = PILImage.fromarray(base_img.astype(np.uint8))
                base_img_pil = base_img_pil.resize((diff_map.shape[1], diff_map.shape[0]), PILImage.BILINEAR)
                base_img = np.array(base_img_pil)
        
        # 将差异图叠加到原始图像上
        overlay = base_img.copy()
        mask_any = (diff_map.sum(axis=-1) > 0)
        overlay[mask_any] = (overlay[mask_any] * 0.4 + diff_map[mask_any] * 0.6).astype(np.uint8)
        
        plt.imshow(overlay)
        plt.title('叠加可视化 (Overlay)', fontsize=14, **font_props)
        plt.axis('off')
        plot_idx += 1
    
    # 添加总体标题和统计信息
    fig.suptitle(f'缺陷热力图分析 - {img_name}\n'
                 f'IoU: {iou:.4f} | Precision: {precision:.4f} | '
                 f'Recall: {recall:.4f} | F1: {f1_score:.4f}',
                 fontsize=16, y=0.98, **font_props)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 热力图已保存: {output_path}")
    print(f"   IoU: {iou:.4f}")
    print(f"   精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1_score:.4f}")
    print(f"   TP: {tp.sum()}, FP: {fp.sum()}, FN: {fn.sum()}, TN: {tn.sum()}")

def main():
    parser = argparse.ArgumentParser(description='脐橙缺陷分割热力图可视化工具')
    parser.add_argument('--pred_path', type=str, 
                       default='./save/mobile_sam_adapter/vit_change_2/pred/398.png',
                       help='预测掩码路径')
    parser.add_argument('--gt_path', type=str,
                       default='./data/orange/masks/398.png',
                       help='真实掩码路径')
    parser.add_argument('--output_path', type=str,
                       default='./defect_heatmap_398.png',
                       help='输出热力图路径')
    parser.add_argument('--data_dir', type=str,
                       default='./data/orange',
                       help='数据集目录')
    parser.add_argument('--img_name', type=str,
                       default='398',
                       help='图像名称（不含扩展名）')
    parser.add_argument('--model_name', type=str,
                       default='mobile_sam_adapter',
                       help='模型名称')
    parser.add_argument('--ckpt_path', type=str,
                       default=None,
                       help='模型权重路径（可选，用于生成置信度图）')
    parser.add_argument('--img_size', type=int,
                       default=256,
                       help='输入图像尺寸')
    
    args = parser.parse_args()
    
    # Import dependencies after argument parsing to allow --help without installation
    import numpy as np
    import torch
    from models import model_dict
    from models.load_ckpt import load_checkpoint
    
    # 加载原始图像（如果可用）
    original_img = load_original_image(args.data_dir, args.img_name, test=True)
    
    # 如果提供了模型权重，计算置信度图
    confidence_map = None
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"加载模型以计算置信度图...")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model_dict[args.model_name]()
            model = model.to(device)
            load_checkpoint(model, args.ckpt_path, device, args.model_name)
            
            confidence_map = compute_prediction_confidence(
                model, args.data_dir, args.img_name, device, args.img_size
            )
            
            if confidence_map is not None:
                print("✅ 置信度图计算完成")
        except Exception as e:
            print(f"加载模型失败: {e}")
    
    # 生成可视化
    visualize_defect_heatmap(
        args.pred_path,
        args.gt_path,
        args.output_path,
        original_img=original_img,
        confidence_map=confidence_map,
        img_name=args.img_name
    )

if __name__ == '__main__':
    main()
