"""
简化版热力图可视化工具 - 不需要加载模型
直接对比预测掩码和真实掩码
"""

import os
import argparse

def load_image_safe(image_path):
    """安全加载图像"""
    import numpy as np
    from PIL import Image
    
    if not os.path.exists(image_path):
        print(f"❌ 错误: 文件不存在 {image_path}")
        return None
    try:
        img = Image.open(image_path).convert('L')
        return np.array(img)
    except Exception as e:
        print(f"❌ 加载图像失败 {image_path}: {e}")
        return None

def visualize_simple_heatmap(pred_path, gt_path, output_path, img_name="image"):
    """
    生成简化版缺陷热力图可视化
    
    参数:
        pred_path: 预测掩码路径
        gt_path: 真实掩码路径
        output_path: 输出图像路径
        img_name: 图像名称
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from PIL import Image
    
    # 加载预测和真实掩码
    pred_mask = load_image_safe(pred_path)
    gt_mask = load_image_safe(gt_path)
    
    if pred_mask is None or gt_mask is None:
        print("❌ 无法加载掩码文件")
        return False
    
    # 调整尺寸以匹配
    if pred_mask.shape != gt_mask.shape:
        print(f"⚠️  调整真实掩码尺寸: {gt_mask.shape} -> {pred_mask.shape}")
        gt_img = Image.fromarray(gt_mask)
        gt_img = gt_img.resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
        gt_mask = np.array(gt_img)
    
    # 二值化掩码
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    # 计算差异图
    # True Positive (TP): 两者都是1
    tp = np.logical_and(pred_binary == 1, gt_binary == 1)
    # False Positive (FP): 预测为1但真实为0 (误报)
    fp = np.logical_and(pred_binary == 1, gt_binary == 0)
    # False Negative (FN): 预测为0但真实为1 (漏报)
    fn = np.logical_and(pred_binary == 0, gt_binary == 1)
    # True Negative (TN): 两者都是0
    tn = np.logical_and(pred_binary == 0, gt_binary == 0)
    
    # 计算评估指标
    intersection = tp.sum()
    union = (pred_binary | gt_binary).sum()
    iou = intersection / union if union > 0 else 0
    
    precision = tp.sum() / (tp.sum() + fp.sum()) if (tp.sum() + fp.sum()) > 0 else 0
    recall = tp.sum() / (tp.sum() + fn.sum()) if (tp.sum() + fn.sum()) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 创建彩色差异图
    diff_map = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
    diff_map[tp] = [0, 255, 0]    # 绿色: True Positive (正确预测的缺陷)
    diff_map[fp] = [255, 0, 0]    # 红色: False Positive (误报 - 错误地预测为缺陷)
    diff_map[fn] = [255, 255, 0]  # 黄色: False Negative (漏报 - 遗漏的缺陷)
    # tn 保持黑色 (背景)
    
    # 创建误差热力图 (绝对差值)
    error_map = np.abs(pred_mask.astype(float) - gt_mask.astype(float))
    
    # 创建图形 - 2行3列
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 预测掩码
    plt.subplot(2, 3, 1)
    plt.imshow(pred_mask, cmap='gray', vmin=0, vmax=255)
    plt.title('预测掩码 (Predicted Mask)', fontsize=14, pad=10)
    plt.axis('off')
    
    # 2. 真实掩码
    plt.subplot(2, 3, 2)
    plt.imshow(gt_mask, cmap='gray', vmin=0, vmax=255)
    plt.title('真实掩码 (Ground Truth)', fontsize=14, pad=10)
    plt.axis('off')
    
    # 3. 差异分析图
    plt.subplot(2, 3, 3)
    plt.imshow(diff_map)
    plt.title('差异分析图 (Difference Map)\n'
              '绿色=正确预测 | 红色=误报 | 黄色=漏报',
              fontsize=12, pad=10)
    plt.axis('off')
    
    # 4. 预测掩码 (伪彩色)
    plt.subplot(2, 3, 4)
    im1 = plt.imshow(pred_mask, cmap='hot', vmin=0, vmax=255)
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('预测掩码热力图', fontsize=14, pad=10)
    plt.axis('off')
    
    # 5. 真实掩码 (伪彩色)
    plt.subplot(2, 3, 5)
    im2 = plt.imshow(gt_mask, cmap='hot', vmin=0, vmax=255)
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.title('真实掩码热力图', fontsize=14, pad=10)
    plt.axis('off')
    
    # 6. 误差热力图
    plt.subplot(2, 3, 6)
    im3 = plt.imshow(error_map, cmap='jet', vmin=0, vmax=255)
    plt.colorbar(im3, fraction=0.046, pad=0.04)
    plt.title('误差热力图 (Error Map)\n'
              '颜色越亮差异越大',
              fontsize=12, pad=10)
    plt.axis('off')
    
    # 添加总体标题和统计信息
    stats_text = (
        f'图像: {img_name}\n'
        f'IoU (交并比): {iou:.4f} | '
        f'Precision (精确率): {precision:.4f} | '
        f'Recall (召回率): {recall:.4f} | '
        f'F1-Score: {f1_score:.4f}\n'
        f'TP={tp.sum():,} | FP={fp.sum():,} | FN={fn.sum():,} | TN={tn.sum():,}'
    )
    
    fig.suptitle(f'脐橙缺陷分割 - 热力图分析\n{stats_text}',
                 fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    print("\n" + "="*60)
    print(f"✅ 热力图已保存: {output_path}")
    print("="*60)
    print(f"📊 评估指标:")
    print(f"   IoU (交并比):     {iou:.4f}")
    print(f"   Precision (精确率): {precision:.4f}")
    print(f"   Recall (召回率):   {recall:.4f}")
    print(f"   F1-Score:         {f1_score:.4f}")
    print(f"\n📈 混淆矩阵统计:")
    print(f"   True Positive  (TP - 正确检测): {tp.sum():>8,} 像素")
    print(f"   False Positive (FP - 误报):     {fp.sum():>8,} 像素")
    print(f"   False Negative (FN - 漏报):     {fn.sum():>8,} 像素")
    print(f"   True Negative  (TN - 正确背景): {tn.sum():>8,} 像素")
    print("="*60 + "\n")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='脐橙缺陷分割热力图可视化工具 (简化版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法 (使用默认路径)
  python visualize_heatmap_simple.py
  
  # 指定特定图像
  python visualize_heatmap_simple.py --img_name 398
  
  # 完全自定义路径
  python visualize_heatmap_simple.py \\
      --pred_path ./save/mobile_sam_adapter/vit_change_2/pred/398.png \\
      --gt_path ./data/orange/masks/398.png \\
      --output_path ./my_heatmap.png
        """
    )
    
    parser.add_argument('--pred_path', type=str,
                       default=None,
                       help='预测掩码路径')
    parser.add_argument('--gt_path', type=str,
                       default=None,
                       help='真实掩码路径')
    parser.add_argument('--output_path', type=str,
                       default=None,
                       help='输出热力图路径')
    parser.add_argument('--img_name', type=str,
                       default='398',
                       help='图像名称（不含扩展名），用于自动构建路径')
    parser.add_argument('--pred_dir', type=str,
                       default='./save/mobile_sam_adapter/vit_change_2/pred',
                       help='预测掩码目录')
    parser.add_argument('--gt_dir', type=str,
                       default='./data/orange/masks',
                       help='真实掩码目录')
    
    args = parser.parse_args()
    
    # 如果没有指定完整路径，使用默认路径模板
    if args.pred_path is None:
        args.pred_path = os.path.join(args.pred_dir, f"{args.img_name}.png")
    
    if args.gt_path is None:
        args.gt_path = os.path.join(args.gt_dir, f"{args.img_name}.png")
    
    if args.output_path is None:
        args.output_path = f"./defect_heatmap_{args.img_name}.png"
    
    print("\n" + "="*60)
    print("脐橙缺陷分割 - 热力图可视化工具")
    print("="*60)
    print(f"📁 预测掩码: {args.pred_path}")
    print(f"📁 真实掩码: {args.gt_path}")
    print(f"💾 输出路径: {args.output_path}")
    print("="*60 + "\n")
    
    # 生成可视化
    success = visualize_simple_heatmap(
        args.pred_path,
        args.gt_path,
        args.output_path,
        img_name=args.img_name
    )
    
    if not success:
        print("\n❌ 可视化失败")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
