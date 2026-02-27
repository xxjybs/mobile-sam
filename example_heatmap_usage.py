#!/usr/bin/env python3
"""
示例脚本: 演示如何使用热力图可视化工具
Example script: Demonstrates how to use the heatmap visualization tools
"""

import os
import subprocess
import sys

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import numpy
        import matplotlib
        from PIL import Image
        print("✅ 所有依赖已安装 / All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖 / Missing dependency: {e}")
        print("\n请安装以下依赖 / Please install the following dependencies:")
        print("pip install numpy matplotlib pillow")
        return False

def example_1_basic():
    """示例1: 基本用法 - 使用默认路径"""
    print("\n" + "="*60)
    print("示例 1: 基本用法 (使用默认路径)")
    print("Example 1: Basic usage (using default paths)")
    print("="*60)
    
    cmd = ["python", "visualize_heatmap_simple.py"]
    print(f"命令 / Command: {' '.join(cmd)}")
    print("\n这将查找:")
    print("  预测掩码: ./save/mobile_sam_adapter/vit_change_2/pred/398.png")
    print("  真实掩码: ./data/orange/masks/398.png")
    print("  输出: ./defect_heatmap_398.png")
    
    # 检查文件是否存在
    pred_path = "./save/mobile_sam_adapter/vit_change_2/pred/398.png"
    gt_path = "./data/orange/masks/398.png"
    
    if os.path.exists(pred_path) and os.path.exists(gt_path):
        print("\n✅ 文件存在，执行可视化...")
        subprocess.run(cmd)
    else:
        print("\n⚠️  文件不存在，跳过执行")
        print(f"   预测文件存在: {os.path.exists(pred_path)}")
        print(f"   真实文件存在: {os.path.exists(gt_path)}")

def example_2_custom_image():
    """示例2: 指定不同的图像"""
    print("\n" + "="*60)
    print("示例 2: 指定不同的图像")
    print("Example 2: Specify a different image")
    print("="*60)
    
    img_name = "123"  # 可以改为任何图像名称
    cmd = [
        "python", "visualize_heatmap_simple.py",
        "--img_name", img_name
    ]
    print(f"命令 / Command: {' '.join(cmd)}")
    print(f"\n这将处理图像: {img_name}.png")

def example_3_custom_paths():
    """示例3: 完全自定义路径"""
    print("\n" + "="*60)
    print("示例 3: 完全自定义路径")
    print("Example 3: Fully custom paths")
    print("="*60)
    
    cmd = [
        "python", "visualize_heatmap_simple.py",
        "--pred_path", "./save/mobile_sam_adapter/vit_change_2/pred/398.png",
        "--gt_path", "./data/orange/masks/398.png",
        "--output_path", "./my_custom_heatmap.png"
    ]
    print(f"命令 / Command:")
    print("python visualize_heatmap_simple.py \\")
    print("    --pred_path ./save/mobile_sam_adapter/vit_change_2/pred/398.png \\")
    print("    --gt_path ./data/orange/masks/398.png \\")
    print("    --output_path ./my_custom_heatmap.png")

def example_4_batch_processing():
    """示例4: 批量处理多张图像"""
    print("\n" + "="*60)
    print("示例 4: 批量处理多张图像")
    print("Example 4: Batch processing multiple images")
    print("="*60)
    
    print("\nPython 批处理示例:")
    print("""
import subprocess

img_names = ['398', '399', '400', '401', '402']

for img_name in img_names:
    print(f"Processing {img_name}...")
    subprocess.run([
        'python', 'visualize_heatmap_simple.py',
        '--img_name', img_name
    ])
    """)
    
    print("\nBash 批处理示例:")
    print("""
#!/bin/bash
for img in 398 399 400 401 402; do
    echo "Processing image $img..."
    python visualize_heatmap_simple.py --img_name $img
done
    """)

def example_5_advanced():
    """示例5: 使用高级版本 (带置信度图)"""
    print("\n" + "="*60)
    print("示例 5: 高级版本 - 带置信度热力图")
    print("Example 5: Advanced version - with confidence heatmap")
    print("="*60)
    
    print("\n命令 / Command:")
    print("python visualize_heatmap.py \\")
    print("    --pred_path ./save/mobile_sam_adapter/vit_change_2/pred/398.png \\")
    print("    --gt_path ./data/orange/masks/398.png \\")
    print("    --output_path ./defect_heatmap_with_confidence_398.png \\")
    print("    --model_name mobile_sam_adapter \\")
    print("    --ckpt_path ./save/mobile_sam_adapter/vit_change_2/mobile_sam_adapter_best.pth \\")
    print("    --img_name 398 \\")
    print("    --img_size 256")
    
    print("\n注意: 高级版本需要:")
    print("  1. 模型权重文件 (.pth)")
    print("  2. 数据集目录 (包含 .npy 文件)")
    print("  3. 正确的模型配置")

def main():
    print("\n" + "="*60)
    print("脐橙缺陷分割 - 热力图可视化工具示例")
    print("Orange Defect Segmentation - Heatmap Visualization Examples")
    print("="*60)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装依赖后再运行示例")
        return 1
    
    # 显示所有示例
    example_1_basic()
    example_2_custom_image()
    example_3_custom_paths()
    example_4_batch_processing()
    example_5_advanced()
    
    print("\n" + "="*60)
    print("查看详细文档: HEATMAP_VISUALIZATION_README.md")
    print("See detailed documentation: HEATMAP_VISUALIZATION_README.md")
    print("="*60 + "\n")
    
    # 如果用户想要，可以执行示例1
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        print("\n尝试运行示例1...")
        example_1_basic()
    
    return 0

if __name__ == '__main__':
    exit(main())
