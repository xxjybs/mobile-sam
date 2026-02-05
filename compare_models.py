"""
多模型对比可视化脚本 / Multi-Model Comparison Visualization Script

生成类似figure2.png的模型对比图，展示不同模型在脐橙缺陷检测数据集上的效果
Generates comparison figures like figure2.png showing different models' performance on orange defect dataset

支持的模型 / Supported models:
- mobile-sam-adapter (.pth格式): 自定义模型
- sam2.1_hiera_tiny (.pt格式)
- segformerb0_hf (.pth格式): HuggingFace SegFormer
- segformerb1_hf (.pth格式): HuggingFace SegFormer
- mobile_sam (.pt格式)
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import model_dict
from models.load_ckpt import load_checkpoint
from dataset.OrangeDefectDataloader import OrangeDefectLoader


# ==================== 配置区 / Configuration ====================

# 模型配置: (模型名称, checkpoint路径, 模型类型)
# Model config: (model_name, checkpoint_path, model_type)
MODEL_CONFIGS = {
    'mobile_sam_adapter': {
        'model_key': 'mobile_sam_adapter',  # model_dict中的键
        'ckpt_path': './save/mobile_sam_adapter/vit_change_2/best.pth',  # 你的模型路径
        'ckpt_format': 'pth',
        'display_name': 'Mobile-SAM-Adapter (Ours)',
    },
    'sam2_hiera_tiny': {
        'model_key': 'sam2_adapter_tiny',
        'ckpt_path': './checkpoints/sam2.1_hiera_tiny.pt',
        'ckpt_format': 'pt',
        'display_name': 'SAM2.1 Hiera Tiny',
    },
    # HuggingFace SegFormer 模型 (用 train_segformer_hf.py 训练的)
    'segformerb0': {
        'model_key': 'segformer_hf_b0',  # 特殊标记，使用HuggingFace加载
        'ckpt_path': './save/segformer_hf/best_model.pth',
        'ckpt_format': 'hf_pth',  # HuggingFace格式的pth
        'display_name': 'SegFormer-B0',
    },
    'segformerb1': {
        'model_key': 'segformer_hf_b1',  # 特殊标记，使用HuggingFace加载
        'ckpt_path': './save/segformer_1hf/best_model.pth',
        'ckpt_format': 'hf_pth',  # HuggingFace格式的pth
        'display_name': 'SegFormer-B1',
    },
    'mobile_sam': {
        'model_key': 'mobile_sam_adapter',  # 使用相同的模型结构
        'ckpt_path': './checkpoints/mobile_sam.pt',
        'ckpt_format': 'pt',
        'display_name': 'Mobile-SAM',
    },
}

# 输出配置 / Output configuration
OUTPUT_DIR = './comparison_results'
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'


# ==================== 辅助函数 / Helper Functions ====================

def strip_model_prefix(state_dict):
    """
    移除 state_dict 键名中的 'model.' 前缀
    train_segformer_hf.py 保存的权重有 'model.' 前缀，需要去掉
    """
    new_state_dict = {}
    has_prefix = False
    
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v  # 去掉 "model." 前缀 (6个字符)
            has_prefix = True
        else:
            new_state_dict[k] = v
    
    if has_prefix:
        print(f"    ✅ 已移除 'model.' 前缀")
    
    return new_state_dict


def analyze_hf_checkpoint(state_dict):
    """
    分析 HuggingFace SegFormer checkpoint 配置 / Analyze checkpoint configuration
    
    从权重文件自动检测:
    - encoder hidden_sizes
    - decoder hidden_size
    """
    config_info = {}
    
    # 1. 检测 decoder_hidden_size (从 decode_head.linear_c.0.proj.weight)
    for k, v in state_dict.items():
        if 'decode_head.linear_c.0.proj.weight' in k:
            config_info['decoder_hidden_size'] = v.shape[0]
            break
    
    # 2. 检测 encoder hidden_sizes (从 layer_norm weights)
    hidden_sizes = []
    for i in range(4):
        for k, v in state_dict.items():
            if f'segformer.encoder.layer_norm.{i}.weight' in k:
                hidden_sizes.append(v.shape[0])
                break
    
    if len(hidden_sizes) == 4:
        config_info['hidden_sizes'] = hidden_sizes
    
    return config_info


def load_hf_segformer(variant, ckpt_path, device, num_classes=2):
    """
    加载 HuggingFace SegFormer 模型 / Load HuggingFace SegFormer model
    
    Args:
        variant: 'b0', 'b1', 'b2', etc.
        ckpt_path: checkpoint 路径
        device: cuda/cpu
        num_classes: 类别数
    
    Returns:
        model: 加载好权重的模型
    """
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
    except ImportError:
        print("❌ transformers library not installed. Install with: pip install transformers")
        return None
    
    # 根据变体创建默认配置
    variant_configs = {
        'b0': {
            'hidden_sizes': [32, 64, 160, 256], 
            'depths': [2, 2, 2, 2], 
            'num_attention_heads': [1, 2, 5, 8],
            'decoder_hidden_size': 256
        },
        'b1': {
            'hidden_sizes': [64, 128, 320, 512], 
            'depths': [2, 2, 2, 2], 
            'num_attention_heads': [1, 2, 5, 8],
            'decoder_hidden_size': 256
        },
        'b2': {
            'hidden_sizes': [64, 128, 320, 512], 
            'depths': [3, 4, 6, 3], 
            'num_attention_heads': [1, 2, 5, 8],
            'decoder_hidden_size': 768
        },
        'b3': {
            'hidden_sizes': [64, 128, 320, 512], 
            'depths': [3, 4, 18, 3], 
            'num_attention_heads': [1, 2, 5, 8],
            'decoder_hidden_size': 768
        },
        'b4': {
            'hidden_sizes': [64, 128, 320, 512], 
            'depths': [3, 8, 27, 3], 
            'num_attention_heads': [1, 2, 5, 8],
            'decoder_hidden_size': 768
        },
        'b5': {
            'hidden_sizes': [64, 128, 320, 512], 
            'depths': [3, 6, 40, 3], 
            'num_attention_heads': [1, 2, 5, 8],
            'decoder_hidden_size': 768
        },
    }
    
    if variant not in variant_configs:
        print(f"⚠️ Unknown SegFormer variant: {variant}, using b0")
        variant = 'b0'
    
    v_config = variant_configs[variant].copy()
    
    # 先加载权重
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # 打印示例键名用于调试
    sample_keys = list(state_dict.keys())[:5]
    print(f"    权重键名示例: {sample_keys}")
    
    # 移除 model. 前缀 (train_segformer_hf.py 保存时会加上这个前缀)
    state_dict = strip_model_prefix(state_dict)
    
    # 从权重中分析配置
    ckpt_config = analyze_hf_checkpoint(state_dict)
    
    if ckpt_config:
        print(f"    === 权重配置分析 ===")
        if 'hidden_sizes' in ckpt_config:
            print(f"    Encoder hidden_sizes: {ckpt_config['hidden_sizes']}")
            v_config['hidden_sizes'] = ckpt_config['hidden_sizes']
        if 'decoder_hidden_size' in ckpt_config:
            print(f"    Decoder hidden_size: {ckpt_config['decoder_hidden_size']}")
            v_config['decoder_hidden_size'] = ckpt_config['decoder_hidden_size']
    
    # 创建配置
    config = SegformerConfig(
        num_labels=num_classes,
        num_encoder_blocks=4,
        depths=v_config['depths'],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=v_config['hidden_sizes'],
        num_attention_heads=v_config['num_attention_heads'],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        decoder_hidden_size=v_config['decoder_hidden_size'],
    )
    
    print(f"    创建模型配置: hidden_sizes={v_config['hidden_sizes']}, decoder_hidden_size={v_config['decoder_hidden_size']}")
    
    # 创建模型
    model = SegformerForSemanticSegmentation(config)
    
    # 跳过分类器层（类别数不匹配）
    keys_to_skip = []
    for key in state_dict.keys():
        if 'decode_head.classifier' in key:
            keys_to_skip.append(key)
    
    if keys_to_skip:
        print(f"    跳过分类器层 (类别数可能不匹配): {len(keys_to_skip)} keys")
        for key in keys_to_skip:
            del state_dict[key]
    
    # 加载权重
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    print(f"    ✅ Loaded: {loaded}/{len(model.state_dict())} params")
    if len(missing) > 0 and len(missing) <= 10:
        print(f"    Missing: {missing}")
    elif len(missing) > 10:
        print(f"    Missing: {len(missing)} params (分类器层会重新学习)")
    
    return model


class HFSegFormerWrapper(torch.nn.Module):
    """
    HuggingFace SegFormer 包装器，统一接口
    """
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
    
    def forward(self, x):
        # HuggingFace SegFormer 输出是 SegformerOutput
        outputs = self.model(x)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]
        
        # 上采样到输入尺寸
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 取前景通道 (channel 1)
        if logits.shape[1] == 2:
            return logits[:, 1:2, :, :]  # [B, 1, H, W]
        else:
            return logits


def load_model_with_checkpoint(model_config, device):
    """
    加载模型并载入checkpoint / Load model and checkpoint
    """
    model_key = model_config['model_key']
    ckpt_path = model_config['ckpt_path']
    ckpt_format = model_config['ckpt_format']
    
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint not found: {ckpt_path}")
        return None
    
    print(f"  Loading model: {model_config['display_name']}")
    
    try:
        # HuggingFace SegFormer 模型
        if ckpt_format == 'hf_pth':
            # 从 model_key 中提取变体 (segformer_hf_b0 -> b0)
            variant = model_key.split('_')[-1]  # 'b0', 'b1', etc.
            hf_model = load_hf_segformer(variant, ckpt_path, device)
            if hf_model is None:
                return None
            model = HFSegFormerWrapper(hf_model)
        
        # 项目内置模型
        elif ckpt_format == 'pth':
            model = model_dict[model_key]()
            load_checkpoint(model, ckpt_path, device, model_key)
        
        else:  # pt格式
            model = model_dict[model_key]()
            # PyTorch 2.6+ 默认 weights_only=True，需要设置为 False
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            
            # 尝试加载，忽略不匹配的键
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"    Loaded: {len(model.state_dict()) - len(missing)}/{len(model.state_dict())} params")
            if len(missing) > 0:
                print(f"    Missing: {len(missing)} params")
            if len(unexpected) > 0:
                print(f"    Unexpected: {len(unexpected)} params")
        
        model = model.to(device)
        model.eval()
        return model
    
    except Exception as e:
        print(f"❌ Error loading {model_config['display_name']}: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_single_image(model, image_tensor, device):
    """
    单张图像预测 / Predict single image
    
    Args:
        model: 模型
        image_tensor: [1, 3, H, W] 图像tensor
        device: cuda/cpu
    
    Returns:
        pred_mask: [H, W] numpy数组, 0-255
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        pred = model(image_tensor)
        pred = torch.sigmoid(pred)
        pred_mask = pred[0, 0]  # [H, W]
        pred_mask = (pred_mask > 0.5).float() * 255
        return pred_mask.cpu().numpy().astype(np.uint8)


def compute_iou(pred, gt):
    """
    计算IoU / Compute IoU
    """
    pred = (pred > 127).astype(np.uint8)
    gt = (gt > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def create_comparison_figure(
    original_images,
    gt_masks,
    predictions_dict,
    sample_indices,
    output_path,
    fig_title="Model Comparison on Orange Defect Dataset"
):
    """
    创建对比图 / Create comparison figure
    
    Args:
        original_images: list of original images [H, W, 3]
        gt_masks: list of ground truth masks [H, W]
        predictions_dict: {model_name: [pred1, pred2, ...]}
        sample_indices: list of sample indices for labeling
        output_path: output file path
        fig_title: figure title
    """
    num_samples = len(original_images)
    num_models = len(predictions_dict)
    
    # 列: Original, GT, Model1, Model2, ...
    num_cols = 2 + num_models
    
    fig = plt.figure(figsize=(3 * num_cols, 3 * num_samples))
    gs = GridSpec(num_samples, num_cols, figure=fig, wspace=0.05, hspace=0.1)
    
    # 列标题
    col_titles = ['Original Image', 'Ground Truth'] + [
        config.get('display_name', name) 
        for name, config in predictions_dict.items()
    ]
    
    for row_idx in range(num_samples):
        # 原图
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(original_images[row_idx])
        ax.axis('off')
        if row_idx == 0:
            ax.set_title('Original Image', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Sample {sample_indices[row_idx]}', fontsize=9)
        
        # GT
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.imshow(gt_masks[row_idx], cmap='gray')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title('Ground Truth', fontsize=10, fontweight='bold')
        
        # 各模型预测
        for col_idx, (model_name, preds_info) in enumerate(predictions_dict.items()):
            ax = fig.add_subplot(gs[row_idx, 2 + col_idx])
            pred = preds_info['predictions'][row_idx]
            
            ax.imshow(pred, cmap='gray')
            ax.axis('off')
            
            if row_idx == 0:
                ax.set_title(preds_info['display_name'], fontsize=10, fontweight='bold')
    
    plt.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Figure saved: {output_path}")


def create_metrics_table(predictions_dict, gt_masks, output_path):
    """
    创建指标对比表格 / Create metrics comparison table
    """
    metrics = {}
    
    for model_name, preds_info in predictions_dict.items():
        ious = [
            compute_iou(pred, gt) 
            for pred, gt in zip(preds_info['predictions'], gt_masks)
        ]
        metrics[preds_info['display_name']] = {
            'mean_iou': np.mean(ious),
            'std_iou': np.std(ious),
            'min_iou': np.min(ious),
            'max_iou': np.max(ious),
        }
    
    # 创建表格图
    fig, ax = plt.subplots(figsize=(10, len(metrics) * 0.5 + 1))
    ax.axis('off')
    
    table_data = []
    headers = ['Model', 'Mean IoU', 'Std', 'Min', 'Max']
    
    for model_name, m in metrics.items():
        table_data.append([
            model_name,
            f"{m['mean_iou']:.4f}",
            f"{m['std_iou']:.4f}",
            f"{m['min_iou']:.4f}",
            f"{m['max_iou']:.4f}",
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Model Performance Metrics', fontsize=12, fontweight='bold', y=0.8)
    plt.savefig(output_path.replace('.png', '_metrics.png'), 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    return metrics


# ==================== 主函数 / Main Function ====================

def main(
    dataset_path="./data/orange",
    sample_indices=None,
    num_samples=5,
    models_to_compare=None,
    output_name="model_comparison"
):
    """
    主函数: 生成模型对比图
    
    Args:
        dataset_path: 数据集路径
        sample_indices: 指定样本索引列表，如 [398, 100, 200]
        num_samples: 如果不指定sample_indices，则随机选择num_samples个样本
        models_to_compare: 要对比的模型列表，如 ['mobile_sam_adapter', 'segformerb0']
                          默认为None则对比所有可用模型
        output_name: 输出文件名前缀
    """
    print("=" * 60)
    print("多模型对比可视化 / Multi-Model Comparison Visualization")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载测试数据集
    print("\n[1/4] Loading test dataset...")
    testset = OrangeDefectLoader(dataset_path, train=False, test=True, size=1024, num_classes=2)
    print(f"  Total test samples: {len(testset)}")
    
    # 选择样本
    if sample_indices is None:
        sample_indices = np.random.choice(len(testset), min(num_samples, len(testset)), replace=False)
        sample_indices = sorted(sample_indices)
    print(f"  Selected samples: {sample_indices}")
    
    # 加载原始图像和GT
    print("\n[2/4] Loading images and ground truth...")
    original_images = []
    gt_masks = []
    image_tensors = []
    
    for idx in sample_indices:
        img_tensor, mask_tensor, _ = testset[idx]
        
        # 原始图像 (转换为可显示格式)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        original_images.append(img_np)
        
        # GT mask
        gt_mask = (mask_tensor.numpy() * 255).astype(np.uint8)
        gt_masks.append(gt_mask)
        
        # 保存tensor用于预测
        image_tensors.append(img_tensor.unsqueeze(0))
    
    # 确定要对比的模型
    if models_to_compare is None:
        models_to_compare = list(MODEL_CONFIGS.keys())
    
    # 过滤存在checkpoint的模型
    available_models = []
    for model_name in models_to_compare:
        if model_name in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_name]
            if os.path.exists(config['ckpt_path']):
                available_models.append(model_name)
            else:
                print(f"⚠️ Skipping {model_name}: checkpoint not found at {config['ckpt_path']}")
        else:
            print(f"⚠️ Unknown model: {model_name}")
    
    if len(available_models) == 0:
        print("❌ No models available for comparison!")
        return
    
    print(f"\n[3/4] Loading and running models: {available_models}")
    
    # 加载模型并预测
    predictions_dict = {}
    
    for model_name in available_models:
        config = MODEL_CONFIGS[model_name]
        print(f"\n  Processing: {config['display_name']}")
        
        model = load_model_with_checkpoint(config, device)
        if model is None:
            continue
        
        # 预测所有选定样本
        preds = []
        for img_tensor in image_tensors:
            pred = predict_single_image(model, img_tensor, device)
            preds.append(pred)
        
        predictions_dict[model_name] = {
            'predictions': preds,
            'display_name': config['display_name'],
        }
        
        # 清理GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 生成对比图
    print(f"\n[4/4] Generating comparison figure...")
    
    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.{FIGURE_FORMAT}")
    create_comparison_figure(
        original_images=original_images,
        gt_masks=gt_masks,
        predictions_dict=predictions_dict,
        sample_indices=list(sample_indices),
        output_path=output_path,
        fig_title="Orange Defect Segmentation - Model Comparison"
    )
    
    # 生成指标表格
    metrics = create_metrics_table(predictions_dict, gt_masks, output_path)
    
    # 打印指标汇总
    print("\n" + "=" * 60)
    print("Performance Summary (on selected samples)")
    print("=" * 60)
    for model_name, m in metrics.items():
        print(f"  {model_name}: Mean IoU = {m['mean_iou']:.4f} ± {m['std_iou']:.4f}")
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Model Comparison Visualization')
    parser.add_argument('--dataset', type=str, default='./data/orange',
                        help='Dataset path')
    parser.add_argument('--samples', type=int, nargs='+', default=None,
                        help='Sample indices to compare, e.g., --samples 398 100 200')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of random samples if --samples not specified')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to compare, e.g., --models mobile_sam_adapter segformerb0')
    parser.add_argument('--output', type=str, default='model_comparison',
                        help='Output filename prefix')
    
    args = parser.parse_args()
    
    main(
        dataset_path=args.dataset,
        sample_indices=args.samples,
        num_samples=args.num_samples,
        models_to_compare=args.models,
        output_name=args.output
    )
