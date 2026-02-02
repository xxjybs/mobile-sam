"""
SegFormer Training Script using HuggingFace Transformers
使用 HuggingFace Transformers 的 SegFormer，支持官方预训练权重

这个脚本使用 HuggingFace 的 SegformerForSemanticSegmentation，
可以直接加载官方 ImageNet 预训练权重，确保与其他模型进行公平对比。

安装依赖:
    pip install transformers

Usage:
    # 方法1: 使用本地下载的权重文件 (推荐，无需网络)
    python train_segformer_hf.py --model nvidia/mit-b0 --local_weights ./checkpoints/segformer_b0_ade.pt --offline
    
    # 方法2: 在线加载 HuggingFace 预训练权重
    python train_segformer_hf.py --model nvidia/segformer-b0-finetuned-ade-512-512
    
    # 方法3: 离线模式，从头训练
    python train_segformer_hf.py --model nvidia/mit-b0 --offline
"""

import os
import time
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import gc
from dataset.OrangeDefectDataloader import OrangeDefectLoader
from helper.loss import IOU, DiceLoss
import random
import shutil
import cv2

# HuggingFace Transformers
try:
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ 请安装 transformers: pip install transformers")


def parse_args():
    parser = argparse.ArgumentParser(description='Train HuggingFace SegFormer on Orange Defect Dataset')
    parser.add_argument('--model', type=str, default='nvidia/mit-b0',
                        help='HuggingFace model name (e.g., nvidia/mit-b0, nvidia/mit-b1)')
    parser.add_argument('--local_weights', type=str, default='',
                        help='Path to local HuggingFace weights file (e.g., ./checkpoints/segformer_b0_ade.pt)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_oversampling', action='store_true', default=True,
                        help='Use oversampling for large defect samples')
    parser.add_argument('--save_dir', type=str, default='./save/segformer_hf',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (2 for binary segmentation)')
    parser.add_argument('--offline', action='store_true', default=False,
                        help='Use offline mode (no network access)')
    return parser.parse_args()


class HFSegFormerWrapper(torch.nn.Module):
    """
    HuggingFace SegFormer 包装类，输出格式与项目其他模型一致
    """
    def __init__(self, model_name='nvidia/mit-b0', num_classes=2, img_size=256, offline=False, local_weights=''):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        # 验证模型名是否是 HuggingFace 格式
        if model_name.startswith('./') or model_name.startswith('/') or model_name.endswith('.pth') or model_name.endswith('.pt'):
            print(f"   ⚠️ 检测到本地路径格式: {model_name}")
            print(f"   ⚠️ 本脚本需要 HuggingFace 模型名称 (如 nvidia/mit-b0)")
            print(f"   ⚠️ 如需使用本地 mmseg 权重，请使用 train_segformer.py")
            print(f"   ⚠️ 将尝试从配置创建模型...")
            model_name = 'nvidia/mit-b0'  # 回退到默认
        
        # 获取模型变体
        if 'b0' in model_name.lower():
            variant = 'b0'
        elif 'b1' in model_name.lower():
            variant = 'b1'
        elif 'b2' in model_name.lower():
            variant = 'b2'
        elif 'b3' in model_name.lower():
            variant = 'b3'
        elif 'b4' in model_name.lower():
            variant = 'b4'
        elif 'b5' in model_name.lower():
            variant = 'b5'
        else:
            variant = 'b0'
        
        print(f"   Model variant: SegFormer-{variant.upper()}")
        
        # 优先使用本地权重文件
        if local_weights and os.path.exists(local_weights):
            print(f"   Loading local HuggingFace weights from: {local_weights}")
            self._create_from_config(variant, num_classes)
            self._load_local_weights(local_weights)
        elif offline:
            print(f"   离线模式: 从配置创建模型 (随机初始化)")
            self._create_from_config(variant, num_classes)
        else:
            try:
                print(f"   Loading pretrained model from HuggingFace: {model_name}")
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True  # 忽略分类头大小不匹配
                )
                print(f"   ✅ Loaded pretrained weights from: {model_name}")
            except Exception as e:
                print(f"   ⚠️ Cannot load pretrained: {e}")
                print(f"   Creating model from config...")
                self._create_from_config(variant, num_classes)
    
    def _load_local_weights(self, weights_path):
        """加载本地保存的 HuggingFace 权重文件"""
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # 检查是否有 'model' 或 'state_dict' 键
            if isinstance(state_dict, dict):
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            
            # 打印权重键名示例
            keys = list(state_dict.keys())
            print(f"   本地权重示例键名: {keys[:5]}")
            
            # 获取模型的键名
            model_keys = list(self.model.state_dict().keys())
            print(f"   模型参数示例键名: {model_keys[:5]}")
            
            # 尝试加载权重
            result = self.model.load_state_dict(state_dict, strict=False)
            loaded_count = len(model_keys) - len(result.missing_keys)
            
            if loaded_count > 0:
                print(f"   ✅ Loaded {loaded_count}/{len(model_keys)} parameters from local weights")
            else:
                print(f"   ⚠️ No parameters loaded from local weights")
                print(f"   权重格式可能与 HuggingFace SegFormer 不兼容")
            
            if result.missing_keys:
                print(f"   Missing keys: {len(result.missing_keys)}")
            if result.unexpected_keys:
                print(f"   Unexpected keys: {len(result.unexpected_keys)}")
                
        except Exception as e:
            print(f"   ⚠️ Error loading local weights: {e}")
            print(f"   将使用随机初始化")
    
    def _create_from_config(self, variant, num_classes):
        """从配置创建模型（用于离线模式或加载失败时）"""
        try:
            config = SegformerConfig.from_pretrained(f'nvidia/mit-{variant}')
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)
            print(f"   ✅ Created model from config (random initialization)")
        except Exception as e2:
            print(f"   ⚠️ 无法从 HuggingFace 获取配置: {e2}")
            print(f"   正在使用本地配置创建模型...")
            
            # 完全离线创建模型
            config = SegformerConfig(
                num_channels=3,
                num_encoder_blocks=4,
                depths=[2, 2, 2, 2] if variant == 'b0' else [2, 2, 2, 2],
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[32, 64, 160, 256] if variant == 'b0' else [64, 128, 320, 512],
                num_attention_heads=[1, 2, 5, 8] if variant == 'b0' else [1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                mlp_ratios=[4, 4, 4, 4],
                num_labels=num_classes,
                decoder_hidden_size=256 if variant == 'b0' else 512,
            )
            self.model = SegformerForSemanticSegmentation(config)
            print(f"   ✅ Created model from local config (random initialization)")
    
    def forward(self, x):
        """
        前向传播
        输入: x - [B, 3, H, W] 图像张量
        输出: logits - [B, 1, H, W] 前景概率（用于二分类）
        """
        # HuggingFace SegFormer 需要输入范围 [0, 1] 或标准化
        # 假设输入已经是 [0, 1] 范围
        
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]
        
        # 上采样到原始尺寸
        logits = F.interpolate(logits, size=(self.img_size, self.img_size), 
                               mode='bilinear', align_corners=False)
        
        # 对于二分类，只返回前景通道 (类别1)
        if self.num_classes == 2:
            # 返回前景的 logit (类别1)
            foreground_logit = logits[:, 1:2, :, :]  # [B, 1, H, W]
            return foreground_logit
        else:
            return logits


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def compute_iou(pred, target, num_classes=2):
    """计算 IoU"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    return ious


def train_one_epoch(model, dataloader, optimizer, criterion_bce, criterion_iou, criterion_dice, device, epoch):
    """训练一个 epoch"""
    model.train()
    losses = []
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        # 确保 masks 是正确的形状 [B, 1, H, W]
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        
        # 归一化 masks 到 [0, 1]
        if masks.max() > 1:
            masks = masks.float() / 255.0
        else:
            masks = masks.float()
        
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(images)  # [B, 1, H, W]
        
        # 计算损失
        loss = criterion_bce(logits, masks)
        loss = loss + criterion_iou(logits, masks)
        
        pred_prob = torch.sigmoid(logits)
        loss = loss + criterion_dice(pred_prob, masks)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if batch_idx % 20 == 0:
            print(f"   Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return np.mean(losses)


def validate(model, dataloader, criterion_bce, criterion_iou, device):
    """验证"""
    model.eval()
    losses = []
    all_bg_ious = []
    all_fg_ious = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            if masks.max() > 1:
                masks = masks.float() / 255.0
            else:
                masks = masks.float()
            
            logits = model(images)
            
            loss = criterion_bce(logits, masks) + criterion_iou(logits, masks)
            losses.append(loss.item())
            
            # 计算 IoU
            preds = (torch.sigmoid(logits) > 0.5).long()
            targets = (masks > 0.5).long()
            
            for i in range(preds.shape[0]):
                ious = compute_iou(preds[i], targets[i])
                if not np.isnan(ious[0]):
                    all_bg_ious.append(ious[0])
                if not np.isnan(ious[1]):
                    all_fg_ious.append(ious[1])
    
    mean_loss = np.mean(losses)
    bg_iou = np.mean(all_bg_ious) if all_bg_ious else 0.0
    fg_iou = np.mean(all_fg_ious) if all_fg_ious else 0.0
    mean_iou = (bg_iou + fg_iou) / 2
    
    return mean_loss, bg_iou, fg_iou, mean_iou


def predict_and_save(model, dataloader, save_dir, device, img_size):
    """预测并保存结果"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            for i in range(preds.shape[0]):
                pred_mask = preds[i, 0].cpu().numpy() * 255
                pred_mask = pred_mask.astype(np.uint8)
                
                save_path = os.path.join(save_dir, f'{idx * images.shape[0] + i}.png')
                cv2.imwrite(save_path, pred_mask)


def main():
    if not HAS_TRANSFORMERS:
        print("❌ 请先安装 transformers: pip install transformers")
        return
    
    args = parse_args()
    
    print("=" * 60)
    print(f"SegFormer Training (HuggingFace) - {args.model}")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    pred_dir = os.path.join(args.save_dir, 'pred')
    os.makedirs(pred_dir, exist_ok=True)
    
    # 创建模型
    print(f"\nCreating model: {args.model}")
    if args.local_weights:
        print(f"   Using local weights: {args.local_weights}")
    model = HFSegFormerWrapper(
        model_name=args.model,
        num_classes=args.num_classes,
        img_size=args.img_size,
        offline=args.offline,
        local_weights=args.local_weights
    )
    model = model.to(device)
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 数据集
    dataset_path = './data/orange_defect'
    print(f"\nLoading dataset from: {dataset_path}")
    
    trainset = OrangeDefectLoader(dataset_path, train=True, test=False, size=args.img_size, num_classes=2)
    testset = OrangeDefectLoader(dataset_path, train=False, test=True, size=args.img_size, num_classes=2)
    
    print(f"   Train samples: {len(trainset)}")
    print(f"   Test samples: {len(testset)}")
    
    # Oversampling
    if args.use_oversampling:
        print("\n==> Computing sample weights for oversampling...")
        sample_weights = []
        for i in range(len(trainset)):
            mask = trainset.masks[i]
            mask_binary = (mask > 0).astype(np.float32)
            defect_ratio = mask_binary.sum() / mask_binary.size
            weight = 1.0 + 4.0 * defect_ratio
            sample_weights.append(weight)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(trainset),
            replacement=True
        )
        train_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True)
        print(f"   Max weight: {max(sample_weights):.4f}")
        print(f"   Min weight: {min(sample_weights):.4f}")
    else:
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
    
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    # 损失函数
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_iou = IOU()
    criterion_dice = DiceLoss()
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 初始验证
    print("\n==> Initial validation...")
    val_loss, bg_iou, fg_iou, mean_iou = validate(model, test_loader, criterion_bce, criterion_iou, device)
    print(f"   Initial - Loss: {val_loss:.4f}, BG IoU: {bg_iou:.4f}, FG IoU: {fg_iou:.4f}, mIoU: {mean_iou:.4f}")
    
    # 训练
    print("\n==> Training...")
    best_miou = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, 
                                     criterion_bce, criterion_iou, criterion_dice, 
                                     device, epoch)
        
        # 验证
        val_loss, bg_iou, fg_iou, mean_iou = validate(model, test_loader, 
                                                       criterion_bce, criterion_iou, device)
        
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"BG IoU: {bg_iou:.4f}, FG IoU: {fg_iou:.4f}, mIoU: {mean_iou:.4f}, "
              f"Time: {elapsed:.2f}s")
        
        # 保存最佳模型
        if mean_iou > best_miou:
            best_miou = mean_iou
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"   ✅ Saved best model (mIoU: {best_miou:.4f})")
    
    # 加载最佳模型并评估
    print("\n" + "=" * 60)
    print("==> Testing best model...")
    print("=" * 60)
    
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, bg_iou, fg_iou, mean_iou = validate(model, test_loader, 
                                                   criterion_bce, criterion_iou, device)
    
    print(f"✅ Best Epoch: {best_epoch}")
    print(f"✅ Background IoU: {bg_iou:.4f}")
    print(f"✅ Foreground IoU: {fg_iou:.4f}")
    print(f"✅ Mean IoU (mIoU): {mean_iou:.4f}")
    
    # 保存预测结果
    print(f"\n==> Saving predictions to: {pred_dir}")
    predict_and_save(model, test_loader, pred_dir, device, args.img_size)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
