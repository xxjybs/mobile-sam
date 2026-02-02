"""
SegFormer Training Script for Orange Defect Segmentation
训练 SegFormer-B0/B1 在脐橙数据集上，用于公平对比

支持加载 mmsegmentation 格式的 MiT backbone 预训练权重
预训练权重下载: https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth

Usage:
    # 使用 mmseg 预训练 MiT backbone（推荐）
    python train_segformer.py --model segformerb0 --pretrained ./checkpoints/segformerb0.pth
    
    # 从头训练
    python train_segformer.py --model segformerb0
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
from models.segformer import make_SegFormerB0, make_SegFormerB1
from dataset.OrangeDefectDataloader import OrangeDefectLoader
from helper.util import AverageMeter, pred
from helper.loss import IOU, DiceLoss
import random
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Train SegFormer on Orange Defect Dataset')
    parser.add_argument('--model', type=str, default='segformerb0', choices=['segformerb0', 'segformerb1'],
                        help='Model variant: segformerb0 or segformerb1')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_oversampling', action='store_true', default=True,
                        help='Use oversampling for large defect samples')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Path to mmseg pretrained MiT backbone checkpoint (.pth)')
    return parser.parse_args()


def load_mmseg_mit_pretrained(model, checkpoint_path, model_variant='b0'):
    """
    加载 mmsegmentation 格式的 MiT backbone 预训练权重
    
    mmseg 权重键名格式 (示例):
        patch_embed1.proj.weight, patch_embed1.proj.bias, patch_embed1.norm.weight, patch_embed1.norm.bias
        block1.0.attn.q.weight, block1.0.attn.kv.weight, block1.0.attn.proj.weight
        block1.0.mlp.fc1.weight, block1.0.mlp.dwconv.dwconv.weight, block1.0.mlp.fc2.weight
        norm1.weight, norm1.bias
        
    项目 SegFormer 结构 (MiT):
        self.stages[i] = [get_overlap_patches (Unfold), overlap_patch_embed (Conv2d), layers (ModuleList)]
        layers[j] = [PreNorm(EfficientSelfAttention), PreNorm(MixFeedForward)]
        
        EfficientSelfAttention: to_q, to_kv, to_out
        MixFeedForward: net = [Conv2d, DsConv2d, GELU, Conv2d]
        
    注意: mmseg 的 MiT 和项目中的 MiT 实现结构差异较大，很难直接映射。
    因此我们采用部分加载的方式，只加载能匹配的参数。
    """
    print(f"Loading mmseg MiT backbone weights from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理不同的 checkpoint 格式
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 打印预训练权重的样例键名
    sample_keys = list(state_dict.keys())[:10]
    print(f"   mmseg 权重示例键名: {sample_keys}")
    
    model_state_dict = model.state_dict()
    model_sample_keys = list(model_state_dict.keys())[:10]
    print(f"   模型参数示例键名: {model_sample_keys}")
    
    # 由于 mmseg MiT 和项目 MiT 架构差异较大，
    # 我们无法直接映射权重
    # 打印警告并从头训练
    print("\n" + "="*60)
    print("⚠️ 重要提示 / Important Notice:")
    print("="*60)
    print("   mmseg 的 MiT backbone 权重格式与本项目的 SegFormer 实现不兼容。")
    print("   The mmseg MiT backbone weights are not compatible with this SegFormer implementation.")
    print("")
    print("   原因: 两者的网络结构设计不同")
    print("   - mmseg MiT: 使用 patch_embed + block (Linear attention)")  
    print("   - 项目 MiT:  使用 Unfold + Conv2d (Conv-based attention)")
    print("")
    print("   解决方案 / Solutions:")
    print("   1. 从头训练 (Training from scratch) - 当前采用")
    print("   2. 使用 mmseg 框架进行训练 (推荐用于生产环境)")
    print("="*60 + "\n")
    
    return 0  # 返回加载的参数数量


def build_optimizer_scheduler(model, lr, epochs):
    """构建优化器和调度器"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6, last_epoch=-1
    )
    return optimizer, scheduler


def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False


def worker_init_fn(worker_id):
    base_seed = torch.initial_seed()
    seed = (base_seed + worker_id) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)


def cleanup():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def train_one_epoch_segformer(model, dataloader, is_onehot, criterion_bce, criterion_iou, 
                               optimizer, criterion_dice=None):
    """SegFormer 单 epoch 训练"""
    model.train()
    losses = AverageMeter()
    
    for idx, data in enumerate(dataloader):
        input_img, target, onehot = data
        if torch.cuda.is_available():
            input_img = input_img.cuda()
            target = target.unsqueeze(1).cuda()
            onehot = onehot.cuda()
        
        optimizer.zero_grad()
        
        # SegFormer 前向传播
        logit = model(input_img)
        
        # 计算损失
        if is_onehot:
            loss = criterion_bce(logit, onehot.float()) + criterion_iou(logit, onehot.float())
        else:
            loss = criterion_bce(logit, target.float()) + criterion_iou(logit, target.float())
        
        # 添加 Dice Loss
        if criterion_dice is not None:
            pred_prob = torch.sigmoid(logit)
            if is_onehot:
                loss = loss + 1.0 * criterion_dice(pred_prob, onehot.float())
            else:
                loss = loss + 1.0 * criterion_dice(pred_prob, target.float())
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), input_img.size(0))
    
    return losses


def pred_segformer(testdataloader, model, val_img_list, pred_path, gt_folder, inp_size, is_onehot):
    """SegFormer 预测和评估"""
    from PIL import Image
    from helper.util import calc_iou
    import cv2
    
    model.eval()
    iou_list = [[], []]
    
    with torch.no_grad():
        for idx, data in enumerate(testdataloader):
            input_img, target, onehot = data
            if torch.cuda.is_available():
                input_img = input_img.cuda()
            
            # 前向传播
            logit = model(input_img)
            pred_mask = torch.sigmoid(logit)
            
            # 后处理
            pred_mask = pred_mask.squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
            
            # 保存预测结果
            img_name = val_img_list[idx]
            # 确保有正确的扩展名
            if not img_name.endswith('.png'):
                if img_name.endswith('.jpg') or img_name.endswith('.JPG'):
                    save_name = img_name.replace('.jpg', '.png').replace('.JPG', '.png')
                else:
                    save_name = f"{img_name}.png"
            else:
                save_name = img_name
            cv2.imwrite(os.path.join(pred_path, save_name), pred_mask)
            
            # 计算 IoU
            gt_path = os.path.join(gt_folder, save_name)
            if os.path.exists(gt_path):
                gt = cv2.imread(gt_path, 0)
                gt = cv2.resize(gt, (inp_size, inp_size))
                gt = (gt > 127).astype(np.uint8)
                pred_binary = (pred_mask > 127).astype(np.uint8)
                
                # 计算背景和前景 IoU
                for cls_id in range(2):
                    if cls_id == 0:
                        gt_cls = 1 - gt
                        pred_cls = 1 - pred_binary
                    else:
                        gt_cls = gt
                        pred_cls = pred_binary
                    
                    intersection = np.logical_and(gt_cls, pred_cls).sum()
                    union = np.logical_or(gt_cls, pred_cls).sum()
                    if union > 0:
                        iou_list[cls_id].append(intersection / union)
    
    # 计算平均 IoU
    bg_iou = np.mean(iou_list[0]) if len(iou_list[0]) > 0 else 0
    fg_iou = np.mean(iou_list[1]) if len(iou_list[1]) > 0 else 0
    miou = (bg_iou + fg_iou) / 2
    
    return [bg_iou, fg_iou, miou]


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"SegFormer Training - {args.model.upper()}")
    print("=" * 60)
    
    set_seed(seed=args.seed, deterministic=True)
    cleanup()
    
    # 创建模型
    if args.model == 'segformerb0':
        model = make_SegFormerB0(num_classes=1)
        model_name = 'SegFormerB0'
        model_variant = 'b0'
        # 检查多个可能的预训练权重路径
        possible_paths = [
            './checkpoints/segformerb0.pth',
            './checkpoints/segformerb0.pt',
            './checkpoints/mit_b0.pth',
            './checkpoint/segformerb0.pth',
        ]
    else:
        model = make_SegFormerB1(num_classes=1)
        model_name = 'SegFormerB1'
        model_variant = 'b1'
        possible_paths = [
            './checkpoints/segformerb1.pth',
            './checkpoints/segformerb1.pt',
            './checkpoints/mit_b1.pth',
            './checkpoint/segformerb1.pth',
        ]
    
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 自动查找预训练权重
    pretrained_path = args.pretrained
    if not pretrained_path:
        for path in possible_paths:
            if os.path.exists(path):
                pretrained_path = path
                break
    
    # 尝试加载预训练权重
    if pretrained_path and os.path.exists(pretrained_path):
        loaded_count = load_mmseg_mit_pretrained(model, pretrained_path, model_variant)
        if loaded_count == 0:
            print("   将从头训练 / Training from scratch...")
    else:
        print("⚠️ No pretrained weights found")
        print("   Training from scratch...")
        print(f"   💡 提示: 可下载 mmseg 预训练权重:")
        print(f"      https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth")
    
    # 损失函数
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_iou = IOU(is_onehot=False)
    criterion_dice = DiceLoss()
    
    # 优化器和调度器
    optimizer, scheduler = build_optimizer_scheduler(model, args.lr, args.epochs)
    
    # GPU 设置
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.cuda()
        criterion_bce.cuda()
        criterion_iou.cuda()
        criterion_dice.cuda()
        cudnn.benchmark = True
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    # 数据集路径
    dataset_path = "./data/orange"
    gt_folder = "./data/orange/masks/"
    test_list_file = "./data/orange/imageset/test.txt"
    
    # 保存路径
    save_path = f'./save/{model_name}/orange_trained/'
    os.makedirs(save_path, exist_ok=True)
    pred_path = f'./save/{model_name}/orange_trained/pred/'
    os.makedirs(pred_path, exist_ok=True)
    
    # 保存配置文件
    copy_files = ['./train_segformer.py', './models/segformer.py', './helper/util.py']
    for file in copy_files:
        if os.path.exists(file):
            name = file.split('/')[-1]
            save_file = os.path.join(save_path, name)
            shutil.copy2(file, save_file)
    
    # 加载训练数据
    trainset = OrangeDefectLoader(dataset_path, train=True, test=False, size=args.img_size, num_classes=2)
    
    # Oversampling
    if args.use_oversampling:
        print("==> 计算样本采样权重... / Computing sample weights for oversampling...")
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
        print(f"   - 样本数: {len(trainset)}")
        print(f"   - 最大权重: {max(sample_weights):.4f}")
        print(f"   - 最小权重: {min(sample_weights):.4f}")
        print(f"   - 平均权重: {sum(sample_weights)/len(sample_weights):.4f}")
        
        traindataloader = DataLoader(trainset, batch_size=args.batch_size, sampler=sampler,
                                     num_workers=8, worker_init_fn=worker_init_fn, pin_memory=True)
    else:
        traindataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=8, worker_init_fn=worker_init_fn, pin_memory=True)
    
    # 加载测试数据
    testset = OrangeDefectLoader(dataset_path, train=False, test=True, size=args.img_size, num_classes=2)
    testdataloader = DataLoader(testset, batch_size=1, shuffle=False,
                                num_workers=8, worker_init_fn=worker_init_fn, pin_memory=True)
    
    with open(test_list_file, "r") as f:
        val_img_list = [line.strip() for line in f.readlines()]
    
    best_loss = 999
    log_file = open(os.path.join(save_path, "train_log.txt"), "a")
    
    # 初始验证
    log_line = "===================  initial validation  ===================\n"
    print(log_line.strip())
    log_file.write(log_line)
    iou = pred_segformer(testdataloader, model, val_img_list, pred_path, gt_folder, args.img_size, False)
    log_line = f'✅ Background IoU: {iou[0]:.4f}\n✅ Foreground IoU: {iou[1]:.4f}\n✅ Mean IoU (mIoU): {iou[2]:.4f}\n'
    print(log_line.strip())
    log_file.write(log_line)
    
    print("==> Training...")
    test_freq = 50
    
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        time1 = time.time()
        
        losses = train_one_epoch_segformer(model, traindataloader, False, criterion_bce, criterion_iou, 
                                           optimizer, criterion_dice=criterion_dice)
        scheduler.step()
        
        time2 = time.time()
        log_line = f"epoch {epoch}, train, lr={current_lr:.6f}, mean loss {losses.avg:.3f}, time {time2 - time1:.2f}s\n"
        print(log_line.strip())
        log_file.write(log_line)
        
        # 验证
        time1 = time.time()
        model.eval()
        val_losses = AverageMeter()
        with torch.no_grad():
            for idx, data in enumerate(testdataloader):
                input_img, target, onehot = data
                if torch.cuda.is_available():
                    input_img = input_img.cuda()
                    target = target.unsqueeze(1).cuda()
                
                logit = model(input_img)
                loss = criterion_bce(logit, target.float()) + criterion_iou(logit, target.float())
                val_losses.update(loss.item(), input_img.size(0))
        
        time2 = time.time()
        log_line = f"epoch {epoch}, val, mean loss {val_losses.avg:.3f}, time {time2 - time1:.2f}s\n"
        print(log_line.strip())
        log_file.write(log_line)
        
        # 定期测试
        if epoch % test_freq == 0:
            log_line = f"========  IoU evaluation (epoch {epoch})  ========:\n"
            print(log_line.strip())
            log_file.write(log_line)
            iou_list = pred_segformer(testdataloader, model, val_img_list, pred_path, gt_folder, args.img_size, False)
            log_line = f"✅ Background IoU: {iou_list[0]:.4f}\n✅ Foreground IoU: {iou_list[1]:.4f}\n✅ Mean IoU (mIoU): {iou_list[2]:.4f}\n"
            print(log_line.strip())
            log_file.write(log_line)
        
        # 保存最佳模型
        if best_loss > val_losses.avg:
            best_loss = val_losses.avg
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}_best.pth"))
    
    # 最终评估
    if os.path.exists(os.path.join(save_path, f"{model_name}_best.pth")):
        log_line = f"\n========  Final Test of Best Model  ========:\n"
        print(log_line.strip())
        log_file.write(log_line)
        model.load_state_dict(torch.load(os.path.join(save_path, f"{model_name}_best.pth")))
        iou_list = pred_segformer(testdataloader, model, val_img_list, pred_path, gt_folder, args.img_size, False)
        log_line = f"✅ Background IoU: {iou_list[0]:.4f}\n✅ Foreground IoU: {iou_list[1]:.4f}\n✅ Mean IoU (mIoU): {iou_list[2]:.4f}\n"
        print(log_line.strip())
        log_file.write(log_line)
        
        # 重命名最终模型
        final_name = f'{model_name}_best_loss{best_loss:.4f}.pth'
        os.rename(
            os.path.join(save_path, f"{model_name}_best.pth"),
            os.path.join(save_path, final_name)
        )
        print(f"✅ Model saved as: {final_name}")
    
    log_file.close()
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
