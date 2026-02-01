"""
SAM2 Hiera Tiny Training Script for Orange Defect Segmentation
训练 SAM2-Hiera-Tiny 在脐橙数据集上，用于公平对比

Usage:
    python train_sam2.py
    python train_sam2.py --epochs 300 --batch_size 2
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
from models.sam2_adapter_tiny import SAM2_Adapter_T
from dataset.OrangeDefectDataloader import OrangeDefectLoader
from helper.util import AverageMeter
from helper.loss import IOU, DiceLoss
import random
import shutil
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Train SAM2-Hiera-Tiny on Orange Defect Dataset')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (SAM2 is memory intensive)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_oversampling', action='store_true', default=True,
                        help='Use oversampling for large defect samples')
    parser.add_argument('--pretrained', type=str, default='./checkpoints/sam2.1_hiera_tiny.pt',
                        help='Path to pretrained SAM2 checkpoint')
    return parser.parse_args()


def build_optimizer_scheduler(model, lr, epochs):
    """构建优化器和调度器"""
    # SAM2 使用较小的学习率，因为模型较大
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7, last_epoch=-1
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


def load_sam2_checkpoint(model, checkpoint_path, device):
    """加载 SAM2 预训练权重（部分加载）"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Pretrained checkpoint not found: {checkpoint_path}")
        print("   Training from scratch...")
        return
    
    print(f"Loading pretrained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model_state = model.state_dict()
    loaded_count = 0
    
    for name, param in state_dict.items():
        # 处理可能的 key 前缀不匹配
        clean_name = name
        if clean_name.startswith('module.'):
            clean_name = clean_name[7:]
        
        if clean_name in model_state:
            if model_state[clean_name].shape == param.shape:
                model_state[clean_name].copy_(param)
                loaded_count += 1
    
    print(f"✅ Loaded {loaded_count}/{len(model_state)} parameters from pretrained checkpoint")


def train_one_epoch_sam2(model, dataloader, is_onehot, criterion_bce, criterion_iou, 
                         optimizer, criterion_dice=None, device='cuda'):
    """SAM2 单 epoch 训练"""
    model.train()
    losses = AverageMeter()
    
    for idx, data in enumerate(dataloader):
        input_img, target, onehot = data
        if torch.cuda.is_available():
            input_img = input_img.to(device)
            target = target.unsqueeze(1).to(device)
            onehot = onehot.to(device)
        
        optimizer.zero_grad()
        
        try:
            # SAM2 前向传播 - 输出是 masks，可能需要选择最佳 mask
            output = model(input_img)
            
            # SAM2 可能输出多个 mask，选择第一个或最佳的
            if isinstance(output, tuple):
                logit = output[0]
            else:
                logit = output
            
            # 如果输出有多个通道（多个 mask），取第一个
            if logit.shape[1] > 1:
                logit = logit[:, 0:1, :, :]
            
            # 确保尺寸匹配
            if logit.shape[2:] != target.shape[2:]:
                logit = F.interpolate(logit, size=target.shape[2:], mode='bilinear', align_corners=False)
            
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
            
            # 梯度裁剪（SAM2 模型较大，防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            losses.update(loss.item(), input_img.size(0))
            
        except Exception as e:
            print(f"Error in batch {idx}: {e}")
            continue
    
    return losses


def pred_sam2(testdataloader, model, val_img_list, pred_path, gt_folder, inp_size, is_onehot, device='cuda'):
    """SAM2 预测和评估"""
    model.eval()
    iou_list = [[], []]
    
    with torch.no_grad():
        for idx, data in enumerate(testdataloader):
            input_img, target, onehot = data
            if torch.cuda.is_available():
                input_img = input_img.to(device)
            
            try:
                # 前向传播
                output = model(input_img)
                
                if isinstance(output, tuple):
                    logit = output[0]
                else:
                    logit = output
                
                if logit.shape[1] > 1:
                    logit = logit[:, 0:1, :, :]
                
                # 确保尺寸匹配
                if logit.shape[2:] != (inp_size, inp_size):
                    logit = F.interpolate(logit, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
                
                pred_mask = torch.sigmoid(logit)
                pred_mask = pred_mask.squeeze().cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                
                # 保存预测结果
                img_name = val_img_list[idx]
                save_name = img_name.replace('.jpg', '.png').replace('.JPG', '.png')
                cv2.imwrite(os.path.join(pred_path, save_name), pred_mask)
                
                # 计算 IoU
                gt_path = os.path.join(gt_folder, save_name)
                if os.path.exists(gt_path):
                    gt = cv2.imread(gt_path, 0)
                    gt = cv2.resize(gt, (inp_size, inp_size))
                    gt = (gt > 127).astype(np.uint8)
                    pred_binary = (pred_mask > 127).astype(np.uint8)
                    
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
            
            except Exception as e:
                print(f"Error in prediction {idx}: {e}")
                continue
    
    # 计算平均 IoU
    bg_iou = np.mean(iou_list[0]) if len(iou_list[0]) > 0 else 0
    fg_iou = np.mean(iou_list[1]) if len(iou_list[1]) > 0 else 0
    miou = (bg_iou + fg_iou) / 2
    
    return [bg_iou, fg_iou, miou]


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SAM2 Hiera Tiny Training for Orange Defect Segmentation")
    print("=" * 60)
    
    set_seed(seed=args.seed, deterministic=True)
    cleanup()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model_name = 'SAM2_Hiera_Tiny'
    print(f"Creating model: {model_name}")
    
    try:
        model = SAM2_Adapter_T(inp_size=args.img_size)
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # 加载预训练权重
    if args.pretrained and os.path.exists(args.pretrained):
        load_sam2_checkpoint(model, args.pretrained, device)
    
    # 损失函数
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_iou = IOU(is_onehot=False)
    criterion_dice = DiceLoss()
    
    # GPU 设置
    if torch.cuda.is_available():
        model.cuda()
        criterion_bce.cuda()
        criterion_iou.cuda()
        criterion_dice.cuda()
        cudnn.benchmark = True
        print('Using CUDA')
    
    # 优化器和调度器
    optimizer, scheduler = build_optimizer_scheduler(model, args.lr, args.epochs)
    
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
    copy_files = ['./train_sam2.py', './models/sam2_adapter_tiny.py', './helper/util.py']
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
                                     num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
    else:
        traindataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
    
    # 加载测试数据
    testset = OrangeDefectLoader(dataset_path, train=False, test=True, size=args.img_size, num_classes=2)
    testdataloader = DataLoader(testset, batch_size=1, shuffle=False,
                                num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
    
    with open(test_list_file, "r") as f:
        val_img_list = [line.strip() for line in f.readlines()]
    
    best_loss = 999
    log_file = open(os.path.join(save_path, "train_log.txt"), "a")
    
    # 初始验证
    log_line = "===================  initial validation  ===================\n"
    print(log_line.strip())
    log_file.write(log_line)
    
    try:
        iou = pred_sam2(testdataloader, model, val_img_list, pred_path, gt_folder, args.img_size, False, device)
        log_line = f'✅ Background IoU: {iou[0]:.4f}\n✅ Foreground IoU: {iou[1]:.4f}\n✅ Mean IoU (mIoU): {iou[2]:.4f}\n'
        print(log_line.strip())
        log_file.write(log_line)
    except Exception as e:
        print(f"Error in initial validation: {e}")
    
    print("==> Training...")
    test_freq = 50
    
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        time1 = time.time()
        
        losses = train_one_epoch_sam2(model, traindataloader, False, criterion_bce, criterion_iou, 
                                      optimizer, criterion_dice=criterion_dice, device=device)
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
                    input_img = input_img.to(device)
                    target = target.unsqueeze(1).to(device)
                
                try:
                    output = model(input_img)
                    if isinstance(output, tuple):
                        logit = output[0]
                    else:
                        logit = output
                    
                    if logit.shape[1] > 1:
                        logit = logit[:, 0:1, :, :]
                    
                    if logit.shape[2:] != target.shape[2:]:
                        logit = F.interpolate(logit, size=target.shape[2:], mode='bilinear', align_corners=False)
                    
                    loss = criterion_bce(logit, target.float()) + criterion_iou(logit, target.float())
                    val_losses.update(loss.item(), input_img.size(0))
                except:
                    continue
        
        time2 = time.time()
        log_line = f"epoch {epoch}, val, mean loss {val_losses.avg:.3f}, time {time2 - time1:.2f}s\n"
        print(log_line.strip())
        log_file.write(log_line)
        
        # 定期测试
        if epoch % test_freq == 0:
            log_line = f"========  IoU evaluation (epoch {epoch})  ========:\n"
            print(log_line.strip())
            log_file.write(log_line)
            iou_list = pred_sam2(testdataloader, model, val_img_list, pred_path, gt_folder, args.img_size, False, device)
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
        iou_list = pred_sam2(testdataloader, model, val_img_list, pred_path, gt_folder, args.img_size, False, device)
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
