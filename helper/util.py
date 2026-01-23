from __future__ import print_function

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from helper.distillation_loss import *

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# --------------------- 评估工具函数 ---------------------
def compute_iou_per_class(pred, gt, class_val):
    pred_mask = (pred == class_val)
    gt_mask = (gt == class_val)
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return np.nan
    return inter / union


def evaluate_segmentation(pred_dir, gt_dir):
    pred_names = sorted(os.listdir(pred_dir))
    ious_fg, ious_bg, iou_list = [], [], []

    for pred_name in tqdm(pred_names, desc="Evaluating"):
        pred_path = os.path.join(pred_dir, pred_name)
        gt_path = os.path.join(gt_dir, pred_name)

        if not os.path.exists(gt_path):
            print(f"[Warning] Ground truth not found for {pred_name}, skipped.")
            continue

        pred = np.array(Image.open(pred_path).convert('L'))
        gt_img = Image.open(gt_path).convert('L')
        if gt_img.size != (pred.shape[1], pred.shape[0]):  # (W, H)
            gt_img = gt_img.resize((pred.shape[1], pred.shape[0]), Image.NEAREST)
        gt = np.array(gt_img)

        pred = (pred > 127).astype(np.uint8)
        gt = (gt > 127).astype(np.uint8)

        iou_bg = compute_iou_per_class(pred, gt, 0)
        iou_fg = compute_iou_per_class(pred, gt, 1)

        if not np.isnan(iou_bg):
            ious_bg.append(iou_bg)
        if not np.isnan(iou_fg):
            ious_fg.append(iou_fg)

    if len(ious_bg) == 0 or len(ious_fg) == 0:
        print("❌ Error: No valid IoU computed.")
        return None

    mean_iou_bg = np.mean(ious_bg)
    mean_iou_fg = np.mean(ious_fg)
    miou = np.mean([mean_iou_bg, mean_iou_fg])
    iou_list = [mean_iou_bg, mean_iou_fg, miou]
    return iou_list


def pred(val_loader, model, val_img_names, save_path, gt_folder, inp_size, is_onehot=False, postprocessor=None):
    """
    模型预测并保存结果
    
    Args:
        val_loader: 验证数据加载器
        model: 模型
        val_img_names: 验证图像名称列表
        save_path: 保存路径
        gt_folder: 真值文件夹路径
        inp_size: 输入尺寸
        is_onehot: 是否使用one-hot编码
        postprocessor: 后处理器实例（可选），用于优化闭合区域分割
    """
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inp = batch[0].cuda()
            if inp.size(2) != inp_size or inp.size(3) != inp_size:
                inp = F.interpolate(inp, size=inp_size, mode='bilinear', align_corners=False)

            pred = model(inp)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            if is_onehot:
                pred_mask = pred.argmax(dim=1)[0]
            else:
                pred_mask = pred[0, 0]
            pred_mask_bin = (pred_mask > 0.5).float() * 255

            mask_img = Image.fromarray(pred_mask_bin.byte().cpu().numpy())
            
            # 应用后处理（如果提供了后处理器）
            if postprocessor is not None:
                mask_img = postprocessor(mask_img)
            
            mask_img.save(os.path.join(save_path, f"{val_img_names[i]}.png"))

    print("✅ 预测完成，结果保存在:", save_path)
    return evaluate_segmentation(save_path, gt_folder)

def Distill_one_epoch(
        model_S, model_T,
        traindataloader, img_size, inp_size_S, inp_size_T, is_onehot,
        NFD_loss, NFD_loss_alpha, KD_loss, KD_loss_alpha, KD_T,
        criterion_bce, criterion_iou, optimizer
):
    model_S.train()
    model_T.eval()
    losses = AverageMeter()
    loss_super = AverageMeter()
    loss_NFD = AverageMeter()
    loss_kd = AverageMeter()
    for idx, data in enumerate(traindataloader):
        input, target, onehot = data
        optimizer.zero_grad()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.unsqueeze(1).cuda()
            onehot = onehot.cuda()
        if img_size != inp_size_S:
            inp_S = F.interpolate(input, size=inp_size_S, mode='bilinear', align_corners=False)
            tgt_S = F.interpolate(target.float(), size=inp_size_S, mode='nearest')
            onehot_S = F.interpolate(onehot.float(), size=inp_size_S, mode='nearest')
        else:
            inp_S = input
            tgt_S = target
            onehot_S = onehot
        if img_size != inp_size_T:
            inp_T = F.interpolate(input, size=inp_size_T, mode='bilinear', align_corners=False)
            tgt_T = F.interpolate(target.float(), size=inp_size_T, mode='nearest')
            onehot_T = F.interpolate(onehot.float(), size=inp_size_T, mode='nearest')
        else:
            inp_T = input
            tgt_T = target
            onehot_T = onehot

        logit_S, features_S = model_S(inp_S)
        if is_onehot:
            loss = criterion_bce(logit_S, onehot_S.float()) + criterion_iou(logit_S, onehot_S.float())
        else:
            loss = criterion_bce(logit_S, tgt_S.float()) + criterion_iou(logit_S, tgt_S.float())
        loss_super.update(loss.item(), input.size(0))

        with torch.no_grad():
            logit_T, features_T = model_T(inp_T)

        if NFD_loss:
            normal_loss = 0
            for i in range(len(features_S)):
                _, t_C, _, _ = features_T[i].shape
                _, s_C, _, _ = features_S[i].shape
                normal_loss_optimizer = NFD_loss_after_conv1x1(t_C, s_C)
                normal_loss_optimizer.cuda()
                normal_loss = normal_loss_optimizer(features_T[i], features_S[i])
                loss = loss + NFD_loss_alpha * normal_loss
                loss_NFD.update(7 * NFD_loss_alpha * normal_loss.item(), input.size(0))

        if KD_loss and is_onehot:
            kd_loss = kl_pixel_loss(logit_T, logit_S, KD_loss_alpha, KD_T)
            loss = loss + KD_loss_alpha * kd_loss
            loss_kd.update(KD_loss_alpha * kd_loss.item(), input.size(0))

        losses.update(loss.item(), input.size(0))
        loss.backward()
        optimizer.step()
    return losses, loss_super, loss_NFD, loss_kd

# from helper.loss import BoundaryEdgeLoss
# criterion_boundary = BoundaryEdgeLoss(
#     edge_kernel=3,         # 边界厚度（近似）
#     bce_weight=1.0,
#     dice_weight=1.0,
#     binarize_thr=0.5,      # 下采样后是否阈值化
#     use_dynamic_pos_weight=True  # 动态平衡正负
# )

def train_one_epoch(
        model, traindataloader, is_onehot,
        criterion_bce, criterion_iou, optimizer,
        criterion_dice=None, criterion_tversky=None
):
    """
    单个epoch的训练循环
    Training loop for a single epoch
    
    Args:
        model: 模型
        traindataloader: 训练数据加载器
        is_onehot: 是否使用one-hot编码
        criterion_bce: BCE损失函数
        criterion_iou: IoU损失函数
        optimizer: 优化器
        criterion_dice: Dice损失函数（可选，用于增强大块区域分割）
        criterion_tversky: Tversky损失函数（可选，用于处理类别不平衡）
    """
    model.train()
    losses = AverageMeter()
    for idx, data in enumerate(traindataloader):
        input, target, onehot = data
        optimizer.zero_grad()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.unsqueeze(1).cuda()
            onehot = onehot.cuda()

        logit = model(input)
        # logit, boundary = model(input)

        if is_onehot:
            loss = criterion_bce(logit, onehot.float()) + criterion_iou(logit, onehot.float())
        else:
            loss = criterion_bce(logit, target.float()) + criterion_iou(logit, target.float()) #+ criterion_boundary(boundary, target)
        
        # 添加 Dice Loss（如果提供）- 对大块区域更敏感
        # Add Dice Loss (if provided) - more sensitive to large regions
        if criterion_dice is not None:
            pred_prob = torch.sigmoid(logit)
            if is_onehot:
                loss = loss + criterion_dice(pred_prob, onehot.float())
            else:
                loss = loss + criterion_dice(pred_prob, target.float())
        
        # 添加 Tversky Loss（如果提供）- 更关注漏检（假阴性）
        # Add Tversky Loss (if provided) - focuses more on false negatives
        if criterion_tversky is not None:
            if is_onehot:
                loss = loss + criterion_tversky(logit, onehot.float())
            else:
                loss = loss + criterion_tversky(logit, target.float())

        losses.update(loss.item(), input.size(0))
        loss.backward()
        optimizer.step()
    return losses


if __name__ == '__main__':

    pass
