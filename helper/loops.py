from __future__ import print_function, division

import os
import sys
import cv2
import time
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from .util import AverageMeter, accuracy
from helper.loss import *
from helper.distillation_loss import kl_pixel_loss as criterion_kd
def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, traindataloader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    criterionBCE = criterion_list[0]
    criterionIOU = criterion_list[1]
    criterion_NFD = criterion_list[2]

    model_s = module_list[0].train()
    model_t = module_list[-1].eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data in enumerate(traindataloader):
        input, target, onehot = data
        target = target.unsqueeze(1).float()
        data_time.update(time.time() - end)
        input_s = F.interpolate(input, size=(256, 256), mode="bilinear", align_corners=False)
        target_s = F.interpolate(target.float(), size=(256, 256), mode="nearest")

        if torch.cuda.is_available():
            input = input.cuda()
            # target = target.cuda()
            input_s = input_s.cuda()
            target_s = target_s.cuda()
            # onehot = onehot.cuda()


        # ===================forward=====================
        logit_s, feat_s = model_s.infer(input_s)
        with torch.no_grad():
            logit_t, feat_t = model_t.infer(input)
        loss_kd = criterion_kd(logit_t, target_s)
        loss_NFD = criterion_NFD(feat_s, feat_t)

        # logit_s = torch.sigmoid(logit_s)
        loss_BCE = criterionBCE(logit_s, target_s)
        loss_IOU = criterionIOU(logit_s, target_s)
        # loss = opt.gamma * loss_BCE + opt.alpha * loss_mse_feat + opt.beta * loss_kd
        loss = opt.gamma * (loss_BCE + loss_IOU) + opt.alpha * loss_NFD + opt.beta * loss_kd
        losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg

def distill_val(testdataloader, module_list, criterion_list, opt):
    """One epoch validation"""
    criterionBCE = criterion_list[0]
    criterionIOU = criterion_list[1]

    model_s = module_list[0].eval()
    model_t = module_list[-1].eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for idx, data in enumerate(testdataloader):

            input, target, onehot = data
            data_time.update(time.time() - end)

            target = target.unsqueeze(1).float()
            data_time.update(time.time() - end)
            input_s = F.interpolate(input, size=(256, 256), mode="bilinear", align_corners=False)
            target_s = F.interpolate(target.float(), size=(256, 256), mode="nearest")

            if torch.cuda.is_available():
                input = input.cuda()
                # target = target.cuda()
                input_s = input_s.cuda()
                target_s = target_s.cuda()
                # onehot = onehot.cuda()

            logit_s, feat_s = model_s.infer(input_s)
            # logit_s = torch.sigmoid(logit_s)
            loss_BCE = criterionBCE(logit_s, target_s)
            loss_IOU = criterionIOU(logit_s, target_s)
            loss = loss_BCE + loss_IOU
            losses.update(loss.item(), input.size(0))

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()
    return losses.avg

def compute_iou_per_class(pred, gt, class_val):
    pred_mask = (pred == class_val)
    gt_mask = (gt == class_val)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return np.nan
    return intersection / union

def evaluate_segmentation(pred_dir, gt_dir):
    pred_names = sorted(os.listdir(pred_dir))
    ious_fg = []
    ious_bg = []
    iou_list = []

    for pred_name in tqdm(pred_names, desc="Evaluating"):
        pred_path = os.path.join(pred_dir, pred_name)
        gt_path = os.path.join(gt_dir, pred_name)

        if not os.path.exists(gt_path):
            print(f"[Warning] Ground truth not found for {pred_name}, skipped.")
            continue

        pred = np.array(Image.open(pred_path).convert('L'))
        gt_img = Image.open(gt_path).convert('L')
        if gt_img.size != (pred.shape[1], pred.shape[0]):  # (width, height)
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
        return

    mean_iou_bg = np.mean(ious_bg)
    iou_list.append(mean_iou_bg)
    mean_iou_fg = np.mean(ious_fg)
    iou_list.append(mean_iou_fg)
    miou = np.mean([mean_iou_bg, mean_iou_fg])
    iou_list.append(miou)

    # print(f"✅ Background IoU: {mean_iou_bg:.4f}")
    # print(f"✅ Foreground IoU: {mean_iou_fg:.4f}")
    # print(f"✅ Mean IoU (mIoU): {miou:.4f}")
    return iou_list

def validate(val_loader, model, opt, is_teacher=False, postprocessor=None):
    """
    验证并保存预测结果
    
    Args:
        val_loader: 验证数据加载器
        model: 模型
        opt: 配置选项
        is_teacher: 是否为教师模型
        postprocessor: 后处理器实例（可选），用于优化闭合区域分割
    """
    # switch to evaluate mode
    model.eval()
    if is_teacher:
        path = opt.pred_t_folder
    else:
        path = opt.pred_s_folder
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inp = batch[0].cuda()
            inp_s = F.interpolate(inp, size=(256, 256), mode="bilinear", align_corners=False)
            if is_teacher:
                # pred = torch.sigmoid(model.infer(inp))  # [1, 1, H, W]
                pred, feat = model.infer(inp)  # [1, 1, H, W]
            else:
                pred, feat = model.infer(inp_s)
                # pred = torch.sigmoid(pred)
            pred_mask = pred[0, 0]  # shape: [H, W]
            pred_mask_bin = (pred_mask > 0.5).float() * 255

            # 转换为 uint8 图像并保存
            mask_img = Image.fromarray(pred_mask_bin.byte().cpu().numpy())
            
            # 应用后处理（如果提供了后处理器）
            if postprocessor is not None:
                mask_img = postprocessor(mask_img)
            
            mask_img.save(os.path.join(path, f"{opt.names[i]}.png"))

    print("✅ 预测完成，结果保存在:", path)

    return evaluate_segmentation(path, opt.gt_folder)
