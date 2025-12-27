"""
训练框架（参数小写，直接写死）
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers.models import opt
import torch.nn.functional as F

from models import model_dict
from models.load_ckpt import load_checkpoint
from dataset.OrangeDefectDataloader import OrangeDefectLoader
from helper.util import AverageMeter
from helper.loss import IOU, BceDiceLoss
from helper.distillation_loss import *


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
    return iou_list

def validate(val_loader, model, is_teacher=False):
    # switch to evaluate mode
    if is_teacher:
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inp = batch[0].cuda()
                pred, _ = model(inp)
                pred_mask = pred[0, 0]  # shape: [H, W]
                pred_mask_bin = (pred_mask > 0.5).float() * 255

                # 转换为 uint8 图像并保存
                mask_img = Image.fromarray(pred_mask_bin.byte().cpu().numpy())
                mask_img.save(os.path.join(T_pred_save_folder, f"{val_img_names[i]}.png"))

        print("✅ 预测完成，结果保存在:", T_pred_save_folder)

        return evaluate_segmentation(T_pred_save_folder, gt_folder)
    else:
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inp = batch[0].cuda()
                inp = F.interpolate(inp, size=inp_size_S, mode='bilinear', align_corners=False)
                pred, _ = model(inp)
                pred_mask = pred[0, 0]  # shape: [H, W]
                pred_mask_bin = (pred_mask > 0.5).float() * 255

                # 转换为 uint8 图像并保存
                mask_img = Image.fromarray(pred_mask_bin.byte().cpu().numpy())
                mask_img.save(os.path.join(pred_save_folder, f"{val_img_names[i]}.png"))

        print("✅ 预测完成，结果保存在:", pred_save_folder)

        return evaluate_segmentation(pred_save_folder, gt_folder)

def main():
    # --------------------- 固定参数 ---------------------
    batch_size = 1
    num_workers = 8
    epochs_total = 500          # ✅ 总共500个epoch
    phase1_epochs = 200         # ✅ 前200个epoch
    phase2_epochs = epochs_total - phase1_epochs
    test_freq = 30
    img_size = 1024
    inp_size_T = 1024
    global inp_size_S
    inp_size_S = 1024

    model_name = 'sam2_adapter_tiny'
    if model_name == 'sam2_adapter_tiny':
        ckpt_path = './checkpoints/sam2.1_hiera_tiny.pt'
        model_S = model_dict[model_name](is_distill=True, inp_size=inp_size_S)
    else:
        ckpt_path = None
        model_S = None
        print("{} is not support".format(model_name))
        sys.exit()

    ckpt_path_T = './save/sam2_adapter_tiny/1024/onehot/sam2_adapter_tiny_best_loss0.3347.pth'
    model_T = model_dict[model_name](is_distill=True, inp_size=inp_size_T)

    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_iou = IOU()

    # ====== Phase-1: 1~200 epoch，1e-4 退火到 1e-6 ======
    def build_phase1_optim_sched(model):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,                   # 初始 1e-4
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase1_epochs, eta_min=1e-5, last_epoch=-1
        )
        return optimizer, scheduler

    # ====== Phase-2: 201~500 epoch，5e-5 + CosineAnnealingLR(T_max=50) ======
    def build_phase2_optim_sched(model):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,                   # 你指定的 0.00005
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False
        )
        # 按你的要求使用 CosineAnnealingLR(T_max=50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6, last_epoch=-1
        )
        # 如果你想要每50个epoch循环一次余弦，请改用下面两行（并注释掉上面的 CosineAnnealingLR）：
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=50, T_mult=1, eta_min=1e-6
        # )
        return optimizer, scheduler

    optimizer, scheduler = build_phase1_optim_sched(model_S)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model_T.cuda()
        model_S.cuda()
        criterion_bce.cuda()
        criterion_iou.cuda()
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if ckpt_path is not None:
        load_checkpoint(model_T, ckpt_path_T, device, model_name)
        load_checkpoint(model_S, ckpt_path, device)

    dataset_path = "./data/orange"
    global gt_folder
    gt_folder = "./data/orange/masks/"
    test_list = "./data/orange/imageset/test.txt"

    save_path = './save/distill/{}/{}/'.format(model_name, inp_size_S)
    os.makedirs(save_path, exist_ok=True)  # ✅ 确保存在
    global pred_save_folder
    pred_save_folder = './save/distill/{}/{}/pred/'.format(model_name, inp_size_S)
    os.makedirs(pred_save_folder, exist_ok=True)
    global T_pred_save_folder
    T_pred_save_folder = './save/distill/{}/{}/T_pred/'.format(model_name, inp_size_S)
    os.makedirs(T_pred_save_folder, exist_ok=True)

    # --------------------- 数据加载 ---------------------
    trainset = OrangeDefectLoader(dataset_path, train=True, test=False, size=img_size, num_classes=2)
    traindataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = OrangeDefectLoader(dataset_path, train=False, test=True, size=img_size, num_classes=2)
    testdataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)
    global val_img_names
    with open(test_list, "r") as f:
        val_img_names = [line.strip() for line in f.readlines()]

    # --------------------- 训练循环 ---------------------
    best_loss = 999
    log_file = open(os.path.join(save_path, "train_log.txt"), "a")

    # 初始验证
    log_line = "===================  validation of student  ===================\n"
    print(log_line.strip())
    log_file.write(log_line)
    iou = validate(testdataloader, model_S)
    log_line = '✅ Background IoU: {:.4f}\n✅ Foreground IoU: {:.4f}\n✅ Mean IoU (mIoU): {:.4f}\n'.format(
        iou[0], iou[1], iou[2])
    print(log_line.strip())
    log_file.write(log_line)

    log_line = "===================  validation of teacher  ===================\n"
    print(log_line.strip())
    log_file.write(log_line)
    iou = validate(testdataloader, model_T, True)
    log_line = '✅ Background IoU: {:.4f}\n✅ Foreground IoU: {:.4f}\n✅ Mean IoU (mIoU): {:.4f}\n'.format(
        iou[0], iou[1], iou[2])
    print(log_line.strip())
    log_file.write(log_line)

    print("==> training...")
    # 标记当前处于哪个Phase（1或2）
    current_phase = 1

    for epoch in range(1, epochs_total + 1):
        # Phase 切换点：从 201 epoch 开始进入 Phase-2
        if epoch == phase1_epochs + 1 and current_phase == 1:
            # 进入 Phase-2，重建优化器/调度器
            optimizer, scheduler = build_phase2_optim_sched(model_S)
            current_phase = 2
            print("==== Switched to Phase-2 optimizer/scheduler (epoch {}) ====".format(epoch))
            log_file.write("==== Switched to Phase-2 optimizer/scheduler (epoch {}) ====\n".format(epoch))

        current_lr = optimizer.param_groups[0]['lr']
        time1 = time.time()

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

            inp_S = F.interpolate(input, size=inp_size_S, mode='bilinear', align_corners=False)
            tgt_S = F.interpolate(target.float(), size=inp_size_S, mode='nearest')

            logit_S, features_S = model_S(inp_S)
            loss = criterion_bce(logit_S, tgt_S.float()) + criterion_iou(logit_S, tgt_S.float())
            loss_super.update(loss.item(), input.size(0))

            with torch.no_grad():
                logit_T, features_T = model_T(input)

            normal_loss = 0
            for i in range(len(features_S)):
                # if i == 7:
                #     break
                _, t_C, _, _ = features_T[i].shape
                _, s_C, _, _ = features_S[i].shape
                normal_loss_optimizer = NFD_loss_after_conv1x1(t_C, s_C)
                normal_loss_optimizer.cuda()
                normal_loss = normal_loss_optimizer(features_T[i], features_S[i])
                loss = loss + 5 * normal_loss
                loss_NFD.update(5*normal_loss.item(), input.size(0))



            losses.update(loss.item(), input.size(0))

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model_S.parameters(), max_norm=5.0)
            optimizer.step()

        # 调度器步进（每个epoch调用一次）
        scheduler.step()

        time2 = time.time()
        log_line = f"epoch {epoch} (phase {current_phase}), train, lr={current_lr:.6f}, mean loss {losses.avg:.3f}, total time {time2 - time1:.2f}\n"
        print(log_line.strip())
        log_file.write(log_line)

        # --------------------- 验证 ---------------------
        time1 = time.time()
        model_S.eval()
        val_losses = AverageMeter()
        with torch.no_grad():
            for idx, data in enumerate(testdataloader):
                input, target, onehot = data
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.unsqueeze(1).cuda()
                    onehot = onehot.cuda()
                inp_S = F.interpolate(input, size=inp_size_S, mode='bilinear', align_corners=False)
                tgt_S = F.interpolate(target.float(), size=inp_size_S, mode='nearest')

                logit, _ = model_S(inp_S)
                vloss = criterion_bce(logit, tgt_S.float()) + criterion_iou(logit, tgt_S.float())
                val_losses.update(vloss.item(), input.size(0))

        time2 = time.time()
        log_line = f"epoch {epoch}, val, mean loss {val_losses.avg:.3f}, total time {time2 - time1:.2f}\n"
        print(log_line.strip())
        log_file.write(log_line)

        if epoch % test_freq == 0:
            log_line = f"========  get student model iou (epoch {epoch})  ========:\n"
            print(log_line.strip())
            log_file.write(log_line)
            iou_list = validate(testdataloader, model_S)
            log_line = f"✅ Background IoU: {iou_list[0]:.4f}\n✅ Foreground IoU: {iou_list[1]:.4f}\n✅ Mean IoU (mIoU): {iou_list[2]:.4f}\n"
            print(log_line.strip())
            log_file.write(log_line)

        # 以验证集loss为准保存最佳
        if best_loss > val_losses.avg:
            best_loss = val_losses.avg
            torch.save(model_S.state_dict(), os.path.join(save_path, f"{model_name}_best.pth"))

    # 训练结束后对最佳模型做一次最终评测与重命名
    if os.path.exists(os.path.join(save_path, f"{model_name}_best.pth")):
        log_line = f"\n========  test of the best model  ========:\n"
        print(log_line.strip())
        log_file.write(log_line)
        load_checkpoint(model_S, os.path.join(save_path, f"{model_name}_best.pth"), device, model_name)
        iou_list = validate(testdataloader, model_S)
        log_line = f"✅ Background IoU: {iou_list[0]:.4f}\n✅ Foreground IoU: {iou_list[1]:.4f}\n✅ Mean IoU (mIoU): {iou_list[2]:.4f}\n"
        print(log_line.strip())
        log_file.write(log_line)

        os.rename(
            os.path.join(save_path, f"{model_name}_best.pth"),
            os.path.join(save_path, f'{model_name}_best_loss{best_loss:.4f}.pth')
        )

    log_file.close()


if __name__ == '__main__':
    main()
