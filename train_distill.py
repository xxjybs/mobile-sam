"""
Orange Defect Segmentation - Warmup -> Joint KD training
DataLoader outputs 1024; Student runs at 512; Teacher runs at 1024 then downsample to 512 for KD.
"""

import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import model_dict
from models.load_ckpt import load_checkpoint
from dataset.OrangeDefectDataloader import OrangeDefectLoader
from helper.util import AverageMeter, pred, Distill_one_epoch, train_one_epoch
from helper.loss import IOU


is_distill = True
inp_size_T = 1024
NFD_loss = False
NFD_loss_alpha = 1
KD_loss = True
KD_loss_alpha = 1
KD_T = 4
# ckpt_path_T = './save/sam2_adapter_tiny/1024/sam2_adapter_tiny_loss_0.404.pth'
ckpt_path_T = './save/sam2_adapter_tiny/1024/onehot/sam2_adapter_tiny_best_loss0.3347.pth'
model_name_T = 'sam2_adapter_tiny'
model_T = model_dict[model_name_T](is_distill=is_distill, inp_size=inp_size_T)


# config of student or model
inp_size_S = 1024
model_name_S = 'sam2_adapter_mamba_tiny'
# ckpt_path_S = './checkpoints/sam2.1_hiera_tiny.pt'
ckpt_path_S = None
model_S = model_dict[model_name_S](is_distill=is_distill, inp_size=inp_size_S)

# train set
batch_size = 1
num_workers = 8
epochs_total = 500
phase1_epochs = 200                 # 你的两阶段优化器/调度器切换点
phase2_epochs = epochs_total - phase1_epochs
test_freq = 30

# data set
img_size = 1024
is_onehot = True
dataset_path = "./data/orange"
gt_folder = "./data/orange/masks/"
test_list_file = "./data/orange/imageset/test.txt"

def main():
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_iou = IOU(is_onehot)


    def build_phase1_optim_sched(model):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase1_epochs, eta_min=1e-5, last_epoch=-1
        )
        return optimizer, scheduler


    def build_phase2_optim_sched(model):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False
        )
        # 按你的要求使用 CosineAnnealingLR(T_max=50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6, last_epoch=-1
        )
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

    if ckpt_path_S is not None:
        load_checkpoint(model_S, ckpt_path_S, device)
    if ckpt_path_T is not None:
        load_checkpoint(model_T, ckpt_path_T, device, model_name_T)

    pred_T = './save/distill/{}/{}/T_pred/'.format(model_name_S, inp_size_S)
    os.makedirs(pred_T, exist_ok=True)
    save_path = './save/distill/{}/{}/'.format(model_name_S, inp_size_S)
    os.makedirs(save_path, exist_ok=True)
    pred_S = './save/distill/{}/{}/pred/'.format(model_name_S, inp_size_S)
    os.makedirs(pred_S, exist_ok=True)

    trainset = OrangeDefectLoader(dataset_path, train=True, test=False, size=img_size, num_classes=2)
    traindataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = OrangeDefectLoader(dataset_path, train=False, test=True, size=img_size, num_classes=2)
    testdataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)
    with open(test_list_file, "r") as f:
        val_img_list = [line.strip() for line in f.readlines()]

    best_loss = 999
    log_file = open(os.path.join(save_path, "train_log.txt"), "a")

    # 初始验证
    log_line = "===================  validation of student  ===================\n"
    print(log_line.strip())
    log_file.write(log_line)
    iou = pred(testdataloader, model_S, val_img_list, pred_S, gt_folder, inp_size_S, is_onehot)
    log_line = '✅ Background IoU: {:.4f}\n✅ Foreground IoU: {:.4f}\n✅ Mean IoU (mIoU): {:.4f}\n'.format(
        iou[0], iou[1], iou[2])
    print(log_line.strip())
    log_file.write(log_line)

    log_line = "===================  validation of teacher  ===================\n"
    print(log_line.strip())
    log_file.write(log_line)
    iou = pred(testdataloader, model_T, val_img_list, pred_T, gt_folder, inp_size_T, is_onehot)
    log_line = '✅ Background IoU: {:.4f}\n✅ Foreground IoU: {:.4f}\n✅ Mean IoU (mIoU): {:.4f}\n'.format(
        iou[0], iou[1], iou[2])
    print(log_line.strip())
    log_file.write(log_line)


    print("==> training...")
    current_phase = 1

    for epoch in range(1, epochs_total + 1):

        if epoch == phase1_epochs + 1 and current_phase == 1:
            optimizer, scheduler = build_phase2_optim_sched(model_S)
            current_phase = 2
            print("==== Switched to Phase-2 optimizer/scheduler (epoch {}) ====".format(epoch))
            log_file.write("==== Switched to Phase-2 optimizer/scheduler (epoch {}) ====\n".format(epoch))

        current_lr = optimizer.param_groups[0]['lr']
        time1 = time.time()
        losses, loss_super, loss_NFD, loss_kd = Distill_one_epoch(model_S, model_T, traindataloader, img_size,
                                                                  inp_size_S, inp_size_T, is_onehot,
                                                                  NFD_loss, NFD_loss_alpha, KD_loss, KD_loss_alpha,
                                                                  KD_T, criterion_bce, criterion_iou,
                                                                  optimizer)
        scheduler.step()

        time2 = time.time()
        log_line = (
            f"epoch {epoch} (phase {current_phase}), train, lr={current_lr:.6f}, mean loss {losses.avg:.3f}, "
            f"super loss {loss_super.avg:.3f}, NFD loss {loss_NFD.avg:.3f}, kd loss {loss_kd.avg:.3f}, total time {time2 - time1:.2f}\n")

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
                if img_size != inp_size_S:
                    inp_S = F.interpolate(input, size=inp_size_S, mode='bilinear', align_corners=False)
                    tgt_S = F.interpolate(target.float(), size=inp_size_S, mode='nearest')
                    onehot_S = F.interpolate(onehot.float(), size=inp_size_S, mode='nearest')
                else:
                    inp_S = input
                    tgt_S = target
                    onehot_S = onehot

                logit_S, _ = model_S(inp_S)
                if is_onehot:
                    loss = criterion_bce(logit_S, onehot_S.float()) + criterion_iou(logit_S, onehot_S.float())
                else:
                    loss = criterion_bce(logit_S, tgt_S.float()) + criterion_iou(logit_S, tgt_S.float())
                val_losses.update(loss.item(), input.size(0))

        time2 = time.time()
        log_line = f"epoch {epoch}, val, mean loss {val_losses.avg:.3f}, total time {time2 - time1:.2f}\n"
        print(log_line.strip())
        log_file.write(log_line)

        if epoch % test_freq == 0:
            log_line = f"========  get student model iou (epoch {epoch})  ========:\n"
            print(log_line.strip())
            log_file.write(log_line)
            iou_list = pred(testdataloader, model_S, val_img_list, pred_S, gt_folder, inp_size_S, is_onehot)
            log_line = f"✅ Background IoU: {iou_list[0]:.4f}\n✅ Foreground IoU: {iou_list[1]:.4f}\n✅ Mean IoU (mIoU): {iou_list[2]:.4f}\n"
            print(log_line.strip())
            log_file.write(log_line)

        # 以验证集loss为准保存最佳
        if best_loss > val_losses.avg:
            best_loss = val_losses.avg
            torch.save(model_S.state_dict(), os.path.join(save_path, f"{model_name_S}_best.pth"))

    # 训练结束后对最佳模型做一次最终评测与重命名
    if os.path.exists(os.path.join(save_path, f"{model_name_S}_best.pth")):
        log_line = f"\n========  test of the best model  ========:\n"
        print(log_line.strip())
        log_file.write(log_line)
        load_checkpoint(model_S, os.path.join(save_path, f"{model_name_S}_best.pth"), device, model_name_S)
        iou_list = pred(testdataloader, model_S, val_img_list, pred_S, gt_folder, inp_size_S, is_onehot)
        log_line = f"✅ Background IoU: {iou_list[0]:.4f}\n✅ Foreground IoU: {iou_list[1]:.4f}\n✅ Mean IoU (mIoU): {iou_list[2]:.4f}\n"
        print(log_line.strip())
        log_file.write(log_line)

        os.rename(
            os.path.join(save_path, f"{model_name_S}_best.pth"),
            os.path.join(save_path, f'{model_name_S}_best_loss{best_loss:.4f}.pth')
        )

    log_file.close()


if __name__ == '__main__':
    main()


