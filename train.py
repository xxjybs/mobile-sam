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
import gc
from models import model_dict
from models.load_ckpt import load_checkpoint
from dataset.OrangeDefectDataloader import OrangeDefectLoader
from helper.util import AverageMeter, pred, Distill_one_epoch, train_one_epoch
from helper.loss import IOU, SmallDefectLoss
import random
import shutil

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

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn settings
    cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False

# def worker_init_fn():
#     seed = 42
#     np.random.seed(seed)
#     random.seed(seed)

def worker_init_fn(worker_id):
    """
    DataLoader worker init to make worker RNGs deterministic and different across workers.

    Called in each worker process with that worker's id (0..num_workers-1).
    """
    # torch.initial_seed() returns a different base seed for each worker/process.
    base_seed = torch.initial_seed()  # 64-bit value
    # make it fit into 32-bit numpy/python seeds and mix with worker_id to avoid collisions
    seed = (base_seed + worker_id) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)
    # If you use libraries that require their own seeds, set them here (e.g., for pillow/augmentations)
    try:
        import torchvision
        # torchvision transforms that rely on random will follow python/np seeds
    except Exception:
        pass


def cleanup():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


inp_size = 256
model_name = 'mobile_sam_adapter'
'''
[
'PVMNet': PVMNet,
'PVMNetPlus': PVMNet_plus,
'sam2_adapter_tiny': './checkpoints/sam2.1_hiera_tiny.pt',
'SegFormerB0': make_SegFormerB0,
'SegFormerB1': make_SegFormerB1,
'sam2_adapter_light': SAM2_Adapter_Light,
'unet': UNet,
'mobile_sam_adapter': './checkpoints/mobile_sam.pt',
]
'''
ckpt_path = './checkpoints/mobile_sam.pt'

# train set
batch_size = 4
num_workers = 8
epochs_total = 500
phase1_epochs = 200                # 你的两阶段优化器/调度器切换点
phase2_epochs = epochs_total - phase1_epochs
test_freq = 30

# data set
img_size = inp_size
is_onehot = False
dataset_path = "./data/orange"
gt_folder = "./data/orange/masks/"
test_list_file = "./data/orange/imageset/test.txt"

def main():
    set_seed(seed=42, deterministic=True)
    cleanup()
    model = model_dict[model_name]()
    
    # Use SmallDefectLoss for better dense small defect detection
    criterion_small_defect = SmallDefectLoss(
        focal_weight=1.0,
        tversky_weight=2.0,
        iou_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        tversky_alpha=0.3,  # Favor recall over precision
        tversky_beta=0.7    # Penalize false negatives more
    )
    
    # Keep original losses for compatibility
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_iou = IOU(is_onehot)

    optimizer, scheduler = build_phase1_optim_sched(model)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.cuda()
        criterion_small_defect.cuda()
        criterion_bce.cuda()
        criterion_iou.cuda()
        cudnn.benchmark = True
        print('cuda available')
    else:
        device = torch.device('cpu')

    if ckpt_path is not None:
        load_checkpoint(model, ckpt_path, device, model_name)

    exp_name = 'vit_change_2'
    save_path = './save/{}/{}/'.format(model_name, exp_name)
    os.makedirs(save_path, exist_ok=True)
    pred_path = './save/{}/{}/pred/'.format(model_name, exp_name)
    os.makedirs(pred_path, exist_ok=True)
    path = './models/MobileSAMv2/mobilesamv2/build_sam.py'
    copy_files = [
        './models/mobile_sam_adapter.py',
        './models/MobileSAMv2/mobilesamv2/build_sam.py',
        # './models/mobile_sam_adapter_change.py',
        # './models/MobileSAMv2/tinyvit/tiny_vit.py',
        './models/MobileSAMv2/tinyvit/tiny_vit_change.py',
        './models/MobileSAMv2/mobilesamv2/modeling/mask_decoder.py',
        './helper/util.py',
        './train.py'
    ]
    for file in copy_files:
        name = file.split('/')[-1]
        save_file = os.path.join(save_path, name)
        shutil.copy2(file, save_file)
    trainset = OrangeDefectLoader(dataset_path, train=True, test=False, size=img_size, num_classes=2)
    traindataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True)
    testset = OrangeDefectLoader(dataset_path, train=False, test=True, size=img_size, num_classes=2)
    testdataloader = DataLoader(testset, batch_size=1, shuffle=False,
                                num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True)

    with open(test_list_file, "r") as f:
        val_img_list = [line.strip() for line in f.readlines()]

    best_loss = 999
    log_file = open(os.path.join(save_path, "train_log.txt"), "a")

    # 初始验证
    log_line = "===================  validation of model  ===================\n"
    print(log_line.strip())
    log_file.write(log_line)
    iou = pred(testdataloader, model, val_img_list, pred_path, gt_folder, inp_size, is_onehot)
    log_line = '✅ Background IoU: {:.4f}\n✅ Foreground IoU: {:.4f}\n✅ Mean IoU (mIoU): {:.4f}\n'.format(
        iou[0], iou[1], iou[2])
    print(log_line.strip())
    log_file.write(log_line)

    print("==> training...")
    current_phase = 1

    for epoch in range(1, epochs_total + 1):

        if epoch == phase1_epochs + 1 and current_phase == 1:
            optimizer, scheduler = build_phase2_optim_sched(model)
            current_phase = 2
            # for name, para in model.named_parameters():
            #     if "mask_decoder" in name:
            #         para.requires_grad_(True)
            print("==== Switched to Phase-2 optimizer/scheduler (epoch {}) ====".format(epoch))
            log_file.write("==== Switched to Phase-2 optimizer/scheduler (epoch {}) ====\n".format(epoch))

        current_lr = optimizer.param_groups[0]['lr']
        time1 = time.time()
        losses = train_one_epoch(model, traindataloader, is_onehot, criterion_bce, criterion_iou, optimizer, criterion_small_defect)
        scheduler.step()

        time2 = time.time()
        log_line = (
            f"epoch {epoch} (phase {current_phase}), train, lr={current_lr:.6f}, mean loss {losses.avg:.3f}, total time {time2 - time1:.2f}\n")
        print(log_line.strip())
        log_file.write(log_line)

        # --------------------- 验证 ---------------------
        time1 = time.time()
        model.eval()
        val_losses = AverageMeter()
        with torch.no_grad():
            for idx, data in enumerate(testdataloader):
                input, target, onehot = data
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.unsqueeze(1).cuda()
                    onehot = onehot.cuda()

                logit = model(input)
                if is_onehot:
                    loss = criterion_bce(logit, onehot.float()) + criterion_iou(logit, onehot.float())
                else:
                    # Use SmallDefectLoss for validation too
                    loss = criterion_small_defect(logit, target.float())
                val_losses.update(loss.item(), input.size(0))

        time2 = time.time()
        log_line = f"epoch {epoch}, val, mean loss {val_losses.avg:.3f}, total time {time2 - time1:.2f}\n"
        print(log_line.strip())
        log_file.write(log_line)

        if epoch % test_freq == 0:
            log_line = f"========  get student model iou (epoch {epoch})  ========:\n"
            print(log_line.strip())
            log_file.write(log_line)
            iou_list = pred(testdataloader, model, val_img_list, pred_path, gt_folder, inp_size, is_onehot)
            log_line = f"✅ Background IoU: {iou_list[0]:.4f}\n✅ Foreground IoU: {iou_list[1]:.4f}\n✅ Mean IoU (mIoU): {iou_list[2]:.4f}\n"
            print(log_line.strip())
            log_file.write(log_line)

        # 以验证集loss为准保存最佳
        if best_loss > val_losses.avg:
            best_loss = val_losses.avg
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}_best.pth"))

    # 训练结束后对最佳模型做一次最终评测与重命名
    if os.path.exists(os.path.join(save_path, f"{model_name}_best.pth")):
        log_line = f"\n========  test of the best model  ========:\n"
        print(log_line.strip())
        log_file.write(log_line)
        load_checkpoint(model, os.path.join(save_path, f"{model_name}_best.pth"), device, model_name)
        iou_list = pred(testdataloader, model, val_img_list, pred_path, gt_folder, inp_size, is_onehot)
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


