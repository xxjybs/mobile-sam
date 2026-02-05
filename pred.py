"""
训练框架（参数小写，直接写死）
"""

import os
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
from helper.loss import IOU
from helper.postprocess import PostProcessor

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

def validate(val_loader, model, postprocessor=None):
    """
    验证并保存预测结果
    
    Args:
        val_loader: 验证数据加载器
        model: 模型
        postprocessor: 后处理器实例（可选），用于优化闭合区域分割
    """
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inp = batch[0].cuda()
            pred = model(inp)
            pred = torch.sigmoid(pred)
            pred_mask = pred[0, 0]  # shape: [H, W]
            pred_mask_bin = (pred_mask > 0.5).float() * 255

            # 转换为 uint8 图像并保存
            mask_img = Image.fromarray(pred_mask_bin.byte().cpu().numpy())
            
            # 应用后处理（如果提供了后处理器）
            if postprocessor is not None:
                mask_img = postprocessor(mask_img)
            
            mask_img.save(os.path.join(pred_save_folder, f"{val_img_names[i]}.png"))

    print("✅ 预测完成，结果保存在:", pred_save_folder)

    return evaluate_segmentation(pred_save_folder, gt_folder)


def main():
    # --------------------- 固定参数 ---------------------

    num_workers = 8
    model_name = 'sam2_adapter_tiny'
    '''
    choices=['PVMNet', 'sam2_adapter_tiny', 'SegFormerB0', 'SegFormerB1']
    '''
    
    # --------------------- 后处理配置 ---------------------
    # 针对密集连续大块缺陷优化的后处理器
    # Post-processor optimized for dense continuous large defects
    enable_postprocess = True  # 是否启用后处理
    
    if enable_postprocess:
        # 可选预设配置:
        # - PostProcessor.for_large_defects(): 针对大块密集缺陷（激进，可能降低IoU）
        # - PostProcessor.for_edge_defects(): 针对边缘小缺陷
        # - PostProcessor.balanced(): 平衡配置
        # - PostProcessor.conservative(): 保守配置，专注填充内部孔洞（推荐）
        # - PostProcessor.minimal(): 最小化处理，仅填充内部孔洞
        # 
        # 推荐使用 conservative() 或 minimal() 配置以提高IoU
        postprocessor = PostProcessor.conservative()
        
        # 或自定义配置:
        # postprocessor = PostProcessor(
        #     enable_closing=True,           # 形态学闭操作，填充小孔洞
        #     closing_kernel_size=3,         # 闭操作核大小（3更保守）
        #     closing_iterations=1,          # 闭操作迭代次数
        #     enable_hole_fill=True,         # 填充闭合区域内孔洞（关键）
        #     enable_small_region_removal=False,  # 不移除小区域（保留边缘检测）
        #     min_region_area=50,            # 最小区域面积阈值
        #     enable_region_connection=False, # 不连接相邻区域（避免过度连接）
        #     connection_max_gap=5           # 连接区域最大间隙
        # )
    else:
        postprocessor = None
    
    ckpt_path = None

    if model_name == 'sam2_adapter_tiny':
        ckpt_path = './save/sam2_adapter_tiny/sam2_adapter_tiny_loss_0.404.pth'
    elif model_name == 'SegFormerB0':
        ckpt_path = './checkpoints/mit_b0.pth'
    elif model_name == 'SegFormerB1':
        ckpt_path = './checkpoints/mit_b1.pth'
    model = model_dict[model_name]()

    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_iou = IOU()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.cuda()
        criterion_bce.cuda()
        criterion_iou.cuda()
        cudnn.benchmark = True

    load_checkpoint(model, ckpt_path, device, model_name)

    dataset_path = "./data/orange"
    global gt_folder
    gt_folder = "./data/orange/masks/"
    test_list = "./data/orange/imageset/test.txt"

    global pred_save_folder
    pred_save_folder = './save/{}/pred/'.format(model_name)
    os.makedirs(pred_save_folder, exist_ok=True)

    # --------------------- 数据加载 ---------------------
    testset = OrangeDefectLoader(dataset_path, train=False, test=True, size=1024, num_classes=2)
    testdataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)
    global val_img_names
    with open(test_list, "r") as f:
        val_img_names = [line.strip() for line in f.readlines()]

    time1 = time.time()
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        for idx, data in enumerate(testdataloader):
            input, target, onehot = data
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.unsqueeze(1).cuda()
                onehot = onehot.cuda()

            logit = model(input)
            loss = criterion_bce(logit, target.float()) + criterion_iou(logit, target.float())
            losses.update(loss.item(), input.size(0))

    time2 = time.time()
    log_line = f"val, mean loss {losses.avg:.3f}, total time {time2 - time1:.2f}\n"
    print(log_line.strip())


    iou = validate(testdataloader, model, postprocessor)
    log_line = '✅ Background IoU: {:.4f}\n✅ Foreground IoU: {:.4f}\n✅ Mean IoU (mIoU): {:.4f}\n'.format(
        iou[0], iou[1], iou[2])
    print(log_line.strip())

if __name__ == '__main__':
    main()
