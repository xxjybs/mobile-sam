import torch.nn as nn
from torch.nn import functional as F
import torch



class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        # self.bceloss = nn.BCEWithLogitsLoss()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class IOU(torch.nn.Module):
    def __init__(self, is_onehot: bool = False):
        super(IOU, self).__init__()
        self.is_onehot = is_onehot

    def _iou(self, pred, target, eps=1e-6):
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - ((inter + eps)/ (union +eps))

        return iou.mean()

    def forward(self, pred, target):
        if self.is_onehot:
            pred = torch.softmax(pred, dim=1)[:, 1:2, ...]
            target = target[:, 1:2, ...]
        else:
            pred = torch.sigmoid(pred)
        return self._iou(pred, target)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=0.5, wd=1.5):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_edge_from_mask(mask, k: int = 3):
    """
    从二值分割掩码生成边界图（形态学梯度：dilate - erode）
    mask: [B,1,H,W]，0/1（或[0,1]实数）
    return: edge ∈ [0,1]，同尺寸
    """
    pad = k // 2
    # dilate / erode 的可微近似
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
    edge = (dil - ero).clamp_(0, 1)
    return edge

class BoundaryEdgeLoss(nn.Module):
    """
    将 target 掩码下采样到 pred 的分辨率，提取边界，再与 pred 边界 logits 计算损失。
    组成： BCEWithLogits（带动态 pos_weight） + Dice（对边界更鲁棒）
    """
    def __init__(
        self,
        edge_kernel: int = 3,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        binarize_thr: float = 0.5,   # 下采样后的 target 是否二值化
        eps: float = 1e-6,
        use_dynamic_pos_weight: bool = True,
        fixed_pos_weight: float | None = None,
    ):
        super().__init__()
        self.edge_kernel = edge_kernel
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.binarize_thr = binarize_thr
        self.eps = eps
        self.use_dynamic_pos_weight = use_dynamic_pos_weight
        self.fixed_pos_weight = fixed_pos_weight

    def _dice_loss(self, logits, target):
        # logits -> prob
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(2,3))
        denom = prob.sum(dim=(2,3)) + target.sum(dim=(2,3)) + self.eps
        dice = 1.0 - (2.0 * inter + self.eps) / denom
        return dice.mean()

    def _bce_logits_loss(self, logits, target):
        if self.fixed_pos_weight is not None:
            pos_weight = torch.tensor(self.fixed_pos_weight, device=logits.device, dtype=logits.dtype)
        elif self.use_dynamic_pos_weight:
            # 动态计算：neg/pos，防止全负/极少正导致训练不稳定
            with torch.no_grad():
                pos = target.sum(dim=(2,3)) + self.eps     # [B,1]
                neg = target.numel() / target.shape[0] - pos  # 每张图总像素 - 正像素
                # 平均到 batch 级别，得到一个标量
                pos_weight = (neg / (pos + self.eps)).mean()
            pos_weight = pos_weight.clamp(1.0, 1000.0)      # 合理约束
        else:
            pos_weight = None

        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss

    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor):
        """
        pred_logits: [B,1,Hp,Wp]  —— 边界预测（logits）
        target_mask: [B,1,Ht,Wt]  —— 分割掩码（0/1）
        return: 标量损失
        """
        assert pred_logits.dim() == 4 and target_mask.dim() == 4
        assert pred_logits.size(1) == 1 and target_mask.size(1) == 1

        B, _, Hp, Wp = pred_logits.shape

        # 1) 下采样 target 到 pred 分辨率
        t = F.interpolate(target_mask.float(), size=(Hp, Wp),
                          mode='bilinear', align_corners=False)

        # 可选二值化（保持边缘更清晰；如需“软边缘”，可关闭此步骤）
        if self.binarize_thr is not None:
            t = (t >= self.binarize_thr).float()

        # 2) 从掩码生成边界标签
        edge_gt = make_edge_from_mask(t, k=self.edge_kernel)  # [B,1,Hp,Wp]

        # 3) 计算损失
        loss_bce  = self._bce_logits_loss(pred_logits, edge_gt)
        loss_dice = self._dice_loss(pred_logits, edge_gt)
        loss = self.bce_weight * loss_bce + self.dice_weight * loss_dice

        # 也可返回中间量以便可视化
        return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # 若模型未输出概率则需 Sigmoid
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky