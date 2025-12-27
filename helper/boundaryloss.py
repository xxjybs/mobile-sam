import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 边界提取工具 ----------
def onehot_to_edge(target_onehot: torch.Tensor,
                   method: str = "sobel",
                   edge_width: int = 3) -> torch.Tensor:
    """
    target_onehot: (B, C, H, W), 0/1 的 one-hot 语义标签
    method: 'sobel' 或 'laplace'
    edge_width: 用 max-pool 对边界进行加粗（像素）
    return: (B, 1, H, W) 的 0/1 边界图（float）
    """
    assert target_onehot.dim() == 4, "target_onehot must be (B, C, H, W)"
    B, C, H, W = target_onehot.shape
    device = target_onehot.device
    dtype = target_onehot.dtype

    # 固定卷积核（不训练），对每个类别通道做 depthwise
    if method.lower() == "sobel":
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        pad = 1
        # depthwise
        gx = F.conv2d(F.pad(target_onehot, (pad, pad, pad, pad), mode="replicate"),
                      kx.repeat(C, 1, 1, 1), groups=C)
        gy = F.conv2d(F.pad(target_onehot, (pad, pad, pad, pad), mode="replicate"),
                      ky.repeat(C, 1, 1, 1), groups=C)
        grad = torch.abs(gx) + torch.abs(gy)  # L1 近似
        edge = (grad > 0).float()             # 二值化

    elif method.lower() in ["laplace", "laplacian"]:
        kl = torch.tensor([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=dtype, device=device).view(1, 1, 3, 3)
        pad = 1
        gl = F.conv2d(F.pad(target_onehot, (pad, pad, pad, pad), mode="replicate"),
                      kl.repeat(C, 1, 1, 1), groups=C)
        edge = (torch.abs(gl) > 0).float()

    else:
        raise ValueError("method must be 'sobel' or 'laplace'.")

    # 合并各类别的边界（取最大/并集）
    edge = edge.max(dim=1, keepdim=True).values  # (B,1,H,W)

    # 用 max-pool 加宽边界宽度
    if edge_width and edge_width > 1:
        r = int(edge_width)
        pad = r // 2
        edge = F.max_pool2d(edge, kernel_size=r, stride=1, padding=pad)

    edge = edge.clamp(0, 1)
    return edge


# ---------- Dice（logits 版） ----------
def dice_loss_with_logits(logits: torch.Tensor,
                          targets: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    logits: (B,1,H,W) 未过 Sigmoid
    targets: (B,1,H,W) 0/1
    """
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    num = 2.0 * (probs * targets).sum(dims)
    den = (probs.pow(2) + targets.pow(2)).sum(dims) + eps
    dice = 1.0 - (num / den)
    return dice.mean()


# ---------- 边界损失主类 ----------
class BoundaryLoss(nn.Module):
    """
    组合损失： L = w_bce * BCEWithLogits(edge_logits, edge_target) + w_dice * Dice
    - 自动从 one-hot 语义图生成边界真值（sobel/laplace）
    - 支持自动 pos_weight 以缓解正负不平衡
    """
    def __init__(self,
                 method: str = "sobel",
                 edge_width: int = 3,
                 w_bce: float = 1.0,
                 w_dice: float = 1.0,
                 auto_pos_weight: bool = True,
                 pos_weight: float | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.method = method
        self.edge_width = edge_width
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.auto_pos_weight = auto_pos_weight
        self.fixed_pos_weight = pos_weight
        self.reduction = reduction

    @torch.no_grad()
    def _make_pos_weight(self, target_edge: torch.Tensor, eps: float = 1e-6):
        # target_edge: (B,1,H,W) 0/1
        pos = target_edge.sum()
        neg = target_edge.numel() - pos
        return (neg + eps) / (pos + eps)

    def forward(self,
                target_onehot: torch.Tensor,
                edge_logits: torch.Tensor) -> torch.Tensor:
        """
        target_onehot: (B, C, H, W), one-hot 语义标签
        edge_logits:   (B, 1, H, W), 未过 Sigmoid 的边界预测
        """
        # 1) 生成边界真值
        with torch.no_grad():
            target_edge = onehot_to_edge(target_onehot, method=self.method, edge_width=self.edge_width)

        # 2) BCE
        if self.fixed_pos_weight is not None:
            pw = torch.tensor(self.fixed_pos_weight, device=edge_logits.device, dtype=edge_logits.dtype)
        elif self.auto_pos_weight:
            pw = self._make_pos_weight(target_edge).to(edge_logits.device).to(edge_logits.dtype)
        else:
            pw = None

        bce = F.binary_cross_entropy_with_logits(edge_logits, target_edge, pos_weight=pw, reduction=self.reduction)

        # 3) Dice
        dice = dice_loss_with_logits(edge_logits, target_edge)

        # 4) 总损失
        loss = self.w_bce * bce + self.w_dice * dice
        return loss
