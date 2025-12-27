# tile_wrapper.py
# 全局二维位置编码 + 分块 (halo) 推理/训练 适配器
# 兼容返回 Tensor 或 dict({"logits", "edge_logits"}) 的分割模型

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Tuple, List, Dict, Union, Optional


# ----------------------------
# 1) 全局二维 Sin/Cos 位置编码
# ----------------------------
class SinCos2DPositionalEncoding(nn.Module):
    """
    生成与输入同通道数 (C) 的 2D 正余弦位置编码并加到输入上：
        x <- x + alpha * PE(C,H,W)
    - learnable_scale: 是否让 alpha 可学习
    - temperature: 频率基数
    """
    def __init__(self, channels: int, temperature: float = 10000.0, learnable_scale: bool = True):
        super().__init__()
        self.channels = channels
        self.temperature = temperature
        self.alpha = nn.Parameter(torch.tensor(1.0)) if learnable_scale else None

    @torch.no_grad()
    def _grid(self, H, W, device, dtype):
        y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return yy, xx

    @torch.no_grad()
    def _pe(self, H, W, device, dtype):
        C = self.channels
        half = C // 2
        yy, xx = self._grid(H, W, device, dtype)
        # y、x 两半通道分别使用一组频率
        invy = torch.pow(self.temperature, -torch.arange(0, max(1, half), 2, device=device, dtype=dtype) / max(1, half))
        invx = torch.pow(self.temperature, -torch.arange(0, max(1, C - half), 2, device=device, dtype=dtype) / max(1, C - half))
        py = torch.stack([torch.sin(yy[..., None]*invy), torch.cos(yy[..., None]*invy)], dim=-1).view(H, W, -1)
        px = torch.stack([torch.sin(xx[..., None]*invx), torch.cos(xx[..., None]*invx)], dim=-1).view(H, W, -1)
        py = py[:, :, :half]
        px = px[:, :, : (C - half)]
        pe = torch.cat([py, px], dim=-1).permute(2, 0, 1).contiguous()  # (C,H,W)
        if pe.shape[0] < C:  # 若C为奇数，最后补零
            pad = torch.zeros(C - pe.shape[0], H, W, device=device, dtype=dtype)
            pe = torch.cat([pe, pad], dim=0)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        pe = self._pe(H, W, x.device, x.dtype).unsqueeze(0).expand(B, -1, -1, -1)  # (B,C,H,W)
        if self.alpha is not None:
            pe = self.alpha * pe
        return x + pe


# ------------------------------------
# 2) 分块与拼接（支持 halo 重叠去缝）
# ------------------------------------
def tile_with_halo(x: torch.Tensor, tile_h: int, tile_w: int, halo: int = 16):
    """
    将 (B,C,H,W) 分块，块边缘包含 halo 重叠区。
    返回：tiles(list of Tensor)、coords(list of 6元组)、整图尺寸(H,W)
    coords: (top, left, t0, l0, t1, l1)
        - (top, left): 该块中心区域在整图中的左上坐标（不含halo）
        - (t0, l0, t1, l1): 该块在整图中截取的实际区域（含halo）
    """
    B, C, H, W = x.shape
    tiles: List[torch.Tensor] = []
    coords: List[Tuple[int, int, int, int, int, int]] = []

    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            t0 = max(0, top - halo)
            l0 = max(0, left - halo)
            t1 = min(H, top + tile_h + halo)
            l1 = min(W, left + tile_w + halo)
            tiles.append(x[..., t0:t1, l0:l1])
            coords.append((top, left, t0, l0, t1, l1))
    return tiles, coords, (H, W)


def _accumulate_patch(dst: torch.Tensor, wmap: torch.Tensor,
                      patch: torch.Tensor, top: int, left: int):
    """把 patch 加权累加到 dst 指定区域，同时累加权重 wmap（用于最后做平均）"""
    B, C, h, w = patch.shape
    dst[..., top:top + h, left:left + w] += patch
    wmap[..., top:top + h, left:left + w] += 1.0


def merge_tiles_center(pred_tiles: List[torch.Tensor],
                       coords: List[Tuple[int, int, int, int, int, int]],
                       full_hw: Tuple[int, int],
                       tile_h: int, tile_w: int, halo: int = 16) -> torch.Tensor:
    """
    将每个分块预测的中心区域（去掉halo）合并回整图，重叠处做平均。
    pred_tiles 的顺序需与 coords 对齐；pred 的形状统一 (B,C,h_i,w_i)。
    """
    assert len(pred_tiles) == len(coords)
    B, C = pred_tiles[0].shape[:2]
    H, W = full_hw
    device = pred_tiles[0].device
    dtype = pred_tiles[0].dtype

    out = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    wmap = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

    for pred, (top, left, t0, l0, t1, l1) in zip(pred_tiles, coords):
        # 取中心有效区（去掉halo）
        c_top = max(0, top - t0)
        c_left = max(0, left - l0)
        c_bot = c_top + min(tile_h, pred.shape[-2] - c_top)
        c_rgt = c_left + min(tile_w, pred.shape[-1] - c_left)
        core = pred[..., c_top:c_bot, c_left:c_rgt]  # (B,C,hc,wc)

        # 放回整图
        _accumulate_patch(out, wmap, core, top, left)

    out = out / wmap.clamp_min(1.0)
    return out


# -----------------------------------------
# 3) 适配器：全局PE + 分块推理（训练也可用）
# -----------------------------------------
ModelOut = Union[torch.Tensor, Dict[str, torch.Tensor]]

class TileSegWrapper(nn.Module):
    """
    将任意分割模型包装为：
      - 在切块前对整图加一次全局二维位置编码（可关）
      - 分块（带 halo）推理/训练
      - 自动将各块输出无缝合并
    兼容：
      - 输出 Tensor: (B, C, H, W)
      - 输出 dict: {"logits": Tensor, "edge_logits": Tensor(可选)}

    参数：
      base_model:       你的分割模型（如 PVMNet）
      input_channels:   输入图像通道数（用于构造 PE）
      tile_size:        (tile_h, tile_w)
      halo:             重叠宽度
      add_global_pe:    是否添加全局位置编码（切块前）
      pe_temperature:   位置编码温度
      pe_learnable:     alpha 是否可学习
      use_autocast:     是否在 forward 中自动套 AMP（也可以外部用 autocast 管理）
      autocast_dtype:   AMP 精度（float16/bfloat16）
      align_corners:    双线性插值对齐角（默认 False，显存更省、数值更稳）
    """
    def __init__(self,
                 base_model: nn.Module,
                 input_channels: int = 3,
                 tile_size: Tuple[int, int] = (256, 256),
                 halo: int = 16,
                 add_global_pe: bool = True,
                 pe_temperature: float = 10000.0,
                 pe_learnable: bool = True,
                 use_autocast: bool = False,
                 autocast_dtype: torch.dtype = torch.float16,
                 align_corners: bool = False):
        super().__init__()
        self.base_model = base_model
        self.tile_h, self.tile_w = tile_size
        self.halo = halo
        self.add_global_pe = add_global_pe
        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype
        self.align_corners = align_corners

        self.posenc = SinCos2DPositionalEncoding(
            channels=input_channels,
            temperature=pe_temperature,
            learnable_scale=pe_learnable
        ) if add_global_pe else None

    def _forward_one(self, x: torch.Tensor) -> ModelOut:
        """对整幅输入（已加PE/或不加）执行分块推理并合并。"""
        tiles, coords, full_hw = tile_with_halo(x, self.tile_h, self.tile_w, self.halo)

        # 收集每块输出（兼容 dict / tensor）
        logits_tiles: List[torch.Tensor] = []
        edge_tiles: Optional[List[torch.Tensor]] = None

        amp_ctx = torch.cuda.amp.autocast(dtype=self.autocast_dtype) if self.use_autocast else nullcontext()

        with amp_ctx:
            for xt in tiles:
                y: ModelOut = self.base_model(xt)
                if isinstance(y, dict):
                    # 主分支
                    logits_tiles.append(y["logits"])
                    # 边界分支（可选）
                    if "edge_logits" in y:
                        if edge_tiles is None:
                            edge_tiles = []
                        edge_tiles.append(y["edge_logits"])
                else:
                    logits_tiles.append(y)

        # 合并主分支
        pred_full = merge_tiles_center(logits_tiles, coords, full_hw, self.tile_h, self.tile_w, self.halo)

        if edge_tiles is not None:
            edge_full = merge_tiles_center(edge_tiles, coords, full_hw, self.tile_h, self.tile_w, self.halo)
            return {"logits": pred_full, "edge_logits": edge_full}
        else:
            return pred_full

    def forward(self, x: torch.Tensor) -> ModelOut:
        """
        x: (B,C,H,W)；支持 B>1（逐张切块并回填到对应位置）
        """
        B = x.shape[0]
        outs: List[ModelOut] = []

        for i in range(B):
            xi = x[i:i+1]  # 保持 batch 维，便于模型中 BN/GN 的一致逻辑
            if self.posenc is not None:
                xi = self.posenc(xi)
            yi = self._forward_one(xi)
            outs.append(yi)

        # 合并 batch 的输出
        if isinstance(outs[0], dict):
            logits = torch.cat([o["logits"] for o in outs], dim=0)
            if "edge_logits" in outs[0]:
                edge_logits = torch.cat([o["edge_logits"] for o in outs], dim=0)
                return {"logits": logits, "edge_logits": edge_logits}
            else:
                return {"logits": logits}
        else:
            return torch.cat(outs, dim=0)


# --------------------
# 使用示例（参考）
# --------------------
if __name__ == "__main__":
    # 假设你已经有 PVMNet 定义（from PVMNet import PVMNet）
    try:
        from models.PVMNet import PVMNet
    except Exception:
        PVMNet = None

    # 1) 构建你的基础模型
    if PVMNet is not None:
        base = PVMNet(num_classes=2, input_channels=3)  # 兼容返回 Tensor 或 dict
    else:
        # 一个占位的玩具模型（仅供本文件自测）
        class Tiny(nn.Module):
            def __init__(self, c_in=3, c_out=2):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(c_in, 32, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, c_out, 1)
                )
            def forward(self, x):
                return self.net(x)
        base = Tiny()

    # 2) 包装：全局PE + 分块
    # model = TileSegWrapper(
    #     base_model=base,
    #     input_channels=3,            # 你的输入通道（RGB=3）
    #     tile_size=(256, 256),        # 分块大小
    #     halo=16,                     # 重叠宽度
    #     add_global_pe=True,          # 切块前加一次全局PE
    #     pe_temperature=10000.0,
    #     pe_learnable=True,
    #     use_autocast=True,          # 如果你的训练脚本外层已用 autocast，这里关掉
    #     autocast_dtype=torch.float16,
    #     align_corners=False
    # ).cuda()
    model = TileSegWrapper(
        base_model=base,
        input_channels=3,
        tile_size=(256, 256),  # 依你的显存调大/调小
        halo=16,
        add_global_pe=True,
        use_autocast=True,  # 若你没有在训练脚本外层包 autocast，则可在这里开启
        autocast_dtype=torch.float16
    ).cuda()

    # 3) 随机跑一下
    x = torch.randn(2, 3, 256, 256).cuda()  # B=2, 支持非整除 tile 的边界
    y = model(x)
    if isinstance(y, dict):
        print("logits:", y["logits"].shape, "edge:", y.get("edge_logits", None) and y["edge_logits"].shape)
    else:
        print("logits:", y.shape)
