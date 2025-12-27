import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 轻量块：深度可分离卷积 ----------
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_bn=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

# --------- 边缘固定算子（可设为可学习） ----------
class FixedKernels(nn.Module):
    def __init__(self, trainable=False, dtype=torch.float32):
        super().__init__()
        self.rgb2y = nn.Conv2d(3, 1, 1, bias=False)
        w = torch.tensor([[0.299, 0.587, 0.114]], dtype=dtype).view(1,3,1,1)
        self.rgb2y.weight = nn.Parameter(w, requires_grad=trainable)

        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=dtype)
        sobel_y = sobel_x.t()
        lap     = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=dtype)

        self.grad = nn.Conv2d(1, 2, 3, padding=1, bias=False)   # [gx, gy]
        self.lap  = nn.Conv2d(1, 1, 3, padding=1, bias=False)   # [lap]
        self.grad.weight = nn.Parameter(torch.stack([sobel_x, sobel_y], 0).unsqueeze(1),
                                        requires_grad=trainable)  # [2,1,3,3]
        self.lap.weight  = nn.Parameter(lap.view(1,1,3,3), requires_grad=trainable)

    def forward(self, x_rgb_64):
        # x_rgb_64: [B,3,64,64]
        y = self.rgb2y(x_rgb_64)             # [B,1,64,64]
        gxgy = self.grad(y)                  # [B,2,64,64]
        lap  = self.lap(y)                   # [B,1,64,64]
        mag  = torch.sqrt(gxgy[:,0:1]**2 + gxgy[:,1:2]**2 + 1e-6)
        return torch.cat([gxgy, mag, lap], dim=1)  # [B,4,64,64]

# --------- 主模块：1024x1024 -> l:[B,16,64,64] ----------
class LiteEdgeExtractor64(nn.Module):
    """
    输入:  x ∈ R^{B×3×1024×1024}
    输出:  l ∈ R^{B×16×64×64}
    结构:  四次 stride=2 下采样到 64×64 的轻量干路
          64×64 尺度的 Sobel/Laplacian 边缘分支（4通道）
          与上下文分支（12通道）融合，1×1 压到 16 通道
    """
    def __init__(self, mid_ch=(24, 32, 48, 64), out_ch=16,
                 trainable_edges=False, use_gn=False):
        super().__init__()
        Norm = (lambda c: nn.GroupNorm(8, c)) if use_gn else (lambda c: nn.BatchNorm2d(c))
        c1, c2, c3, c4 = mid_ch  # 对应 512,256,128,64 尺度的通道

        # 1024 -> 512
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False),
            Norm(c1), nn.ReLU(inplace=True),
            DWConvBlock(c1, c1, k=3, s=1, p=1)
        )
        # 512 -> 256
        self.stage2 = nn.Sequential(
            DWConvBlock(c1, c2, k=3, s=2, p=1),
            DWConvBlock(c2, c2, k=3, s=1, p=1)
        )
        # 256 -> 128
        self.stage3 = nn.Sequential(
            DWConvBlock(c2, c3, k=3, s=2, p=1),
            DWConvBlock(c3, c3, k=3, s=1, p=1)
        )
        # 128 -> 64
        self.stage4 = nn.Sequential(
            DWConvBlock(c3, c4, k=3, s=2, p=1),
            DWConvBlock(c4, c4, k=3, s=1, p=1)
        )

        # 直接把 RGB 下采样到 64×64 做边缘
        self.rgb_down_to_64 = nn.Sequential(
            nn.AvgPool2d(2,2),  # 1024->512
            nn.AvgPool2d(2,2),  # 512 ->256
            nn.AvgPool2d(2,2),  # 256 ->128
            nn.AvgPool2d(2,2),  # 128 ->64
        )
        self.fixed = FixedKernels(trainable=trainable_edges)

        # 上下文分支（在 64×64 上）
        self.ctx = nn.Sequential(
            DWConvBlock(c4, c4, k=3, s=1, p=1),
            nn.Conv2d(c4, 12, 1, bias=False),
            Norm(12), nn.ReLU(inplace=True)
        )

        # 融合 4(边缘) + 12(上下文) -> 16
        self.fuse = nn.Sequential(
            nn.Conv2d(4 + 12, out_ch, 1, bias=False),
            Norm(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        x: [B,3,1024,1024] -> l: [B,16,64,64]
        """
        x1 = self.stem(x)     # [B,c1,512,512]
        x2 = self.stage2(x1)  # [B,c2,256,256]
        x3 = self.stage3(x2)  # [B,c3,128,128]
        x4 = self.stage4(x3)  # [B,c4, 64, 64]

        rgb_64   = self.rgb_down_to_64(x)  # [B,3,64,64]
        edge_64  = self.fixed(rgb_64)      # [B,4,64,64]
        ctx_64   = self.ctx(x4)            # [B,12,64,64]

        l = self.fuse(torch.cat([edge_64, ctx_64], dim=1))  # [B,16,64,64]
        return l

# ---------- 简单自测 ----------
if __name__ == "__main__":
    B = 2
    x = torch.randn(B, 3, 1024, 1024)
    net = LiteEdgeExtractor64(out_ch=16, trainable_edges=True, use_gn=False)
    l = net(x)
    print(l.shape)  # 期望: [B,16,64,64]
