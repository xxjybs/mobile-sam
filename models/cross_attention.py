import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- 小工具：深度可分离卷积 -----------------
class DWConvBlock(nn.Module):
    def __init__(self, c, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, s, p, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

# ----------------- 多方向边缘：Sobel/Scharr -----------------
class MultiDirEdge(nn.Module):
    """
    生成多方向边缘响应：{0°, 90°, 45°, 135°}，支持 sobel 或 scharr
    对每个输入通道独立卷积（groups=C），输出 edge_ch 通道（默认=输入C或指定投影）
    """
    def __init__(self, in_ch=3, out_ch=64, mode='sobel'):
        super().__init__()
        assert mode in ['sobel', 'scharr']
        self.mode = mode
        # 基础核（Sobel / Scharr）
        if mode == 'sobel':
            kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        else:  # scharr（对噪声更鲁棒）
            kx = torch.tensor([[3,0,-3],[10,0,-10],[3,0,-3]], dtype=torch.float32)
        ky = kx.t().contiguous()
        # 45° 与 135° 方向核（用旋转近似）
        k45 = torch.tensor([[0,1,2],[-1,0,1],[-2,-1,0]], dtype=torch.float32)
        k135= torch.tensor([[2,1,0],[1,0,-1],[0,-1,-2]], dtype=torch.float32)

        self.register_buffer('kx',   kx.view(1,1,3,3))
        self.register_buffer('ky',   ky.view(1,1,3,3))
        self.register_buffer('k45', k45.view(1,1,3,3))
        self.register_buffer('k135',k135.view(1,1,3,3))

        # 方向间融合的小 head（把4*C -> out_ch）
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch*4, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DWConvBlock(out_ch)
        )

    def forward(self, x):  # (B,C,H,W)
        B,C,H,W = x.shape
        device = x.device
        kx   = self.kx.to(device).expand(C,1,3,3)
        ky   = self.ky.to(device).expand(C,1,3,3)
        k45  = self.k45.to(device).expand(C,1,3,3)
        k135 = self.k135.to(device).expand(C,1,3,3)

        gx   = F.conv2d(x, kx, padding=1, groups=C)
        gy   = F.conv2d(x, ky, padding=1, groups=C)
        g45  = F.conv2d(x, k45, padding=1, groups=C)
        g135 = F.conv2d(x, k135, padding=1, groups=C)

        # 方向响应取绝对值（也可用平方和开方，但 abs 更省算）
        stack = torch.cat([gx.abs(), gy.abs(), g45.abs(), g135.abs()], dim=1)  # (B,4C,H,W)
        return self.fuse(stack)  # (B,out_ch,H,W)

# ----------------- 多尺度金字塔（SPP/PPM风格） -----------------
class MultiScalePyramid(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, bins=(1,2,4,8)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.bins = bins
        self.projs = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False) for _ in bins])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch*(len(bins)+1), out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DWConvBlock(out_ch)
        )
    def forward(self, x):  # (B,3,H,W) 或 (B,C,H,W)
        B,_,H,W = x.shape
        f0 = self.stem(x)  # (B,out,H,W)
        feats = [f0]
        for b, proj in zip(self.bins, self.projs):
            p = F.adaptive_avg_pool2d(f0, (b,b))
            p = proj(p)
            p = F.interpolate(p, (H,W), mode='bilinear', align_corners=False)
            feats.append(p)
        return self.fuse(torch.cat(feats, dim=1))  # (B,out,H,W)

# ----------------- 纯卷积 Prompt Adapter（极低显存） -----------------
class ConvPromptAdapter(nn.Module):
    """
    三路（多方向边缘 + 多尺度金字塔 + 可选手工特征） -> 1x1 融合 -> DWConv 提炼 -> Prompt
    mode='add'  输出 prompt (B,Cq,H,W)，用于 x += prompt
    mode='film' 输出 gamma,beta (B,Cq,H,W)，用于 x = x*(1+tanh(gamma))+beta
    """
    def __init__(self, c_img=3, c_k=12, c_q=3, use_handcrafted=True, bins=(1,2,4,8), edge_mode='scharr', mode='film', bottleneck=8):
        super().__init__()
        self.use_handcrafted = use_handcrafted
        self.mode = mode

        self.edge = MultiDirEdge(in_ch=c_img, out_ch=c_k, mode=edge_mode)
        self.msp  = MultiScalePyramid(in_ch=c_img, out_ch=c_k, bins=bins)
        if use_handcrafted:
            self.hand_proj = nn.Sequential(
                nn.Conv2d(c_img, c_k, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_k),
                nn.ReLU(inplace=True)
            )

        fuse_in = 2*c_k if not use_handcrafted else 3*c_k
        self.fuse3 = nn.Sequential(
            nn.Conv2d(fuse_in, c_k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_k),
            nn.ReLU(inplace=True),
            DWConvBlock(c_k)  # 3x3 DWConv 提炼
        )

        # 映射到目标通道（与主干该 stage 的 Cq 对齐）
        self.to_cq = nn.Conv2d(c_k, c_q, 1, 1, 0, bias=False)

        # 两层 MLP 等效：用 1x1 conv 实现（逐像素）
        mid = max(4, c_q // bottleneck)
        self.tune = nn.Sequential(nn.Conv2d(c_q, mid, 1, 1, 0, bias=False), nn.GELU())
        if mode == 'film':
            self.up_gamma = nn.Conv2d(mid, c_q, 1, 1, 0, bias=False)
            self.up_beta  = nn.Conv2d(mid, c_q, 1, 1, 0, bias=False)
        else:
            self.up = nn.Conv2d(mid, c_q, 1, 1, 0, bias=False)

        # 可学习的三路门控（稳定训练，允许网络自己调配三路强度）
        self.gates = nn.Parameter(torch.ones(3) if use_handcrafted else torch.ones(2))

    def forward(self, img_bchw, handcrafted_bchw=None):
        """
        img_bchw:         (B,3,H,W) 原图或其增强图
        handcrafted_bchw: (B,3,H,W) 你的手工特征（FFT/高斯/SRM），与 img 同分辨率
        """
        B,_,H,W = img_bchw.shape
        f_edge = self.edge(img_bchw)   # (B,c_k,H,W)
        f_msp  = self.msp(img_bchw)    # (B,c_k,H,W)
        feats  = [self.gates[0]*f_edge, self.gates[1]*f_msp]

        if self.use_handcrafted:
            assert handcrafted_bchw is not None, "use_handcrafted=True 需要提供 handcrafted_bchw"
            f_hand = self.hand_proj(handcrafted_bchw)  # (B,c_k,H,W)
            feats.append(self.gates[2]*f_hand)

        ctx = self.fuse3(torch.cat(feats, dim=1))  # (B,c_k,H,W)
        ctx = self.to_cq(ctx)                      # (B,c_q,H,W)

        z = self.tune(ctx)                         # (B,mid,H,W)
        if self.mode == 'film':
            gamma = self.up_gamma(z)
            beta  = self.up_beta(z)
            return gamma, beta                    # (B,Cq,H,W), (B,Cq,H,W)
        else:
            prompt = self.up(z)
            return prompt                         # (B,Cq,H,W)


import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgePromptBranch(nn.Module):
    """
    输入:   (B, 3, H, W) 原图(或增强图)
    输出:   (B, prompt_embed_dim, H_e, W_e) 的 dense prompt
    设计:   Sobel 边缘 -> 分组卷积细化 -> 1x1 投影到 prompt_embed_dim
    """
    def __init__(self, in_ch=3, mid_ch=64, prompt_embed_dim=256):
        super().__init__()
        # Sobel 核（固定）
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        ky = kx.t().contiguous()
        self.register_buffer('kx', kx.view(1,1,3,3))
        self.register_buffer('ky', ky.view(1,1,3,3))

        # 轻量提炼：DWConv + PWConv
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, mid_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        # 投影到 prompt 维度
        self.to_prompt_dim = nn.Conv2d(mid_ch, prompt_embed_dim, 1, 1, 0, bias=False)

        # 学习型门控，用于与 no_mask_embed 融合
        self.gate = nn.Parameter(torch.tensor(0.5))  # 初值0.5，可调

    def forward(self, img_bchw, target_hw: int):
        """
        img_bchw:  (B,3,H,W)
        target_hw: self.image_embedding_size (例如 64)
        """
        B, C, H, W = img_bchw.shape
        # 调整到与 image_embedding 对齐的分辨率
        x = F.interpolate(img_bchw, size=(target_hw, target_hw),
                          mode='bilinear', align_corners=False)

        # Sobel 梯度（逐通道 groups 卷积）
        C = x.shape[1]
        kx = self.kx.to(x.device).expand(C,1,3,3)
        ky = self.ky.to(x.device).expand(C,1,3,3)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        mag = torch.sqrt((gx**2 + gy**2).clamp_min(1e-12))  # (B,C,H_e,W_e)

        # 轻量提炼 + 投影
        f = self.refine(mag)                  # (B,mid_ch,H_e,W_e)
        dense_edge_prompt = self.to_prompt_dim(f)  # (B,prompt_embed_dim,H_e,W_e)
        return dense_edge_prompt

