import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
import math
from mamba_ssm import Mamba
from pytorch_wavelets import DWTForward


# ---------------- helpers ----------------
def _best_gn_groups(ch, max_groups=32):
    g = min(max_groups, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return g

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, groups=1, act_layer=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.gn   = nn.GroupNorm(_best_gn_groups(out_ch), out_ch)
        self.act  = act_layer(inplace=True)
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


# ---------------- wavelet downsample ----------------
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_gn_act = ConvGNAct(in_ch * 4, out_ch, k=1, s=1, p=0)

    def forward(self, x):
        yL, yH = self.wt(x)                  # yL: [B, C, H/2, W/2]; yH[0]: [B, C, 3, H/2, W/2]
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_gn_act(x)
        return x


# ---------------- PVM / token mixer ----------------
class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2,
                 bottleneck_ratio=0.5, res_scale=0.5, drop_path=0.1):
        super().__init__()
        assert input_dim % 4 == 0, "input_dim must be divisible by 4."
        self.input_dim  = input_dim
        self.output_dim = output_dim

        b_dim = max(16, int(input_dim * bottleneck_ratio))
        if b_dim % 4 != 0:
            b_dim = 4 * ((b_dim + 3) // 4)

        self.pre_ln  = nn.LayerNorm(input_dim)
        self.bott_in = nn.Linear(input_dim, b_dim)

        self.mamba = Mamba(
            d_model=b_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.post_ln  = nn.LayerNorm(b_dim)
        self.bott_out = nn.Linear(b_dim, output_dim)

        self.res_scale = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # if x.dtype == torch.float16:
        #     x = x.to(torch.float32)
        B, C = x.shape[:2]
        L = x.shape[2:].numel()
        hw = x.shape[2:]

        x_flat = x.reshape(B, C, L).transpose(1, 2)

        z = self.pre_ln(x_flat)
        z = self.bott_in(z)

        z1, z2, z3, z4 = torch.chunk(z, 4, dim=-1)
        m1 = self.mamba(z1) + z1
        m2 = self.mamba(z2) + z2
        m3 = self.mamba(z3) + z3
        m4 = self.mamba(z4) + z4
        z  = torch.cat([m1, m2, m3, m4], dim=-1)

        z = self.post_ln(z)
        z = self.bott_out(z)
        z = z * self.res_scale
        z = z.transpose(1, 2).reshape(B, self.output_dim, *hw)

        if self.output_dim == C:
            return x + self.drop_path(z)
        else:
            return self.drop_path(z)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, dw_channels, out_channels, kernel_size=3, stride=1, padding=1, act_layer=nn.SiLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, kernel_size, stride, padding,
                      groups=dw_channels, bias=False),
            nn.GroupNorm(_best_gn_groups(dw_channels), dw_channels),
            act_layer(inplace=True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_best_gn_groups(out_channels), out_channels),
            act_layer(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., gn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=False)
        self.gn1 = nn.GroupNorm(_best_gn_groups(hidden_features), hidden_features) if gn else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=False)
        self.gn2 = nn.GroupNorm(_best_gn_groups(out_features), out_features) if gn else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)

    def forward(self, x):
        x = self.fc1(x); x = self.gn1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.gn2(x); x = self.drop(x)
        return x


class PVMFormer(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., act_layer=nn.GELU, drop=0.01, drop_path=0.1):
        super().__init__()
        self.token_mixer = PVMLayer(
            input_dim=input_dim,
            output_dim=input_dim,
            bottleneck_ratio=0.5,
            res_scale=0.5,
            drop_path=drop_path
        )
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(in_features=input_dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, gn=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        out = x + self.drop_path(self.token_mixer(x))
        out = out + self.drop_path(self.mlp(out))
        return out


class Down_Sample(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.DS_layer = DepthwiseSeparableConv(
            dw_channels=input_channel, out_channels=output_channel,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.DownSample = Down_wt(input_channel, output_channel)

    def forward(self, x):
        x = self.DS_layer(F.avg_pool2d(x, 2, 2)) + self.DownSample(x)
        return x


# ---------------- BiFPN-lite 解码器 ----------------
class FastNormalizedFusion(nn.Module):
    """BiFPN 节点的可学习归一化加权融合"""
    def __init__(self, n_inputs, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, x_list):
        # 保证权重非负
        w = torch.relu(self.w)
        ws = w / (w.sum() + self.eps)
        out = 0.0
        for xi, wi in zip(x_list, ws):
            out = out + wi * xi
        return out


class SeparableConvGNAct(nn.Module):
    def __init__(self, ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, k, s, p, groups=ch, bias=False)
        self.gn = nn.GroupNorm(_best_gn_groups(ch), ch)
        self.pw = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.gn(x)
        x = self.pw(x)
        x = self.act(x)
        return x


class BiFPNLite(nn.Module):
    """
    输入: 四层 [P2, P3, P4, P5] (分别对应 H/4, H/8, H/16, H/32)，每层通道相同。
    输出: 同分辨率的四层 [P2o, P3o, P4o, P5o]。
    """
    def __init__(self, channels, num_layers=1):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers

        # 每层一个堆叠
        self.td_fuse23 = FastNormalizedFusion(2)  # P4_up + P3
        self.td_fuse12 = FastNormalizedFusion(2)  # P3_up + P2
        self.bu_fuse23 = FastNormalizedFusion(3)  # P3 + P3_td + P2_down
        self.bu_fuse34 = FastNormalizedFusion(3)  # P4 + P4_td + P3_down
        self.bu_fuse45 = FastNormalizedFusion(2)  # P5 + P4_down

        # 对每个节点做轻量分离卷积
        self.sep_p5 = SeparableConvGNAct(channels)
        self.sep_p4 = SeparableConvGNAct(channels)
        self.sep_p3 = SeparableConvGNAct(channels)
        self.sep_p2 = SeparableConvGNAct(channels)

        # 顶-下/下-顶后再来一遍轻量卷积稳定
        self.refine_p5 = SeparableConvGNAct(channels)
        self.refine_p4 = SeparableConvGNAct(channels)
        self.refine_p3 = SeparableConvGNAct(channels)
        self.refine_p2 = SeparableConvGNAct(channels)

    def forward(self, p2, p3, p4, p5):
        # top-down
        p5_td = self.sep_p5(p5)
        p4_td = self.td_fuse23([F.interpolate(p5_td, scale_factor=2, mode='bilinear', align_corners=False), p4])
        p4_td = self.sep_p4(p4_td)
        p3_td = self.td_fuse23([F.interpolate(p4_td, scale_factor=2, mode='bilinear', align_corners=False), p3])
        p3_td = self.sep_p3(p3_td)
        p2_td = self.td_fuse12([F.interpolate(p3_td, scale_factor=2, mode='bilinear', align_corners=False), p2])
        p2_td = self.sep_p2(p2_td)

        # bottom-up
        p3_out = self.bu_fuse23([p3, p3_td, F.avg_pool2d(p2_td, 2, 2)])
        p3_out = self.refine_p3(p3_out)
        p4_out = self.bu_fuse34([p4, F.avg_pool2d(p3_td, 2, 2), F.avg_pool2d(p3_out, 2, 2)])
        p4_out = self.refine_p4(p4_out)
        p5_out = self.bu_fuse45([p5, F.avg_pool2d(p4_out, 2, 2)])
        p5_out = self.refine_p5(p5_out)

        # 最高分辨率走 refine
        p2_out = self.refine_p2(p2_td)

        return p2_out, p3_out, p4_out, p5_out


# ---------------- EdgeHead（Sobel/Laplacian） ----------------
class FixedK3(nn.Module):
    """固定 3x3 卷积核（不训练）"""
    def __init__(self, kernel):
        super().__init__()
        k = torch.as_tensor(kernel, dtype=torch.float32).view(1, 1, 3, 3)
        self.weight = nn.Parameter(k, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)
    def forward(self, x):
        # 对每通道分别做（depthwise）
        B, C, H, W = x.shape
        w = self.weight.repeat(C, 1, 1, 1)
        x = self.pad(x)
        out = F.conv2d(x, w, groups=C)
        return out


class EdgeHead(nn.Module):
    """
    输入: 高分辨率特征 (B, C, H, W)
    输出: 边界 logits (B, 1, H, W)
    """
    def __init__(self, in_ch, mid_ch=64):
        super().__init__()
        # Sobel X / Y & Laplacian
        sobel_x = [[-1,0,1], [-2,0,2], [-1,0,1]]
        sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
        lap     = [[0,1,0],[1,-4,1],[0,1,0]]
        self.kx = FixedK3(sobel_x)
        self.ky = FixedK3(sobel_y)
        self.kl = FixedK3(lap)

        # 把三路边缘提示压回到特征空间并输出 logits
        self.reduce = nn.Sequential(
            ConvGNAct(in_ch * 3, mid_ch, k=1, s=1, p=0),
            DepthwiseSeparableConv(mid_ch, mid_ch),
        )
        self.out = nn.Conv2d(mid_ch, 1, 1, 1, 0)

    def forward(self, feat):
        ex = self.kx(feat)
        ey = self.ky(feat)
        el = self.kl(feat)
        edge_feat = torch.cat([ex, ey, el], dim=1)
        edge_feat = self.reduce(edge_feat)
        edge_logit = self.out(edge_feat)
        return edge_logit


# ---------------- 主网络 ----------------
class PVMNet_plus(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, c_list=[32,48,64,80,96,128],
                 drop_path=0.1, return_edge=True, decoder_channels=None):
        super().__init__()
        # 统一通道（与原版保持一致）
        c_list = [64, 96, 128, 160, 192, 256]
        dec_ch = decoder_channels or c_list[1]  # BiFPN 通道

        # stem & shallow
        self.stem = ConvGNAct(input_channels, c_list[0], k=3, s=1, p=1)
        self.restruct = ConvGNAct(input_channels, c_list[1], k=3, s=1, p=1)

        self.encoder0 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_best_gn_groups(c_list[0]), c_list[0]),
            nn.SiLU(inplace=True),
        )
        self.encoder1 = nn.Sequential(
            DepthwiseSeparableConv(c_list[0], c_list[1], 3, stride=1, padding=1),
            DepthwiseSeparableConv(c_list[1], c_list[1], 2, 2, 0),
        )
        self.encoder2 = nn.Sequential(PVMFormer(c_list[1], drop_path=drop_path),
                                      Down_Sample(c_list[1], c_list[2]))
        self.encoder3 = nn.Sequential(PVMFormer(c_list[2], drop_path=drop_path),
                                      Down_Sample(c_list[2], c_list[3]))
        self.encoder4 = nn.Sequential(PVMFormer(c_list[3], drop_path=drop_path),
                                      Down_Sample(c_list[3], c_list[4]))
        self.encoder5 = nn.Sequential(PVMFormer(c_list[4], drop_path=drop_path),
                                      Down_Sample(c_list[4], c_list[5]))

        # lateral 统一通道到 dec_ch（供 BiFPN 使用）
        self.lat_p2 = nn.Conv2d(c_list[2], dec_ch, 1, 1, 0, bias=False)  # en2 (H/4)
        self.lat_p3 = nn.Conv2d(c_list[3], dec_ch, 1, 1, 0, bias=False)  # en3 (H/8)
        self.lat_p4 = nn.Conv2d(c_list[4], dec_ch, 1, 1, 0, bias=False)  # en4 (H/16)
        self.lat_p5 = nn.Conv2d(c_list[5], dec_ch, 1, 1, 0, bias=False)  # en5 (H/32)

        self.bifpn = BiFPNLite(dec_ch)

        # 与 en1 (H/2) 融合的头
        self.en1_proj = nn.Conv2d(c_list[1], dec_ch, 1, 1, 0, bias=False)
        self.fuse_h2  = DepthwiseSeparableConv(dec_ch + dec_ch, dec_ch)

        # 最高分辨率分支 (H/2 -> H)
        self.h1_head = DepthwiseSeparableConv(dec_ch + c_list[1], c_list[1])

        # EdgeHead：接在 H/2 融合后的高分辨率特征上（再上采到 H）
        self.edge_head = EdgeHead(dec_ch)

        # 最终输出
        self.final = nn.Sequential(
            DepthwiseSeparableConv(2 * c_list[1], c_list[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_list[1], num_classes, 1, 1, 0)
        )

        self.return_edge = return_edge
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # encoder
        en0 = self.encoder0(x)     # H,   W
        en1 = self.encoder1(en0)   # H/2, W/2
        en2 = self.encoder2(en1)   # H/4
        en3 = self.encoder3(en2)   # H/8
        en4 = self.encoder4(en3)   # H/16
        en5 = self.encoder5(en4)   # H/32

        # 准备 P2..P5
        p2 = self.lat_p2(en2)
        p3 = self.lat_p3(en3)
        p4 = self.lat_p4(en4)
        p5 = self.lat_p5(en5)

        # BiFPN 融合
        p2o, p3o, p4o, p5o = self.bifpn(p2, p3, p4, p5)   # P2 最高分辨率 (H/4)

        # 将 P2o 上采到 H/2，与 en1 融合（轻量头）
        h2 = F.interpolate(p2o, scale_factor=2, mode='bilinear', align_corners=False)  # H/2
        en1p = self.en1_proj(en1)
        h2 = torch.cat([h2, en1p], dim=1)
        h2 = self.fuse_h2(h2)  # (B, dec_ch, H/2, W/2)

        # 边界分支（在 H/2 做 edge logits，再上采到 H 供监督或可视化）
        edge_logits_h2 = self.edge_head(h2)                   # (B,1,H/2,W/2)
        edge_logits = F.interpolate(edge_logits_h2, scale_factor=2, mode='bilinear', align_corners=False)  # H

        # 主分支上采到 H，与 restruct(x) 融合，再输出类别 logits
        h1 = torch.cat([F.interpolate(h2, scale_factor=2, mode='bilinear', align_corners=False),  # H
                        self.restruct(x)], dim=1)
        h1 = self.h1_head(h1)  # (B, c_list[1], H, W)

        out = torch.cat([h1, self.restruct(x)], dim=1)  # [B, 2*c1, H, W]
        out = self.final(out).contiguous()              # segmentation logits

        if self.return_edge:
            return out, edge_logits
        else:
            return out

import torch
from ptflops import get_model_complexity_info

if __name__ == "__main__":

    model = PVMNet_plus().cuda()
    image_size = 256
    input_size = (3, image_size, image_size)
    detail = True
    # 计算 FLOPs 和 Params
    flops, params = get_model_complexity_info(
        model, input_size,
        as_strings=True,
        print_per_layer_stat=detail,  # ✅ 打印每一层
        verbose=detail
    )
    print("=" * 60)
    print(f"Input size: {input_size}")
    print(f"Total FLOPs: {flops}")
    print(f"Total Params: {params}")
    print("=" * 60)

    # 前向推理，检查输出
    x = torch.randn(1, *input_size).cuda()
    with torch.no_grad():
        out = model(x)

    if isinstance(out, (tuple, list)):
        print("Output contains multiple tensors:")
        for i, o in enumerate(out):
            print(f" - Output[{i}] shape: {o.shape}")
    elif isinstance(out, dict):
        print("Output is a dict:")
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f" - {k}: {v.shape}")
            else:
                print(f" - {k}: {type(v)}")
    else:
        print("Output shape:", out.shape)
