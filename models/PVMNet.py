import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba
from pytorch_wavelets import DWTForward


class Down_Sample(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        super(Down_Sample, self).__init__()
        self.DS_layer = DepthwiseSeparableConv(dw_channels=input_channel, out_channels=output_channel,
                                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(input_channel * 4, output_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

    def forward(self, x):
        out = self.DS_layer(F.avg_pool2d(x, 2, 2))
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x+out


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, dw_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, kernel_size, stride, padding, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PVMFormer(nn.Module):

    def __init__(self, input_dim, mlp_ratio=4.,
                 act_layer=nn.GELU, drop=0.01):
        super().__init__()

        self.token_mixer = PVMLayer(input_dim=input_dim, output_dim=input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(in_features=input_dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        out = x + self.token_mixer(x)
        out = out + self.mlp(out) + x
        return out


class PVMNet(nn.Module):

    def __init__(self, num_classes=2, input_channels=3, c_list=[32, 48, 64, 80, 96, 128]):
        super().__init__()
        c_list = [64, 96, 128, 160, 192, 256]
        self.restruct = nn.Conv2d(input_channels, c_list[1], 3, stride=1, padding=1)

        self.encoder0 = nn.Sequential(
            # nn.GroupNorm(1, input_channels),
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder1 = nn.Sequential(
            # nn.GroupNorm(1, c_list[0]),
            DepthwiseSeparableConv(c_list[0], c_list[1], 3, stride=1, padding=1),
            # nn.GroupNorm(1, c_list[1]),
            DepthwiseSeparableConv(c_list[1], c_list[1], 2, 2, 0)
        )

        self.encoder2 = nn.Sequential(
            PVMFormer(c_list[1]),
            Down_Sample(c_list[1], c_list[2])
        )
        self.encoder3 = nn.Sequential(
            PVMFormer(c_list[2]),
            Down_Sample(c_list[2], c_list[3])
        )
        self.encoder4 = nn.Sequential(
            PVMFormer(c_list[3]),
            Down_Sample(c_list[3], c_list[4])
        )
        self.encoder5 = nn.Sequential(
            PVMFormer(c_list[4]),
            Down_Sample(c_list[4], c_list[5])
        )

        self.fusion = nn.Sequential(
            DepthwiseSeparableConv(c_list[1] + c_list[5], c_list[1], 3, 1, 1),
        )

        self.final = nn.Sequential(
            # nn.GroupNorm(1, 2 * c_list[1]),
            DepthwiseSeparableConv(2 * c_list[1], c_list[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_list[1], num_classes, 1, 1, 0)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
        en0 = self.encoder0(x)
        en1 = self.encoder1(en0)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)
        en5 = self.encoder5(en4)

        en5 = F.interpolate(en5, scale_factor=(16, 16), mode='bilinear', align_corners=True)

        out = torch.cat([en1, en5], 1)
        out = self.fusion(out)

        restruct = self.restruct(x)
        out = torch.cat([F.interpolate(out, scale_factor=(2, 2), mode='bilinear', align_corners=True), restruct],
                        dim=1)  # b, 2*c1, H/2, W/2
        out = self.final(out).contiguous()
        # out = torch.sigmoid(out).contiguous()
        return out
