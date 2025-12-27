import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam_adapter.modeling.models import register
# from models.sam_adapter.modeling.mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer
from models.MobileSAMv2.mobilesamv2.build_sam import *
from models.EdgeNAT_layer.fusion import FFM_SCAtten
from models.LiteEdgeExtractor import LiteEdgeExtractor64

logger = logging.getLogger(__name__)
from models.sam_adapter.modeling.iou_loss import IOU
from typing import Any, Optional, Tuple


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W
# 定义一个基本残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 如果stride>1或通道数变化，需要shortcut投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out



class Mobile_sam_adapter(nn.Module):
    def __init__(self, inp_size=1024):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = 1024
        self.image_encoder = build_sam_vit_t_encoder()
        self.prompt_embed_dim = 256
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.dense_prompt = nn.Sequential(
            BasicBlock(3, 32, stride=2),  # [b, 32, 512, 512]
            BasicBlock(32, 64, stride=2),  # [b, 64, 256, 256]
            BasicBlock(64, 128, stride=2),  # [b, 128, 128, 128]
            BasicBlock(128, 256, stride=2),  # [b, 256, 64, 64]
        )
        self.pe_layer = PositionEmbeddingRandom(256 // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // 16
        self.no_mask_embed = nn.Embedding(1, 256)

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def forward(self, x):
        self.input = x
        bs = x.size(0)
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.dense_prompt(self.input)
        self.features = self.image_encoder(self.input)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks
        return masks
        # if self.training:
        #     edge = self.edge_prompt(dense_embeddings)
        #     return masks, edge
        # else:
        #     return masks

    def infer(self, input):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(input)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        # masks = torch.sigmoid(masks)
        return masks, self.features

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

class DenseToSparseEmbeddings(nn.Module):
    """
    把稠密特征图 F:[B,C_in,H,W] 映射为稀疏 embeddings:[B, N, out_dim]
    - N = bins[0] * bins[1] （默认 4x4=16）
    - 可选权重图 weight:[B,1,H,W]（如边界概率），做加权自适应池化
    - 投影到 out_dim（默认 256），可加 LayerNorm、L2 normalize、可学习位置编码
    """
    def __init__(self,
                 c_in: int,
                 out_dim: int = 256,
                 bins: tuple = (4, 4),
                 use_pos_embed: bool = True,
                 use_layernorm: bool = True,
                 l2_normalize: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.out_dim = out_dim
        self.bins = bins
        self.eps = eps
        self.proj = nn.Conv2d(c_in, out_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
        self.l2_normalize = l2_normalize

        Hbin, Wbin = bins
        self.num_tokens = Hbin * Wbin
        if use_pos_embed:
            # 每个网格一个可学习位置向量（[1,N,256]）
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, out_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    @torch.no_grad()
    def _fallback_avg(self, Fp, Hbin, Wbin):
        # 无权均值，防止权重极小导致数值不稳
        return F.adaptive_avg_pool2d(Fp, output_size=(Hbin, Wbin))

    def forward(self, F_dense: torch.Tensor, weight: torch.Tensor | None = None):
        """
        F_dense: [B, C_in, H, W]  稠密特征图
        weight : [B, 1,    H, W]  可选权重图（如边界概率 0~1）
        return : [B, N, out_dim]  （N = bins[0]*bins[1]）
        """
        B, _, H, W = F_dense.shape
        Hbin, Wbin = self.bins

        # 1) 通道投影到 out_dim
        Fp = self.proj(F_dense)  # [B, out_dim, H, W]

        # 2)（加权）自适应池化到 Hbin x Wbin
        if weight is not None:
            # 分子/分母分别做自适应平均（同窗口内 “和/面积” 的等价）
            num = F.adaptive_avg_pool2d(Fp * weight, output_size=(Hbin, Wbin))  # [B,out_dim,Hbin,Wbin]
            den = F.adaptive_avg_pool2d(weight,     output_size=(Hbin, Wbin))   # [B,1,     Hbin,Wbin]
            pooled = num / (den + self.eps)
            # 对 den≈0 的格子回退到无权均值
            empty = (den <= self.eps).float()
            if empty.any():
                avg_unweighted = self._fallback_avg(Fp, Hbin, Wbin)
                pooled = pooled * (1.0 - empty) + avg_unweighted * empty
        else:
            pooled = self._fallback_avg(Fp, Hbin, Wbin)  # [B,out_dim,Hbin,Wbin]

        # 3) 展平为 token 序列 [B, N, out_dim]
        tokens = pooled.permute(0, 2, 3, 1).reshape(B, Hbin * Wbin, self.out_dim)  # [B,N,D]

        # 4) 规范化 & 位置编码
        tokens = self.norm(tokens)  # LayerNorm 按最后一维
        if self.l2_normalize:
            tokens = F.normalize(tokens, p=2, dim=-1)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed  # [1,N,D] 会自动广播到 batch

        return tokens


if __name__ == "__main__":
    B = 1
    F_dense = torch.randn(B, 256, 64, 64)  # 你的稠密特征图（比如 ViT 的语义图或融合后的边界语义图）


    pooler = DenseToSparseEmbeddings(
        c_in=256, out_dim=256, bins=(4, 4),
        use_pos_embed=True, use_layernorm=True, l2_normalize=False
    )

    sparse_embeddings = pooler(F_dense, weight=None)  # 形状: [1,16,256]
    print(sparse_embeddings.shape)
