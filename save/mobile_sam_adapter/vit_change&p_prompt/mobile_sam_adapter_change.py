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

import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryBasedPromptGenerator(nn.Module):
    """
    从 image_feats (B, C_in, H, W) 上通过 cross-attention 查询生成:
      - sparse_embeddings: [B, num_queries, embed_dim]
      - dense_embeddings:  [B, embed_dim, H_out, W_out]  (H_out/W_out 默认 = H/W)
    Uses nn.MultiheadAttention for cross-attention (seq-first interface).
    """
    def __init__(self,
                 in_ch=256,
                 embed_dim=256,
                 num_queries=16,
                 num_heads=8,
                 dense_grid_size=8,   # grid queries resolution (G x G)
                 dense_out_size=None, # if None -> use input H/W
                 q_init_std=0.01):
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.dense_grid_size = dense_grid_size
        self.dense_out_size = dense_out_size  # final dense spatial size

        # if encoder features channels != embed_dim, project them
        if in_ch != embed_dim:
            self.feat_proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1)
        else:
            self.feat_proj = nn.Identity()

        # optional learned 2D positional embedding (added to keys/values)
        # shape: (1, embed_dim, H_ref, W_ref) -> will be resized to match input H/W
        self.pos_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        )

        # Sparse queries: learnable content / positional embeddings
        # Stored as (num_queries, embed_dim) and used as query tokens
        self.sparse_q = nn.Parameter(torch.randn(num_queries, embed_dim) * q_init_std)

        # Dense grid queries: low-res spatial queries, shape (G*G, embed_dim)
        self.dense_grid_q = nn.Parameter(torch.randn(dense_grid_size * dense_grid_size, embed_dim) * q_init_std)

        # MultiheadAttention expects (L, B, E) inputs
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)

        # optional small MLPs to refine outputs
        self.sparse_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.dense_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, image_feats):
        """
        image_feats: [B, C_in, H, W]  (in your case C_in=256, H=W=64)
        returns:
            sparse_embeddings: [B, num_queries, embed_dim]
            dense_embeddings:  [B, embed_dim, H_out, W_out]
        """
        B, C, H, W = image_feats.shape
        device = image_feats.device
        H_out = self.dense_out_size[0] if (self.dense_out_size is not None) else H
        W_out = self.dense_out_size[1] if (self.dense_out_size is not None) else W

        # 1) project encoder features to embed_dim if needed
        feats = self.feat_proj(image_feats)  # [B, E, H, W]

        # 2) add positional bias (learned conv) -> helps attention know spatial layout
        pos = self.pos_conv(feats)           # [B, E, H, W]
        feats = feats + pos

        # 3) flatten spatial dims -> (S, B, E) for MultiheadAttention
        S = H * W
        feat_flat = feats.flatten(2).permute(2, 0, 1).contiguous()  # [S, B, E]

        # --- Sparse queries cross-attend to image feats ---
        # form queries: (L_q, B, E)
        q_sparse = self.sparse_q.unsqueeze(1).expand(-1, B, -1)  # [num_queries, B, E]
        # cross-attn: query=q_sparse, key=feat_flat, value=feat_flat
        # out: [num_queries, B, E]
        sparse_out, _ = self.cross_attn(q_sparse, feat_flat, feat_flat)
        sparse_out = sparse_out.permute(1, 0, 2).contiguous()    # -> [B, num_queries, E]
        sparse_out = self.sparse_mlp(sparse_out)                 # refine
        # final sparse embeddings shape: [B, num_queries, E]
        sparse_embeddings = sparse_out

        # --- Dense grid queries cross-attend to image feats ---
        G = self.dense_grid_size
        grid_n = G * G
        q_dense = self.dense_grid_q.unsqueeze(1).expand(-1, B, -1)  # [G*G, B, E]
        dense_out, _ = self.cross_attn(q_dense, feat_flat, feat_flat)  # [G*G, B, E]
        dense_out = dense_out.permute(1, 2, 0).contiguous()  # [B, E, G*G]
        dense_out = dense_out.view(B, self.embed_dim, G, G)  # [B, E, G, G]
        dense_out = dense_out.permute(0, 1, 2, 3).contiguous()
        # refine each grid token with MLP: apply per-location mlp via conv1x1 trick
        dense_out = dense_out.flatten(2).permute(0, 2, 1)   # [B, G*G, E]
        dense_out = self.dense_mlp(dense_out)               # [B, G*G, E]
        dense_out = dense_out.permute(0, 2, 1).contiguous().view(B, self.embed_dim, G, G)  # [B, E, G, G]

        # upsample to desired output resolution (H_out, W_out)
        dense_embeddings = F.interpolate(dense_out, size=(H_out, W_out), mode='bilinear', align_corners=False)
        # final dense_embeddings: [B, E, H_out, W_out]

        return sparse_embeddings, dense_embeddings


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
        self.prompt_from_feat = QueryBasedPromptGenerator(
            in_ch=256, embed_dim=256, num_queries=16, num_heads=8, dense_grid_size=8
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
        self.features = self.image_encoder(self.input)
        image_feats = self.features  # [B,256,64,64]
        sparse_embeddings, dense_embeddings = self.prompt_from_feat(image_feats)
        # sparse: [B,16,256], dense: [B,256,64,64]

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



# if __name__ == "__main__":
#     B = 1
#     F_dense = torch.randn(B, 256, 64, 64)  # 你的稠密特征图（比如 ViT 的语义图或融合后的边界语义图）
#
#
#     pooler = DenseToSparseEmbeddings(
#         c_in=256, out_dim=256, bins=(4, 4),
#         use_pos_embed=True, use_layernorm=True, l2_normalize=False
#     )
#
#     sparse_embeddings = pooler(F_dense, weight=None)  # 形状: [1,16,256]
#     print(sparse_embeddings.shape)
