# Self-contained VJEPA 2.1 vision-transformer encoder for posetail.
#
# This file vendors the encoder half of the VJEPA 2.1 backbone so posetail no
# longer depends on the external `vjepa2` / `hub` package. Only the four
# `vjepa2_1_vit_*_384` builders are exposed, and only the *encoder* is built and
# loaded (the predictor is discarded by posetail). Weights are downloaded from
# the same public URLs used by the upstream `hub.backbones` loader.
#
# Ported from https://github.com/facebookresearch/vjepa2 (MIT / FB license):
#   - app/vjepa_2_1/models/vision_transformer.py   (VisionTransformer + arch fns)
#   - app/vjepa_2_1/models/utils/modules.py         (Block, RoPEAttention, ...)
#   - app/vjepa_2_1/models/utils/patch_embed.py     (PatchEmbed, PatchEmbed3D)
#   - src/masks/utils.py                            (apply_masks)
#   - src/utils/tensors.py                          (trunc_normal_)
#   - src/hub/backbones.py                          (builders + weight loading)
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Licensed under the MIT / accompanying license of the original source tree.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# small tensor / mask utils (from src/utils/tensors.py, src/masks/utils.py)
# ---------------------------------------------------------------------------


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample. Inlined from timm to avoid the dep."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


# ---------------------------------------------------------------------------
# patch embeddings (from app/vjepa_2_1/models/utils/patch_embed.py)
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """Image (video) to Patch Embedding"""

    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# transformer blocks / attention (from app/vjepa_2_1/models/utils/modules.py)
# ---------------------------------------------------------------------------


def rotate_queries_or_keys(x, pos, n_registers, has_cls_first):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = N - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)

    emb_sin = freq.sin()
    emb_cos = freq.cos()

    emb_sin = emb_sin.repeat_interleave(2, dim=-1)
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)

    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)

    parts = []
    if n_cls:
        parts.append(x_cls)
    parts.append(out_ctx)
    if n_registers:
        parts.append(x_reg)
    out = torch.cat(parts, dim=-2)

    return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        self.pretrained_patch_size = patch_size
        if patch_size == 14:
            self.pretrained_grid_size = int(252 / patch_size)
        elif patch_size == 16:
            self.pretrained_grid_size = int(256 / patch_size)

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False):
        B, N, C = x.size()
        N_ctx = N - self.n_registers
        grid_depth = int(N_ctx // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(
                    int(grid_depth * self.grid_size * self.grid_size), device=x.device
                )
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        if self.interpolate_rope:
            if H_patches is None:
                H_patches = int(self.grid_size)
            if W_patches is None:
                W_patches = int(self.grid_size)
            h_mask = h_mask * (self.pretrained_grid_size - 1) / (H_patches - 1)
            w_mask = w_mask * (self.pretrained_grid_size - 1) / (W_patches - 1)

        s = 0
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        s += self.w_dim

        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        else:
            return x, None


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_sdpa=True, is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_rope = use_rope
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
                n_registers=n_registers,
                has_cls_first=has_cls_first,
                interpolate_rope=interpolate_rope,
                patch_size=patch_size,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False, mode="video"):
        if self.use_rope:
            y, attn = self.attn(self.norm1(x), mask=mask, T=T, H_patches=H_patches, W_patches=W_patches, return_attn=return_attn)
        else:
            y = self.attn(self.norm1(x))
            attn = None
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        else:
            return x, None


# ---------------------------------------------------------------------------
# vision transformer (from app/vjepa_2_1/models/vision_transformer.py)
# ---------------------------------------------------------------------------


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        is_causal=False,
        use_rope=False,
        init_type: str = "default",
        handle_nonsquare_inputs=True,
        img_temporal_dim_size=None,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        modality_embedding=True,
        n_output_distillation=4,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.init_type = init_type
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size

        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size, tubelet_size=tubelet_size, in_chans=in_chans, embed_dim=embed_dim
            )
            self.num_patches = (
                (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        if self.img_temporal_dim_size is not None:
            if not isinstance(self.img_temporal_dim_size, int):
                raise ValueError(f"img_temporal_dim_size must be an int, got {self.img_temporal_dim_size}")
            self.patch_embed_img = PatchEmbed3D(patch_size=patch_size, tubelet_size=1, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed_img = None

        self.uniform_power = uniform_power

        self.use_rope = use_rope
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=img_size[0] // patch_size,
                    grid_depth=num_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    n_registers=n_registers,
                    has_cls_first=has_cls_first,
                    interpolate_rope=interpolate_rope,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )

        self.attn_out = False
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        if depth == 12:
            self.hierarchical_layers = [2, 5, 8, 11]
            if n_output_distillation == 4:
                self.out_layers_distillation = [2, 5, 8, 11]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [11]
        elif depth == 24:
            self.hierarchical_layers = [5, 11, 17, 23]
            if n_output_distillation == 4:
                self.out_layers_distillation = [5, 11, 17, 23]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [23]
        elif depth == 40:
            self.hierarchical_layers = [9, 19, 29, 39]
            if n_output_distillation == 4:
                self.out_layers_distillation = [9, 19, 29, 39]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [39]
        elif depth == 48:
            self.hierarchical_layers = [11, 23, 37, 47]
            if n_output_distillation == 4:
                self.out_layers_distillation = [11, 23, 37, 47]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [47]
        else:
            print("Check the code! ;)")
        self.norms_block = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))])

        self.cls_token = None
        self.return_hierarchical = False

        self.modality_embedding = False
        if modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            return
        if self.init_type == "default":
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=self.init_std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif self.init_type == "xavier_uniform":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif self.init_type == "xavier_normal":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            raise ValueError(f"Unknown init type {self.init_type}")

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def check_temporal_dim(self, shape) -> bool:
        if self.img_temporal_dim_size is not None:
            if shape[2] == self.img_temporal_dim_size:
                return True
        return False

    def forward(self, x, masks=None, training=False):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        elif x.ndim == 5:
            _, _, T, H, W = x.shape
            if self.check_temporal_dim(x.shape):
                T = T // 1
            else:
                T = T // self.tubelet_size

        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        if not self.use_rope:
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)

        if self.check_temporal_dim(x.shape):
            assert self.patch_embed_img is not None
            x = self.patch_embed_img(x)
            mode = "img"
            if self.modality_embedding:
                x += self.img_mod_embed.repeat(x.shape[0], 1, 1)
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x += self.video_mod_embed.repeat(x.shape[0], 1, 1)

        if not self.use_rope:
            x += pos_embed

        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        outs = []
        hier = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x, attn = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    use_reentrant=False,
                    return_attn=self.attn_out,
                    mode=mode,
                )
            else:
                x, attn = blk(
                    x,
                    mask=masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    return_attn=self.attn_out,
                    mode=mode,
                )

            if self.out_layers is not None and i in self.out_layers:
                out_idx = self.hierarchical_layers.index(i)
                out_norm = self.norms_block[out_idx](x)
                outs.append(out_norm)

            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        if self.out_layers is not None:
            return outs

        if training or self.return_hierarchical:
            hier = torch.cat(hier, dim=2)
            return hier
        else:
            x = self.norms_block[-1](x)
            return x

    def interpolate_pos_encoding(self, x, pos_embed):
        _, N, dim = pos_embed.shape

        if self.is_video:
            _, _, T, H, W = x.shape
            if H == self.img_height and W == self.img_width and T == self.num_frames:
                return pos_embed
            elif H == self.img_height and W == self.img_width and T < self.num_frames:
                new_N = int((T // self.tubelet_size) * (H // self.patch_size) * (W // self.patch_size))
                return pos_embed[:, :new_N, :]

            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            N_t = self.num_frames // self.tubelet_size
            N_h = self.img_height // self.patch_size
            N_w = self.img_width // self.patch_size
            assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"

            scale_factor = (T / N_t, H / N_h, W / N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed
        else:
            _, _, H, W = x.shape
            if H == self.img_height and W == self.img_width:
                return pos_embed

            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="bicubic",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


def vit_base(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


def vit_large(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


def vit_giant_xformers(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=22, mlp_ratio=48 / 11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


def vit_gigantic_xformers(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=1664, depth=48, num_heads=26, mlp_ratio=64 / 13,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


_ARCH_BUILDERS = {
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_giant_xformers": vit_giant_xformers,
    "vit_gigantic_xformers": vit_gigantic_xformers,
}


# ---------------------------------------------------------------------------
# builders + weight loading (from src/hub/backbones.py, VJEPA 2.1, encoder-only)
# ---------------------------------------------------------------------------

VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

# model_name -> (encoder arch, checkpoint file stem)
ARCH_NAME_MAP = {
    "vjepa2_1_vit_base_384": ("vit_base", "vjepa2_1_vitb_dist_vitG_384"),
    "vjepa2_1_vit_large_384": ("vit_large", "vjepa2_1_vitl_dist_vitG_384"),
    "vjepa2_1_vit_giant_384": ("vit_giant_xformers", "vjepa2_1_vitg_384"),
    "vjepa2_1_vit_gigantic_384": ("vit_gigantic_xformers", "vjepa2_1_vitG_384"),
}


def _clean_backbone_key(state_dict):
    for key, val in state_dict.copy().items():
        _ = state_dict.pop(key)
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        state_dict[key] = val
    return state_dict


def _make_vjepa2_1_encoder(
    model_name="vjepa2_1_vit_large_384",
    checkpoint_key="target_encoder",
    img_size=384,
    patch_size=16,
    tubelet_size=2,
    num_frames=64,
    pretrained: bool = True,
    **kwargs,
):
    """Build the VJEPA 2.1 encoder and (optionally) load pretrained weights.

    posetail discards the predictor, so only the encoder is built here. Weights
    are loaded strict=True (the 2.1 checkpoints have no pos_embed since the model
    uses RoPE) after stripping `module.`/`backbone.` prefixes.

    NOTE: upstream never forwards `n_output_distillation` to the encoder -- it only
    parameterizes the (discarded) predictor -- so the encoder always uses its
    default of 4 hierarchical outputs. posetail relies on this (it sets the scene
    embed dim to ``encoder.embed_dim * 4`` in hierarchical mode), so we mirror it
    by not passing `n_output_distillation` here.
    """
    vit_encoder_kwargs = dict(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    vit_encoder_kwargs.update(**kwargs)

    arch_name = ARCH_NAME_MAP[model_name][0]
    encoder = _ARCH_BUILDERS[arch_name](**vit_encoder_kwargs)

    if pretrained:
        model_file = ARCH_NAME_MAP[model_name][-1]
        url = VJEPA_BASE_URL + f"/{model_file}.pt"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        encoder_state_dict = _clean_backbone_key(state_dict[checkpoint_key])
        encoder.load_state_dict(encoder_state_dict, strict=True)

    return encoder


def vjepa2_1_vit_base_384(*, pretrained: bool = True, **kwargs):
    encoder = _make_vjepa2_1_encoder(
        model_name="vjepa2_1_vit_base_384",
        checkpoint_key="ema_encoder",
        img_size=384,
        pretrained=pretrained,
        **kwargs,
    )
    return encoder, None


def vjepa2_1_vit_large_384(*, pretrained: bool = True, **kwargs):
    encoder = _make_vjepa2_1_encoder(
        model_name="vjepa2_1_vit_large_384",
        checkpoint_key="ema_encoder",
        img_size=384,
        pretrained=pretrained,
        **kwargs,
    )
    return encoder, None


def vjepa2_1_vit_giant_384(*, pretrained: bool = True, **kwargs):
    encoder = _make_vjepa2_1_encoder(
        model_name="vjepa2_1_vit_giant_384",
        img_size=384,
        pretrained=pretrained,
        **kwargs,
    )
    return encoder, None


def vjepa2_1_vit_gigantic_384(*, pretrained: bool = True, **kwargs):
    encoder = _make_vjepa2_1_encoder(
        model_name="vjepa2_1_vit_gigantic_384",
        img_size=384,
        pretrained=pretrained,
        **kwargs,
    )
    return encoder, None
