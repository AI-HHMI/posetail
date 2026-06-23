#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from posetail.posetail.networks import EmbedV2V
from posetail.posetail.cube import is_point_visible, project_points_torch
from posetail.posetail.cube import CameraSelfAttention
from posetail.posetail.cube import signed_log1p, signed_expm1
from posetail.posetail.utils import get_fourier_encoding, apply_rope_1d

from einops import rearrange, repeat, einsum

from hub.backbones import (
    vjepa2_1_vit_base_384,
    vjepa2_1_vit_large_384,
    vjepa2_1_vit_giant_384,
    vjepa2_1_vit_gigantic_384,
)

# weird hackery for vjepa
import hub # from vjepa
import os
import sys
vjepa_path = os.path.dirname(os.path.dirname(hub.__path__[0]))
sys.path.append(vjepa_path)


def sample_feature_cubes_time(feature_planes, camera_group,
                              cube_centers, query_time, cube_interval,
                              corr_radius=1, downsample_ratio=1,
                              v2v=None):
    """Inputs:
     feature_planes: list of [B, T, C, H, W] tensors (one per camera)
     camera_group: list of cameras
     cube_centers: b k 3
     query_time: b k (time index for each query)
     cube_interval: single float
    
    Returns:
      volume: b d k total
    """
        
    cube_size = corr_radius * 2 + 1
    n_cams = len(feature_planes)
    B, K, _ = cube_centers.shape
    
    # get coordinates of each cube
    offsets = (torch.arange(cube_size, device=cube_centers.device, dtype=cube_centers.dtype) - corr_radius)
    xs, ys, zs = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    xyz_unit = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)  # (total, 3) unit offsets
    if torch.is_tensor(cube_interval) and cube_interval.ndim >= 1:
        # per-batch interval: (B,) → (B, total, 3)
        xyz = xyz_unit[None] * cube_interval[:, None, None]
        cube_coords = cube_centers[:, :, None, :] + xyz[:, None, :, :]  # (B, K, total, 3)
    else:
        xyz = xyz_unit * cube_interval  # (total, 3)
        cube_coords = cube_centers[..., None, :] + xyz  # (B, K, total, 3)
    cube_coords_flat = rearrange(cube_coords, 'b k total r -> (b k total) r')
    p2d_flat = project_points_torch(
        camera_group=camera_group, 
        coords_3d=cube_coords_flat, 
        downsample_factor=downsample_ratio,
    )

    p2d = rearrange(p2d_flat, 'ncams (b k total) r -> ncams b k total r',
                    b=B, k=K)

    all_samples = []
    all_masks = []
    for ix_cam in range(n_cams):
        b, t, d, h, w = feature_planes[ix_cam].shape
        scale = torch.tensor([w, h], device=p2d.device) 
        p2d_scaled = 2 * p2d[ix_cam] / scale - 1

        # Create visibility mask: True if within [-1, 1] bounds
        valid_mask = ((p2d_scaled[..., 0] >= -1) & (p2d_scaled[..., 0] <= 1) &
                      (p2d_scaled[..., 1] >= -1) & (p2d_scaled[..., 1] <= 1))
        
        # Gather the relevant time slices for each query: [B, K, C, H, W]
        b_idx = torch.arange(b, device=feature_planes[ix_cam].device)[:, None].expand(-1, K)
        feats_gathered = feature_planes[ix_cam][b_idx, query_time]  # b k d h w
        
        # For grid_sample, we need [B*K, C, H, W] input and [B*K, total, 1, 2] grid
        feats_flat = rearrange(feats_gathered, 'b k d h w -> (b k) d h w')
        grid_flat = rearrange(p2d_scaled, 'b k total r -> (b k) total 1 r')
        
        samples_flat = F.grid_sample(
            input=feats_flat,
            grid=grid_flat,
            align_corners=False,
            padding_mode="zeros")  # (b k) d total 1
        
        samples = rearrange(samples_flat, '(b k) d total 1 -> b d k total', b=b, k=K)
        all_samples.append(samples)
        all_masks.append(valid_mask)

    # volumes: cams b d k total            
    volumes = torch.stack(all_samples)
    masks = torch.stack(all_masks)  # ncams b k total
    masks_float = masks.float()
    masks_expanded = repeat(masks_float, "ncams b k total -> ncams b d k total",
                            d=volumes.shape[2])
    count = masks_expanded.sum(dim=0).clamp(min=1.0)
    mean_volume = (volumes * masks_expanded).sum(dim=0) / count
    
    mv_flat = rearrange(mean_volume, 'b d k (x y z) -> (b k) d z y x',
                        x=cube_size, y=cube_size, z=cube_size)
    if v2v is not None:
        mv_flat = v2v(mv_flat)

    mean_volume = rearrange(mv_flat, '(b k) d z y x -> b d k (x y z)',
                            b=B, k=K) 

    return mean_volume


def sample_patches(images: torch.Tensor, centers: torch.Tensor,
                   query_times: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Sample patches from images at given center coordinates and times.
    
    Args:
        images: [B, T, C, H, W]
        centers: [B, Z, R] pixel coordinates (x, y) of patch centers, R=2
        query_times: [B, Z] time indices into T
        patch_size: P
    
    Returns:
        patches: [B, Z, C, P, P]
    """
    B, T, C, H, W = images.shape
    Z = centers.shape[1]
    P = patch_size

    # Build patch offset grid: [P, P, 2]
    lin = torch.arange(P, dtype=centers.dtype, device=centers.device)
    grid_y, grid_x = torch.meshgrid(lin, lin, indexing='ij')
    offset_x = grid_x - (P - 1) / 2.0
    offset_y = grid_y - (P - 1) / 2.0
    offsets = torch.stack([offset_x, offset_y], dim=-1)  # [P, P, 2]

    # Compute absolute pixel coordinates: [B, Z, P, P, 2]
    px = centers[:, :, None, None, :] + offsets[None, None, :, :, :]  # [B, Z, P, P, 2]

    # Normalize to [-1, 1] for grid_sample with align_corners=False
    scales = torch.tensor([W, H], dtype=images.dtype, device=images.device)
    grid = (2.0 * px + 1.0) / scales - 1.0  # [B, Z, P, P, 2]

    # Gather frames at query times: [B, T, C, H, W] -> [B, Z, C, H, W]
    t_idx = repeat(query_times, 'b z -> b z c h w', c=C, h=H, w=W)
    frames = torch.gather(images, dim=1, index=t_idx)  # [B, Z, C, H, W]

    # Flatten batch and Z dimensions for grid_sample
    frames_flat = rearrange(frames, 'b z c h w -> (b z) c h w')
    grid_flat = rearrange(grid, 'b z p q r -> (b z) p q r')

    # Sample patches
    patches_flat = F.grid_sample(
        frames_flat,
        grid_flat,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )  # [B*Z, C, P, P]

    # Reshape back to [B, Z, C, P, P]
    patches = rearrange(patches_flat, '(b z) c p q -> b z c p q', b=B, z=Z)

    return patches


class PatchProcessor(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, 
                 conv_channels=[32, 64, 128], kernel_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Build conv layers
        layers = []
        prev_c = in_channels
        for c in conv_channels:
            layers.extend([
                nn.Conv2d(prev_c, c, kernel_size, padding=kernel_size//2),
                nn.GELU(),
            ])
            prev_c = c
        self.convs = nn.Sequential(*layers)
        
        # MLP to process flattened features
        mlp_in_dim = conv_channels[-1] * patch_size * patch_size
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
    def forward(self, patches):
        """
        Args:
            patches: [B, C, P, P]
        Returns:
            embed: [B, embed_dim]
        """
        # Apply convs
        x = self.convs(patches)  # [B, C_out, P, P]
        
        # Flatten spatial dimensions
        x = rearrange(x, 'b c p q -> b (c p q)')
        
        # Apply MLP
        embed = self.mlp(x)  # [B, embed_dim]
        
        return embed


class QueryEncoder(nn.Module):
    def __init__(self, embed_dim=256, decoder_dim=256,
                 n_frames=16, corr_radius=2, max_freq=10,
                 patch_size=9, use_volume_embedding=True,
                 principal_point_embedding=False,
                 intrinsic_embedding=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.corr_radius = corr_radius
        self.max_freq = max_freq
        self.patch_size = patch_size
        self.n_frames = n_frames
        self.use_volume_embedding = use_volume_embedding
        self.principal_point_embedding = principal_point_embedding
        self.intrinsic_embedding = intrinsic_embedding

        self.t_query_embed = nn.Embedding(n_frames, embed_dim)
        self.t_target_embed = nn.Embedding(n_frames, embed_dim)

        self.vis_embed = nn.Embedding(2, embed_dim)

        if self.use_volume_embedding:
            vdim = 8
            self.v2v = EmbedV2V(3, vdim)
            in_dim_vol = (corr_radius * 2 + 1) ** 3 * vdim
            self.linear_volume = nn.Linear(in_dim_vol, embed_dim)

        self.linear_pos = nn.Linear(4 * max_freq + 2, embed_dim)
        self.linear_depth = nn.Linear(2 * max_freq + 1, embed_dim)
        if self.principal_point_embedding:
            self.linear_pp = nn.Linear(4 * max_freq + 2, embed_dim)
        if self.intrinsic_embedding:
            self.linear_intrinsic = nn.Linear(4 * max_freq + 2, embed_dim)

        self.patch_processor = PatchProcessor(
            in_channels=3,
            patch_size=patch_size,
            embed_dim=embed_dim,
            conv_channels=[32, 64, 128],
        )

        self.depth_norm_scale = nn.Parameter(torch.tensor([1.0]))

        # Learnable missing tokens substituted for depth/volume in 2D queries
        self.missing_depth = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.missing_depth, std=0.02)
        if self.use_volume_embedding:
            self.missing_volume = nn.Parameter(torch.zeros(embed_dim))
            nn.init.normal_(self.missing_volume, std=0.02)
        if self.intrinsic_embedding:
            self.missing_intrinsic = nn.Parameter(torch.zeros(embed_dim))
            nn.init.normal_(self.missing_intrinsic, std=0.02)

        self.n_fusion_terms = (6 + int(self.use_volume_embedding)
                               + int(self.principal_point_embedding)
                               + int(self.intrinsic_embedding))
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * self.n_fusion_terms, self.n_fusion_terms),
            nn.Sigmoid()
        )
        nn.init.normal_(self.gate[0].weight, std=0.01)
        nn.init.constant_(self.gate[0].bias, 0.0)

        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(embed_dim * 4, decoder_dim),
        )

    def _interp_time_embed(self, emb, times, n_frames):
        """emb(times) with the learned time table linearly resized to n_frames first, so clips
        longer/shorter than the trained n_frames work (the table is a 1-D position encoding over
        frames). Backward-compatible no-op when n_frames == the stored table size."""
        if n_frames == emb.num_embeddings:
            return emb(times)
        w = F.interpolate(emb.weight.t().unsqueeze(0), size=n_frames,
                          mode='linear', align_corners=False).squeeze(0).t()
        return F.embedding(times, w)

    def forward(self, preprocessed_views, camera_group,
                query_coords, query_time, target_time,
                cube_scale):
        """
        Args:
            preprocessed_views: list of [B, T, C, H, W] tensors
            camera_group: list of camera dicts (can be None for 2D mode)
            query_coords: [B, T_query, R] where R=2 (2D) or R=3 (3D)
            query_time: [B, T_query] time indices
            target_time: [B, T_query] time indices
            cube_scale: float for cube sampling (ignored in 2D mode)

        Returns:
            [B, T_query, N_cams, decoder_dim]
        """
        B, T_query, coord_dim = query_coords.shape
        n_cams = len(preprocessed_views)

        sizes = torch.stack([
            torch.tensor([view.shape[-1], view.shape[-2]],
                         dtype=query_coords.dtype, device=query_coords.device)
            for view in preprocessed_views
        ])  # [n_cams, 2]  (W, H)

        # Build p2d_full [ncams, B, T_query, 2]: projected pixel coords per camera
        if coord_dim == 3:
            p2d_full = project_points_torch(camera_group, query_coords)
        else:
            # 2D coords are already pixel-space for the single camera
            p2d_full = rearrange(query_coords, 'b t r -> 1 b t r')

        # Position encoding (shared)
        pp = rearrange(p2d_full, 'ncams b t r -> b t ncams r') / sizes
        pp = pp * 2.0 - 1.0
        fourier_pos = get_fourier_encoding(pp, min_freq=0, max_freq=self.max_freq)
        fourier_pos = torch.cat([pp, fourier_pos], dim=-1)
        embed_pos = self.linear_pos(fourier_pos)

        if self.principal_point_embedding:
            principal_pt = torch.stack([
                (cam['mat'][:2, 2] - cam['offset']).to(query_coords.dtype)
                for cam in camera_group
            ])  # [n_cams, 2]
            principal_pt_norm = principal_pt / sizes * 2.0 - 1.0       # [n_cams, 2]
            principal_pt_norm = repeat(principal_pt_norm, 'cams r -> b t cams r', b=B, t=T_query)
            fourier_pp = get_fourier_encoding(principal_pt_norm, min_freq=0, max_freq=self.max_freq)
            fourier_pp = torch.cat([principal_pt_norm, fourier_pp], dim=-1)
            embed_pp = self.linear_pp(fourier_pp)                       # [B, T_query, n_cams, embed_dim]

        # Intrinsic (focal-length) embedding: normalized focal (fx/image_size, fy/image_size)
        # in the padded-canvas coordinate system the network actually sees. fx/fy and the
        # canvas are co-transformed by crop+resize, so this grows for tight crops (resize
        # scales fx up) -> a clean, dimensionless scale-regime signal. 2D queries get the
        # learnable missing_intrinsic token (mirrors missing_depth).
        if self.intrinsic_embedding:
            if coord_dim == 3:
                focal = torch.stack([
                    torch.stack([cam['mat'][0, 0], cam['mat'][1, 1]]).to(query_coords.dtype)
                    for cam in camera_group
                ])  # [n_cams, 2]  (fx, fy)
                focal_norm = focal / sizes                              # [n_cams, 2]  (fx/W, fy/H)
                focal_norm = repeat(focal_norm, 'cams r -> b t cams r', b=B, t=T_query)
                fourier_intr = get_fourier_encoding(focal_norm, min_freq=0, max_freq=self.max_freq)
                fourier_intr = torch.cat([focal_norm, fourier_intr], dim=-1)
                embed_intrinsic = self.linear_intrinsic(fourier_intr)   # [B, T_query, n_cams, embed_dim]
            else:
                embed_intrinsic = repeat(self.missing_intrinsic,
                                         'd -> b t c d', b=B, t=T_query, c=n_cams)

        # Patch embeddings (shared)
        patches = torch.stack([
            sample_patches(preprocessed_views[i], p2d_full[i],
                           query_time, self.patch_size)
            for i in range(n_cams)
        ])  # [n_cams, B, T_query, C, P, P]
        patches_flat = rearrange(patches, 'cams b t c p q -> (cams b t) c p q')
        embed_flat = self.patch_processor(patches_flat)
        embed_patch = rearrange(embed_flat, '(cams b t) d -> b t cams d',
                                cams=n_cams, b=B, t=T_query)

        # Time embeddings (shared). Resize the learned time tables to the actual clip length so
        # n_frames can differ from training (no-op when it matches).
        n_frames_cur = preprocessed_views[0].shape[1]
        embed_query_time  = repeat(self._interp_time_embed(self.t_query_embed, query_time, n_frames_cur),
                                   'b t d -> b t cams d', cams=n_cams)
        embed_target_time = repeat(self._interp_time_embed(self.t_target_embed, target_time, n_frames_cur),
                                   'b t d -> b t cams d', cams=n_cams)

        # Depth, visibility, volume: computed for 3D; missing tokens for 2D
        if coord_dim == 3:
            # Depth
            centers = torch.stack([cam['center'] for cam in camera_group]).to(query_coords.dtype)
            qc = rearrange(query_coords, 'b t r -> b t 1 r')
            cs_bnc = rearrange(cube_scale, 'cams b -> b 1 cams')
            raw_depths = torch.linalg.norm(qc - centers, dim=-1) / cs_bnc
            depths = torch.log(raw_depths + 1e-6) * self.depth_norm_scale
            dr = rearrange(depths, 'b t ncams -> b t ncams 1')
            fourier_depth = get_fourier_encoding(dr, min_freq=0, max_freq=self.max_freq)
            fourier_depth = torch.cat([dr, fourier_depth], dim=-1)
            embed_depth = self.linear_depth(fourier_depth)

            # Visibility
            qflat = rearrange(query_coords, 'b t r -> (b t) r')
            visible = torch.stack([is_point_visible(cam, qflat, margin=2)
                                   for cam in camera_group])
            visible = rearrange(visible, 'ncams (b t) -> b t ncams', b=B)
            embed_vis = self.vis_embed(visible.to(torch.int32))

            # Volume (optional)
            if self.use_volume_embedding:
                cube_scale_shared = torch.median(cube_scale, dim=0).values  # (B,)
                volumes = sample_feature_cubes_time(
                    preprocessed_views, camera_group, query_coords, query_time,
                    cube_scale_shared * 2, corr_radius=self.corr_radius, v2v=self.v2v)
                volumes = rearrange(volumes, 'b d t total -> b t 1 (d total)')
                embed_volume = self.linear_volume(volumes)
                embed_volume = repeat(embed_volume, 'b t 1 d -> b t cams d', cams=n_cams)
        else:
            embed_depth = repeat(self.missing_depth, 'd -> b t c d', b=B, t=T_query, c=n_cams)
            # Visibility is a plain bounds check on pixel coordinates
            margin = 2
            W, H = sizes[0, 0], sizes[0, 1]
            in_bounds = ((query_coords[..., 0] >= margin) & (query_coords[..., 0] < W - margin) &
                         (query_coords[..., 1] >= margin) & (query_coords[..., 1] < H - margin))
            embed_vis = self.vis_embed(rearrange(in_bounds, 'b t -> b t 1').to(torch.int32))
            if self.use_volume_embedding:
                embed_volume = repeat(self.missing_volume, 'd -> b t c d', b=B, t=T_query, c=n_cams)

        # Gated fusion (shared)
        embed_terms = [embed_patch, embed_query_time, embed_target_time,
                       embed_pos, embed_depth, embed_vis]
        if self.principal_point_embedding:
            embed_terms.append(embed_pp)
        if self.intrinsic_embedding:
            embed_terms.append(embed_intrinsic)
        if self.use_volume_embedding:
            embed_terms.append(embed_volume)

        embed_stack = torch.stack(embed_terms, dim=-2)
        embed_for_gate = rearrange(embed_stack, 'b t c n d -> b t c (n d)')
        weights = self.gate(embed_for_gate)
        combined_embed = einsum(weights, embed_stack, 'b t c n, b t c n d -> b t c d')

        combined_embed = self.fusion_norm(combined_embed)
        return self.fusion_mlp(combined_embed)


class SceneRepresentation(nn.Module):
    def __init__(self, version='large', freeze_encoder=True, n_frames=16, image_size=256,
                 hierarchical_features=True, decoder_dim=None,
                 proj_prenorm=False, proj_mlp=False,
                 video_encoder_finetune_last_n_layers=None):
        super().__init__()

        # Initialize encoder
        if version == 'base':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_base_384()
        elif version == 'large':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_large_384()
        elif version == 'giant':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_giant_384()
        elif version == 'gigantic':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_gigantic_384()

        self.encoder = vjepa_encoder

        self.encoder.return_hierarchical = hierarchical_features
        self.encoder.use_activation_checkpointing = True # not freeze_encoder

        if hierarchical_features:
            self.embed_dim = self.encoder.embed_dim * 4
        else:
            self.embed_dim = self.encoder.embed_dim

        self.patch_size = self.encoder.patch_size
        self.tubelet_size = self.encoder.tubelet_size

        # When set, only the last N transformer blocks (+ final norm) of the
        # video encoder are trainable; the rest stay frozen. None means every
        # encoder param is trainable whenever gradients are enabled.
        if video_encoder_finetune_last_n_layers is not None:
            n_blocks = len(self.encoder.blocks)
            assert 0 < video_encoder_finetune_last_n_layers <= n_blocks, (
                f'video_encoder_finetune_last_n_layers must be in [1, {n_blocks}], '
                f'got {video_encoder_finetune_last_n_layers}')
        self.video_encoder_finetune_last_n_layers = video_encoder_finetune_last_n_layers

        self.set_encoder_requires_grad(not freeze_encoder)
        if freeze_encoder:
            self.encoder.eval()

        self.n_frames = n_frames
        self.image_size = image_size

        n_tokens = (n_frames // self.tubelet_size) * (image_size // self.patch_size) * (image_size // self.patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_tokens, self.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02, a=-2 * 0.02, b=2 * 0.02)

        # Canonical (T',H',W') grid the pos_embed param was allocated for. forward()
        # trilinearly interpolates the pos_embed to the actual input grid when they differ, so
        # n_frames / image_size can change at finetune/inference WITHOUT resizing the parameter
        # (checkpoints still load; identity / no-op when the grids match -> backward-compatible).
        self._pe_grid = (n_frames // self.tubelet_size,
                         image_size // self.patch_size,
                         image_size // self.patch_size)

        if decoder_dim is not None:
            in_dim = self.embed_dim

            # Optional per-level pre-projection normalization. With hierarchical
            # features the scene vector is N VJEPA levels concatenated along the
            # feature dim, so we LayerNorm each level's chunk independently before
            # projecting -- targeting the cross-depth scale mismatch. Gracefully
            # degrades to a single LayerNorm when features are not hierarchical.
            if proj_prenorm:
                self.n_proj_levels = 4 if hierarchical_features else 1
                assert in_dim % self.n_proj_levels == 0
                level_dim = in_dim // self.n_proj_levels
                self.pre_norms = nn.ModuleList(
                    [nn.LayerNorm(level_dim) for _ in range(self.n_proj_levels)])
            else:
                self.n_proj_levels = 1
                self.pre_norms = None

            # Projection from the (concatenated) encoder dim down to decoder_dim:
            # either a single Linear or a small MLP bottleneck (Linear-GELU-Linear).
            if proj_mlp:
                self.kv_proj = nn.Sequential(
                    nn.Linear(in_dim, decoder_dim),
                    nn.GELU(),
                    nn.Linear(decoder_dim, decoder_dim),
                )
                for m in self.kv_proj:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
            else:
                self.kv_proj = nn.Linear(in_dim, decoder_dim)
                nn.init.xavier_uniform_(self.kv_proj.weight)
                nn.init.zeros_(self.kv_proj.bias)

            self.kv_norm = nn.LayerNorm(decoder_dim)
            self.embed_dim = decoder_dim
        else:
            self.kv_proj = None
            self.kv_norm = None
            self.pre_norms = None

    def set_encoder_requires_grad(self, requires_grad):
        """Toggle gradients for the underlying video encoder, e.g. to unfreeze
        it partway through training. When finetune_last_n_layers is set and
        gradients are being enabled, only the last N transformer blocks (plus
        the final norm, if the encoder variant has one) are made trainable."""
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad

        if requires_grad and self.video_encoder_finetune_last_n_layers is not None:
            # re-freeze everything, then unfreeze only the last N blocks (plus
            # the final norm if this encoder variant has one -- the VJEPA 2.1
            # VisionTransformer does not).
            for param in self.encoder.parameters():
                param.requires_grad = False
            trainable = list(self.encoder.blocks[-self.video_encoder_finetune_last_n_layers:])
            encoder_norm = getattr(self.encoder, 'norm', None)
            if encoder_norm is not None:
                trainable.append(encoder_norm)
            for module in trainable:
                for param in module.parameters():
                    param.requires_grad = True

        self.freeze_encoder = not requires_grad

    def _pos_embed_for(self, gT, gH, gW):
        """pos_embed resized to grid (gT,gH,gW): trilinear interp over time+space when the
        requested grid differs from the stored one, identity otherwise (backward-compatible)."""
        sT, sH, sW = self._pe_grid
        if (gT, gH, gW) == (sT, sH, sW):
            return self.pos_embed
        D = self.pos_embed.shape[-1]
        pe = self.pos_embed.reshape(1, sT, sH, sW, D).permute(0, 4, 1, 2, 3)
        pe = F.interpolate(pe, size=(gT, gH, gW), mode='trilinear', align_corners=False)
        return pe.permute(0, 2, 3, 4, 1).reshape(1, gT * gH * gW, D)

    def forward(self, views):
        """
        Args:
            views: list of [B, T, C, H, W] tensors (preprocessed images, one list per camera)

        Returns:
            encoded_views: [C, B, N_tokens, embed_dim] tensor
        """

        encoded_list = []
        for view in views:
            xr = rearrange(view, 'b t c h w -> b c t h w')
            feat = self.encoder(xr)  # [B, n_tokens, embed_dim]
            gT = view.shape[1] // self.tubelet_size
            gH = view.shape[3] // self.patch_size
            gW = view.shape[4] // self.patch_size
            feat = feat + self._pos_embed_for(gT, gH, gW)
            if self.kv_proj is not None:
                if self.pre_norms is not None:
                    chunks = torch.chunk(feat, self.n_proj_levels, dim=-1)
                    feat = torch.cat(
                        [norm(c) for norm, c in zip(self.pre_norms, chunks)], dim=-1)
                feat = self.kv_proj(feat)
                feat = self.kv_norm(feat)
            encoded_list.append(feat)
        encoded = torch.stack(encoded_list)  # [cams, B, n_tokens, embed_dim]

        return encoded


class TemporalSelfAttention(nn.Module):
    """Per-point temporal self-attention with 1-D RoPE.

    Expects input of shape (batch, T, embed_dim) where batch absorbs
    cams·B·K. Attends across the T (time) axis; out_proj is zero-initialized
    so the block is a no-op at init.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.num_heads)
        positions = torch.arange(T, device=x.device)
        q = apply_rope_1d(q, positions)
        k = apply_rope_1d(k, positions)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, num_modes=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = nn.Embedding(num_modes, dim)
        self.beta = nn.Embedding(num_modes, dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x, mode_idx):
        return self.norm(x) * (1 + self.gamma(mode_idx)) + self.beta(mode_idx)


class Decoder(nn.Module):
    def __init__(self, embed_dim=256, encoder_dim=1024,
                 num_heads=8, num_layers=8,
                 mlp_ratio=4.0, dropout=0.05,
                 use_camera_self_attention=True,
                 use_temporal_self_attention=False,
                 output_mode="direct",
                 head_3d_grid_size=8,
                 head_3d_grid_radius=1.0,
                 log_3d_output=False,
                 log_3d_eps=0.1,
                 depth_log_min=-2.5,
                 depth_log_max=2.0,
                 image_size=256,
                 f_eff_scale=False,
                 soft_argmax_temperature=0.5,
                 soft_argmax_threshold=20,
                 soft_argmax_temperature_learnable=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers
        self.use_camera_self_attention = use_camera_self_attention
        self.use_temporal_self_attention = use_temporal_self_attention
        self.output_mode = output_mode
        self.f_eff_scale = f_eff_scale
        self.head_3d_grid_size = head_3d_grid_size
        self.head_3d_grid_radius = head_3d_grid_radius
        self.image_size = image_size
        # Soft-argmax decode knobs (per-axis local-window expectation in _grid_decode). Defaults
        # (temp 0.5, window 20, fixed) reproduce the original hardcoded behavior exactly. The
        # temperature is the soft-argmax sharpness ("beta"); learnable lets it adapt per the data
        # (1 extra scalar param, loaded fresh under strict=False, no effect when not learnable).
        self.soft_argmax_threshold = int(soft_argmax_threshold)
        self._soft_argmax_temp_fixed = float(soft_argmax_temperature)
        self.log_soft_argmax_temp = (
            nn.Parameter(torch.tensor(math.log(float(soft_argmax_temperature))))
            if soft_argmax_temperature_learnable else None)
        # signed-log warp of the 3D output (3D only). See cube.signed_log1p/expm1.
        self.log_3d_output = log_3d_output
        self.log_3d_eps = log_3d_eps
        self.log_3d_c_range = float(math.log1p(head_3d_grid_radius / log_3d_eps))

        # Grid (classification) output modes, mirroring TrackerTapNext:
        #   "grid"      = absolute ray-local position via per-axis marginal soft-argmax.
        #   "gridresid" = the grid head emits a motion residual offset (added to the
        #                 per-track query anchor); depth stays the absolute log grid.
        # Both supervise the bins with cross-entropy (losses.py grid_softmax_loss).
        self.is_grid = output_mode in ('grid', 'gridresid')
        self.is_resid = output_mode in ('residual', 'resdirect', 'gridresid')

        if self.is_grid:
            G = head_3d_grid_size
            # 3D bin centres: signed-log spaced (denser near 0) when log_3d_output,
            # still spanning exactly [-radius, radius]; otherwise linear. Per-axis
            # (marginal) — NOT the joint cartesian grid (that explodes as G**3 and is
            # never CE-supervised).
            if log_3d_output:
                cr = self.log_3d_c_range
                grid_1d = signed_expm1(torch.linspace(-cr, cr, G), log_3d_eps)
            else:
                grid_1d = torch.linspace(-head_3d_grid_radius, head_3d_grid_radius, G)
            self.register_buffer("grid_1d", grid_1d)
            # Depth grid is ALWAYS linear-in-log over [depth_log_min, depth_log_max]
            # (its own representation; never touched by log_3d_output).
            self.register_buffer("depth_grid", torch.linspace(depth_log_min, depth_log_max, G))
            # 2D pixel bin centres at i+0.5 to match coordinate_softmax_loss's
            # round(target-0.5) quantization (losses.py:25-30).
            P = image_size
            self.register_buffer("pix_grid", torch.arange(P, dtype=torch.float32) + 0.5)
            self.g3d_lo, self.g3d_hi = -head_3d_grid_radius, head_3d_grid_radius
            self.gd_lo, self.gd_hi = depth_log_min, depth_log_max

        if self.use_camera_self_attention:
            self.camera_attns = nn.ModuleList([
                CameraSelfAttention(embed_dim=embed_dim,
                                    num_heads=num_heads)
                for _ in range(num_layers)
            ])
        else:
            self.camera_attns = None

        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kdim=encoder_dim,
                vdim=encoder_dim,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # Adaptive layer norms — mode-conditioned throughout the transformer stack
        self.norm0s = nn.ModuleList([AdaptiveLayerNorm(embed_dim) for _ in range(num_layers)])
        self.norm1s = nn.ModuleList([AdaptiveLayerNorm(embed_dim) for _ in range(num_layers)])
        self.norm2s = nn.ModuleList([AdaptiveLayerNorm(embed_dim) for _ in range(num_layers)])

        if self.use_temporal_self_attention:
            self.temporal_attns = nn.ModuleList([
                TemporalSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ])
            self.norm_ts = nn.ModuleList([AdaptiveLayerNorm(embed_dim) for _ in range(num_layers)])

        # Sliding-window latent hand-off: fuse the previous window's final decoder
        # latent (for the overlap frames) into this window's query tokens before the
        # transformer stack. LayerNorm gives the carried latent unit scale; the Linear
        # is zero-initialised so the hand-off is a no-op at load time -- a model
        # warm-started from a non-windowed checkpoint behaves identically until this
        # learns to use the carried state. No-op (and skipped) when prev_latent is None.
        self.latent_carry = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.latent_carry[1].weight)
        nn.init.zeros_(self.latent_carry[1].bias)

        # Per-mode output heads: index 0 = 2D, index 1 = 3D.
        # In grid mode the heads emit per-axis bin logits (marginal soft-argmax):
        #   3D    -> 3*G logits,  depth -> G logits,  2D -> 2*P pixel logits.
        # Otherwise they regress 3 / 1 / 2 values directly.
        head_3d_out    = 3 * head_3d_grid_size if self.is_grid else 3
        head_2d_out    = 2 * image_size        if self.is_grid else 2
        head_depth_out = head_3d_grid_size     if self.is_grid else 1

        def _make_heads(out_dim):
            return nn.ModuleList([
                nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_dim))
                for _ in range(2)
            ])

        self.heads_3d      = _make_heads(head_3d_out)
        self.heads_2d      = _make_heads(head_2d_out)
        self.heads_vis     = _make_heads(1)
        self.heads_conf    = _make_heads(1)
        self.heads_depth   = _make_heads(head_depth_out)
        self.heads_conf_3d = _make_heads(1)

        # Variance-matched, dimension-invariant head init.
        # The LayerNorm gives each head a unit-variance input (Sum_i x_i^2 = embed_dim), so for
        # W ~ N(0, std^2) with zero bias the head-output std is std * sqrt(embed_dim). We fix that
        # *output* std (sigma_out) and back out std = sigma_out / sqrt(embed_dim), so the effective
        # init scale is independent of latent_dim (a fixed std would grow as sqrt(embed_dim)).
        HEAD_OUT_STD_REG = 0.01
        HEAD_OUT_STD_LOGIT = 0.25
        reg_std = HEAD_OUT_STD_REG / (self.embed_dim ** 0.5)
        logit_std = HEAD_OUT_STD_LOGIT / (self.embed_dim ** 0.5)

        # Weight init — applied to both mode heads. In grid mode the 3D/2D/depth heads
        # are zero-init so the per-axis softmax starts uniform -> the decoded value is
        # the grid centre (0 ray-local / image centre / mid log-depth), mirroring
        # tracker_tapnext.py:194-197.
        for m in range(2):
            if self.is_grid:
                for head in [self.heads_3d[m][1], self.heads_2d[m][1], self.heads_depth[m][1]]:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
            else:
                for head in [self.heads_2d[m][1], self.heads_depth[m][1], self.heads_3d[m][1]]:
                    nn.init.normal_(head.weight, std=reg_std)
                    nn.init.zeros_(head.bias)

            for head in [self.heads_vis[m][1], self.heads_conf[m][1], self.heads_conf_3d[m][1]]:
                nn.init.normal_(head.weight, mean=0.0, std=logit_std)
                nn.init.zeros_(head.bias)

        # Learnable output scales (shared across modes), initialised to data-driven
        # magnitudes (see scripts/estimate_scale_stats.py).
        #
        # Absolute outputs (direct/grid 3D head, depth head) regress ~depth/cube_scale,
        # which equals the effective focal `f_eff` (~1e3, varies ~13x across datasets):
        #   - f_eff_scale on  → these are multiplied by f_eff per-camera in the forward,
        #     so the learnable residual collapses to ~1 (and stays uniform across datasets).
        #   - f_eff_scale off → the scale must absorb the whole f_eff, so it inits near the
        #     cross-dataset median (~1e3).
        # The 3D *residual* and the 2D outputs are motion / pixel quantities that carry no
        # f_eff, so they init the same regardless of the flag.
        #
        # Init values do not affect checkpoint loading (load_state_dict overwrites the
        # Parameters); when f_eff_scale is off the forward is also unchanged, so old
        # checkpoints load and infer identically.
        abs_scale = 1.0 if self.f_eff_scale else 1000.0

        if self.is_grid:
            # Grid modes: the bins carry the metric magnitude, so these scales are
            # unused in the forward. Defined (harmless) only so attribute access and
            # checkpoint round-trips stay uniform across modes.
            self.scale_3d = nn.Parameter(torch.tensor([abs_scale]))
            self.scale_2d = nn.Parameter(torch.tensor([128.0]))
        elif self.output_mode == 'direct':
            self.scale_3d = nn.Parameter(torch.tensor([abs_scale]))
            self.scale_2d = nn.Parameter(torch.tensor([128.0]))
        elif self.output_mode == 'residual':
            self.scale_3d = nn.Parameter(torch.tensor([8.0]))
            self.scale_2d = nn.Parameter(torch.tensor([6.0]))
        elif self.output_mode == 'resdirect':
            # index 0 = 2D-query head (direct/absolute), index 1 = 3D-query head (residual)
            self.scale_3d = nn.Parameter(torch.tensor([abs_scale, 8.0]))
            self.scale_2d = nn.Parameter(torch.tensor([6.0]))

        self.scale_depth = nn.Parameter(torch.tensor([abs_scale]))

    def _grid_decode(self, logits, grid_values):
        """Per-axis masked soft-argmax over a fixed grid of bin centres (threshold 20
        bins, temperature 0.5), mirroring tracker_tapnext.py:379-392. ``logits``
        (..., K), ``grid_values`` (K,) -> decoded value (...)."""
        soft_argmax_threshold = self.soft_argmax_threshold
        softmax_temperature = (torch.exp(self.log_soft_argmax_temp)
                               if self.log_soft_argmax_temp is not None
                               else self._soft_argmax_temp_fixed)
        K = logits.shape[-1]
        argmax = logits.argmax(dim=-1, keepdim=True)
        index = torch.arange(K, device=logits.device)
        mask = (torch.abs(argmax - index) <= soft_argmax_threshold).float()
        probs = F.softmax(logits * softmax_temperature, dim=-1) * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return (probs * grid_values.to(probs.dtype)).sum(dim=-1)

    def forward(self, scene_features, query_embeds, rays, mode_idx, prev_latent=None):
        """
        Args:
            scene_features: [N_cams, B, N_tokens, encoder_dim] from SceneRepresentation
            query_embeds: [B, T, K, N_cams, embed_dim]  T=frames, K=points
            rays: [B, T, K, N_cams, 4, 4]
            mode_idx: LongTensor of shape [1] — 0 for 2D queries, 1 for 3D queries
            prev_latent: [B, T, K, N_cams, embed_dim] or None — the previous sliding
                window's final decoder latent, frame-aligned to this window, fused into
                the query tokens before the stack (sliding-window hand-off). None for the
                first window / non-windowed pass.
        Returns:
            outputs: [B, T, K, N_cams, out_dim]
            grid_logits: dict or None
            latent: [B, T, K, N_cams, embed_dim] — this window's final decoder latent,
                to carry into the next window.
        """
        B, T, K, N_cams, embed_dim = query_embeds.shape
        assert embed_dim == self.embed_dim

        kv = rearrange(scene_features, 'cams b tokens dim -> (cams b) tokens dim')
        x = rearrange(query_embeds, 'b t k cams dim -> (cams b) (t k) dim')
        rays_r = rearrange(rays, 'b t k cams d e -> (b t k) cams d e')

        # Sliding-window hand-off: add the carried latent into the query tokens.
        # Zero-init Linear -> no-op until trained (warm-start safe); see __init__.
        if prev_latent is not None:
            carry = rearrange(prev_latent, 'b t k cams dim -> (cams b) (t k) dim')
            x = x + self.latent_carry(carry)

        for layer_idx in range(self.num_layers):
            if self.use_camera_self_attention:
                x_cam = rearrange(x, '(cams b) (t k) dim -> (b t k) cams dim',
                                  b=B, cams=N_cams, t=T, k=K)
                attn_out = self.camera_attns[layer_idx](self.norm0s[layer_idx](x_cam, mode_idx), rays_r)
                x_cam = x_cam + attn_out
                x = rearrange(x_cam, '(b t k) cams dim -> (cams b) (t k) dim',
                              b=B, cams=N_cams, t=T, k=K)

            if self.use_temporal_self_attention:
                x_t = rearrange(x, '(cams b) (t k) dim -> (cams b k) t dim',
                                cams=N_cams, b=B, t=T, k=K)
                attn_out = self.temporal_attns[layer_idx](self.norm_ts[layer_idx](x_t, mode_idx))
                x_t = x_t + attn_out
                x = rearrange(x_t, '(cams b k) t dim -> (cams b) (t k) dim',
                              cams=N_cams, b=B, t=T, k=K)

            x_normed = self.norm1s[layer_idx](x, mode_idx)
            attn_out, _ = self.cross_attns[layer_idx](
                query=x_normed, key=kv, value=kv, need_weights=False
            )
            x = x + attn_out

            x = x + self.mlps[layer_idx](self.norm2s[layer_idx](x, mode_idx))

        m = mode_idx.item()
        grid_logits = None
        if self.is_grid:
            # Per-axis marginal soft-argmax over fixed bins, mirroring TrackerTapNext.
            # The decoded values carry their own metric magnitude, so the learnable
            # scale_3d/scale_2d/scale_depth are bypassed. Raw bin logits are returned
            # for the cross-entropy supervision in losses.py.
            G = self.head_3d_grid_size
            raw_3d = self.heads_3d[m](x)                                    # (..., 3G)
            logits_3d = rearrange(raw_3d, '... (d k) -> ... d k', d=3, k=G)  # (..., 3, G)
            out_3d = self._grid_decode(logits_3d, self.grid_1d)            # (..., 3)

            logits_depth = self.heads_depth[m](x)                          # (..., G)
            # decode log-depth then exp -> normalized depth (>0); tracker_encoder
            # multiplies by cube_scale [* f_eff] and must NOT softplus this.
            out_depth = torch.exp(self._grid_decode(logits_depth, self.depth_grid))[..., None]

            raw_2d = self.heads_2d[m](x)                                    # (..., 2P) [x|y]
            logits_2d_da = rearrange(raw_2d, '... (a p) -> ... a p', a=2, p=self.image_size)
            out_2d = self._grid_decode(logits_2d_da, self.pix_grid)        # (..., 2) abs pixels

            # Reshape from the flat (cams b)(t k) working layout to b t k cams ...,
            # matching `output` below (tracker_encoder then moves cams to the front).
            grid_logits = {
                'logits_3d': rearrange(logits_3d, '(cams b) (t k) d g -> b t k cams d g',
                                       cams=N_cams, b=B, t=T, k=K),     # (b,t,k,cams,3,G)
                'logits_depth': rearrange(logits_depth, '(cams b) (t k) g -> b t k cams g',
                                          cams=N_cams, b=B, t=T, k=K),  # (b,t,k,cams,G)
                'logits_2d': rearrange(raw_2d, '(cams b) (t k) p -> b t k cams p',
                                       cams=N_cams, b=B, t=T, k=K),     # (b,t,k,cams,2P)
            }
        else:
            if self.log_3d_output:
                # Continuous warp: value = signed_expm1(compressed head output). Clamp to
                # c_range + fp32 keep expm1/its gradient bounded; scale_3d is dropped (eps
                # sets the slope, the clamp the reach). Init head ~0 -> output 0 (parity).
                cr = self.log_3d_c_range
                out_3d = signed_expm1(self.heads_3d[m](x).clamp(-cr, cr), self.log_3d_eps).to(x.dtype)
            else:
                scale_3d = self.scale_3d[m] if self.output_mode == 'resdirect' else self.scale_3d
                out_3d = self.heads_3d[m](x) * scale_3d
            out_2d    = self.heads_2d[m](x) * self.scale_2d
            out_depth = self.heads_depth[m](x) * self.scale_depth
        out_vis     = self.heads_vis[m](x)
        out_conf    = self.heads_conf[m](x)
        out_conf_3d = self.heads_conf_3d[m](x)
        output = torch.cat([out_3d, out_2d, out_vis, out_conf, out_depth, out_conf_3d], dim=-1)

        output = rearrange(output, '(cams b) (t k) dim -> b t k cams dim',
                           cams=N_cams, b=B, t=T, k=K)
        latent = rearrange(x, '(cams b) (t k) dim -> b t k cams dim',
                           cams=N_cams, b=B, t=T, k=K)
        return output, grid_logits, latent
