#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from posetail.posetail.networks import EmbedV2V
from posetail.posetail.cube import is_point_visible, project_points_torch
from posetail.posetail.cube import CameraSelfAttention
from posetail.posetail.cube import signed_log1p, signed_expm1
from posetail.posetail.utils import get_fourier_encoding, apply_rope_1d, get_3d_sincos_pos_embed

from einops import rearrange, repeat, einsum

from posetail.posetail.vjepa2 import (
    vjepa2_1_vit_base_384,
    vjepa2_1_vit_large_384,
    vjepa2_1_vit_giant_384,
    vjepa2_1_vit_gigantic_384,
)


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
                 intrinsic_embedding=False,
                 occlusion_embedding=False,
                 time_embed_mode='learned'):
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
        self.occlusion_embedding = occlusion_embedding

        # Query/target time encoding. 'learned' (default) = the original learned frame
        # tables + length-interpolation (backward-compatible, bit-identical). 'fourier_rel'
        # = Fourier encodings of absolute query, absolute target, AND the relative gap
        # (target-query) as a third fusion term -- length-agnostic and gap-consistent.
        assert time_embed_mode in ('learned', 'fourier_rel'), \
            f"time_embed_mode must be 'learned' | 'fourier_rel', got {time_embed_mode!r}"
        self.time_embed_mode = time_embed_mode
        # Fixed normalizer for Fourier time encodings (NOT the runtime clip length -- a fixed
        # constant keeps a given absolute frame gap mapped to the same encoding at any T).
        self.time_norm = float(n_frames)
        if self.time_embed_mode == 'learned':
            self.t_query_embed = nn.Embedding(n_frames, embed_dim)
            self.t_target_embed = nn.Embedding(n_frames, embed_dim)
        else:  # 'fourier_rel'
            self.linear_query_time = nn.Linear(2 * max_freq + 1, embed_dim)
            self.linear_target_time = nn.Linear(2 * max_freq + 1, embed_dim)
            self.linear_gap = nn.Linear(2 * max_freq + 1, embed_dim)

        self.vis_embed = nn.Embedding(2, embed_dim)

        # Occlusion query term: a 3-valued per-camera state {occluded=0, visible=1,
        # unknown=-1} indexed as value+1 (-1->0, 0->1, 1->2). Complements (does not
        # replace) the geometric vis_embed: is_point_visible is True for both an on-ray
        # foreground and background point, so occlusion carries the bit that separates them.
        if self.occlusion_embedding:
            self.occ_embed = nn.Embedding(3, embed_dim)

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
                               + int(self.intrinsic_embedding)
                               + int(self.occlusion_embedding)
                               # fourier_rel adds a dedicated relative-gap fusion term
                               + int(self.time_embed_mode == 'fourier_rel'))
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

    def _fourier_time(self, times, linear):
        """Fourier-encode a scalar frame quantity (absolute frame or relative gap) and
        project to embed_dim. times: (B, T) int/float -> (B, T, embed_dim). Uses a FIXED
        normalizer (self.time_norm) so a given absolute value maps to the same encoding at
        any clip length. Mirrors the depth Fourier pattern."""
        # get_fourier_encoding expects (b, s, n, r); use n=r=1 for the scalar time, then squeeze.
        s = (times.to(torch.float32) / self.time_norm)[..., None, None]   # (B, T, 1, 1)
        feat = torch.cat([s, get_fourier_encoding(s, min_freq=0, max_freq=self.max_freq)], dim=-1)
        return linear(feat)[..., 0, :]                                    # (B, T, embed_dim)

    def forward(self, preprocessed_views, camera_group,
                query_coords, query_time, target_time,
                cube_scale, occlusion=None):
        """
        Args:
            preprocessed_views: list of [B, T, C, H, W] tensors
            camera_group: list of camera dicts (can be None for 2D mode)
            query_coords: [B, T_query, R] where R=2 (2D) or R=3 (3D)
            query_time: [B, T_query] time indices
            target_time: [B, T_query] time indices
            cube_scale: float for cube sampling (ignored in 2D mode)
            occlusion: [B, T_query, N_cams] per-camera occlusion state {0,1,-1} for the
                occlusion_embedding term, or None (treated as all-unknown). Only used
                when self.occlusion_embedding is set.

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

        # Time embeddings (shared, not per-camera -> computed at (b,t,d) then broadcast over cams).
        embed_gap = None
        if self.time_embed_mode == 'learned':
            # Resize the learned time tables to the actual clip length so n_frames can differ
            # from training (no-op when it matches).
            n_frames_cur = preprocessed_views[0].shape[1]
            embed_query_time  = repeat(self._interp_time_embed(self.t_query_embed, query_time, n_frames_cur),
                                       'b t d -> b t cams d', cams=n_cams)
            embed_target_time = repeat(self._interp_time_embed(self.t_target_embed, target_time, n_frames_cur),
                                       'b t d -> b t cams d', cams=n_cams)
        else:  # 'fourier_rel': absolute query + absolute target + relative gap (extra term)
            embed_query_time  = repeat(self._fourier_time(query_time, self.linear_query_time),
                                       'b t d -> b t cams d', cams=n_cams)
            embed_target_time = repeat(self._fourier_time(target_time, self.linear_target_time),
                                       'b t d -> b t cams d', cams=n_cams)
            embed_gap = repeat(self._fourier_time(target_time - query_time, self.linear_gap),
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

        # Occlusion (optional): per-camera 3-valued state {occluded=0, visible=1,
        # unknown=-1}, indexed as value+1 (-1->0, 0->1, 1->2). None -> all-unknown (index 0),
        # matching occlusion-dropout during training and no-info queries at inference.
        if self.occlusion_embedding:
            if occlusion is None:
                occ_idx = torch.zeros((B, T_query, n_cams), dtype=torch.long,
                                      device=query_coords.device)
            else:
                occ_idx = (occlusion.to(torch.long) + 1).clamp_(0, 2)
            embed_occ = self.occ_embed(occ_idx)  # [B, T_query, n_cams, embed_dim]

        # Gated fusion (shared)
        embed_terms = [embed_patch, embed_query_time, embed_target_time]
        if embed_gap is not None:
            # fourier_rel: dedicated relative-gap fusion term (kept adjacent to the time terms).
            embed_terms.append(embed_gap)
        embed_terms += [embed_pos, embed_depth, embed_vis]
        if self.principal_point_embedding:
            embed_terms.append(embed_pp)
        if self.intrinsic_embedding:
            embed_terms.append(embed_intrinsic)
        if self.occlusion_embedding:
            embed_terms.append(embed_occ)
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
                 video_encoder_finetune_last_n_layers=None,
                 pos_embed_mode='learned'):
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

        # Positional signal added on TOP of the VJEPA encoder output (which already carries
        # the backbone's own positional scheme). Modes:
        #   'learned' : trainable absolute table, trilinearly interpolated to the input grid
        #               (the original behavior; backward-compatible default).
        #   'sincos'  : fixed factorized 3-D sin/cos, computed analytically at the input grid
        #               -- length-agnostic, no parameter, no interpolation.
        #   'none'    : ablation -- no added positional signal; rely solely on the backbone.
        # Only 'learned' registers a parameter; the rest of the module is identical.
        #   'spatial' : learned SPATIAL-ONLY table (gH*gW), shared across all frames --
        #               pairs with decoder temporal RoPE (see TrackerEncoder 'ropepos'):
        #               spatial position is additive/absolute, temporal is relative RoPE.
        assert pos_embed_mode in ('learned', 'sincos', 'none', 'spatial'), \
            f"pos_embed_mode must be 'learned' | 'sincos' | 'none' | 'spatial', got {pos_embed_mode!r}"
        self.pos_embed_mode = pos_embed_mode
        # Encoder-dim positional width (self.embed_dim is reassigned to decoder_dim below).
        self.pos_embed_dim = self.embed_dim
        self._pe_grid = None
        self._pe_grid_spatial = None
        if self.pos_embed_mode == 'learned':
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
        elif self.pos_embed_mode == 'spatial':
            # Spatial-only learned table (gH*gW, D), broadcast across all frames. No time
            # axis: temporal position is supplied by the decoder's cross-attention RoPE.
            gH0 = image_size // self.patch_size
            gW0 = image_size // self.patch_size
            self.pos_embed = nn.Parameter(torch.zeros(1, gH0 * gW0, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02, a=-2 * 0.02, b=2 * 0.02)
            self._pe_grid_spatial = (gH0, gW0)
        else:
            self.pos_embed = None
        # Per-(grid,device,dtype) cache for the fixed sincos table (recomputing the
        # ~N*D tensor every forward would be wasteful; the grid is constant in practice).
        self._sincos_cache = {}

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

    def _spatial_pos_embed_for(self, gT, gH, gW):
        """Spatial-only learned pos_embed for grid (gH,gW), tiled across gT frames.
        Bilinearly interpolates the stored (gH0,gW0) table when the requested spatial grid
        differs (identity otherwise), then repeats it t-major -> (1, gT*gH*gW, D). The same
        spatial code is shared by every frame; time is handled by the decoder RoPE."""
        sH, sW = self._pe_grid_spatial
        D = self.pos_embed.shape[-1]
        if (gH, gW) == (sH, sW):
            spatial = self.pos_embed                                   # (1, gH*gW, D)
        else:
            pe = self.pos_embed.reshape(1, sH, sW, D).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(gH, gW), mode='bilinear', align_corners=False)
            spatial = pe.permute(0, 2, 3, 1).reshape(1, gH * gW, D)
        # Tokens are ordered t-major, so tiling the spatial block gT times gives each frame
        # the identical spatial code.
        return spatial.repeat(1, gT, 1)                               # (1, gT*gH*gW, D)

    def _sincos_pos_embed_for(self, gT, gH, gW, device, dtype):
        """Fixed factorized 3-D sin/cos table for grid (gT,gH,gW), cached per
        (grid, device, dtype) since the grid is constant across a run."""
        key = (gT, gH, gW, device, dtype)
        pe = self._sincos_cache.get(key)
        if pe is None:
            pe = get_3d_sincos_pos_embed(self.pos_embed_dim, gT, gH, gW, device, dtype)
            self._sincos_cache[key] = pe
        return pe

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
            if self.pos_embed_mode != 'none':
                gT = view.shape[1] // self.tubelet_size
                gH = view.shape[3] // self.patch_size
                gW = view.shape[4] // self.patch_size
                if self.pos_embed_mode == 'learned':
                    feat = feat + self._pos_embed_for(gT, gH, gW)
                elif self.pos_embed_mode == 'spatial':
                    feat = feat + self._spatial_pos_embed_for(gT, gH, gW)
                else:  # 'sincos'
                    feat = feat + self._sincos_pos_embed_for(gT, gH, gW, feat.device, feat.dtype)
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


class RoPECrossAttention(nn.Module):
    """Cross-attention (pose-query tokens attend to scene tokens) with 1-D temporal RoPE on
    Q and K. Queries are rotated by their frame index, keys by their scene-token frame, both
    in the same frame units, so the attention score is biased by RELATIVE time -- and is
    length-agnostic (no table, defined at any T). Drop-in replacement for the nn.MultiheadAttention
    cross-attention used in the Decoder; keys/values come from the scene tokens (kv_dim).

    Unlike TemporalSelfAttention, out_proj is NOT zero-initialised: cross-attention is the main
    information pathway (not an added residual block), so it must contribute from step 0."""

    def __init__(self, embed_dim, num_heads, kv_dim, cross_attn_dim=None,
                 dropout=0.0, rope_base=10000.0):
        super().__init__()
        # Attention (q/k/v) width, decoupled from the decoder residual stream (embed_dim),
        # exactly like DecoupledCrossAttention. Defaults to embed_dim, which keeps the
        # projections at their original shapes so existing rope checkpoints load unchanged.
        attn_dim = cross_attn_dim if cross_attn_dim is not None else embed_dim
        assert attn_dim % num_heads == 0, (
            f'cross_attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})')
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.dropout = dropout
        # RoPE frequency base. The 10000 default is tuned for thousand-token contexts; over
        # short clips (~16-24 frames) most dim-pairs barely rotate, so a smaller base (~100-1000)
        # spreads angular resolution across the few positions actually present.
        self.rope_base = rope_base
        self.q_proj = nn.Linear(embed_dim, attn_dim)
        self.k_proj = nn.Linear(kv_dim, attn_dim)
        self.v_proj = nn.Linear(kv_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, embed_dim)

    def forward(self, query, kv, q_pos, k_pos):
        """query: (Bc, Lq, embed_dim); kv: (Bc, Lk, kv_dim);
        q_pos: (Lq,) query frame per token; k_pos: (Lk,) scene-token frame. Same units."""
        q = rearrange(self.q_proj(query), 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(self.k_proj(kv),    'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(self.v_proj(kv),    'b l (h d) -> b h l d', h=self.num_heads)
        q = apply_rope_1d(q, q_pos, base=self.rope_base)
        k = apply_rope_1d(k, k_pos, base=self.rope_base)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = rearrange(out, 'b h l d -> b l (h d)')
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


class DecoupledCrossAttention(nn.Module):
    """Cross-attention whose internal (q/k/v/attention) width is decoupled from the
    decoder residual-stream width.

    Queries come from the decoder stream (``latent_dim``); keys/values come from the scene
    representation (``kv_dim`` = scene_proj_dim). All three are projected to ``cross_attn_dim``
    (the attention width), attention runs over ``num_heads`` heads of size
    ``cross_attn_dim // num_heads``, and ``out_proj`` maps back to ``latent_dim`` for the
    residual add. This lets the scene->decoder readout capacity rise (a larger
    ``cross_attn_dim``) without widening the rest of the decoder stack.

    With ``cross_attn_dim == latent_dim`` (and ``kv_dim`` unchanged) the block is numerically
    equivalent to ``nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads,
    kdim=vdim=kv_dim, batch_first=True)``: split q/k/v/out ``nn.Linear``s of the same shapes,
    the same ``scaled_dot_product_attention`` kernel, and the same ``1/sqrt(head_dim)`` scale.
    Existing nn.MultiheadAttention checkpoints therefore reproduce their predictions exactly
    once their fused ``in_proj`` is split into q/k/v by ``train_utils._convert_cross_attn``.
    """

    def __init__(self, latent_dim, kv_dim, cross_attn_dim, num_heads, dropout=0.0):
        super().__init__()
        assert cross_attn_dim % num_heads == 0, (
            f'cross_attn_dim ({cross_attn_dim}) must be divisible by num_heads ({num_heads})')
        self.latent_dim = latent_dim
        self.kv_dim = kv_dim
        self.cross_attn_dim = cross_attn_dim
        self.num_heads = num_heads
        self.head_dim = cross_attn_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(latent_dim, cross_attn_dim)
        self.k_proj = nn.Linear(kv_dim, cross_attn_dim)
        self.v_proj = nn.Linear(kv_dim, cross_attn_dim)
        self.out_proj = nn.Linear(cross_attn_dim, latent_dim)
        # xavier on the projections (dimension-robust; matches nn.MultiheadAttention's
        # in_proj init style and keeps the wider-cross_attn_dim case well-scaled). Loaded
        # checkpoints overwrite these, so init never affects backward-compatible loads.
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, query, kv, attn_mask=None):
        """query: (B, Lq, latent_dim); kv: (B, Lk, kv_dim) -> (B, Lq, latent_dim).

        attn_mask: optional (B, 1|num_heads, Lq, Lk) additive float or bool mask
        (True = keep, matching F.scaled_dot_product_attention); None = unmasked.
        """
        q = rearrange(self.q_proj(query), 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(self.k_proj(kv),    'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(self.v_proj(kv),    'b l (h d) -> b h l d', h=self.num_heads)
        dropout_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        attn = rearrange(attn, 'b h l d -> b l (h d)')
        return self.out_proj(attn)


class Decoder(nn.Module):
    def __init__(self, embed_dim=256, encoder_dim=1024,
                 num_heads=8, num_layers=8,
                 cross_attn_dim=None,
                 mlp_ratio=4.0, dropout=0.05,
                 use_camera_self_attention=True,
                 use_temporal_self_attention=False,
                 cross_attn_rope=False,
                 cross_attn_rope_base=10000.0,
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
                 soft_argmax_temperature_learnable=False,
                 enable_subpixel_refinement=False,
                 subpixel_scale=0.05,
                 subpixel_temperature=10.0,
                 grid_decode_space="head",
                 learnable_scale=False,
                 learnable_scale_depth=False,
                 scale_init=1.0,
                 scale_delta=2.0,
                 use_memory=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        # Cross-attention internal width. Defaults to embed_dim (= latent_dim), which makes the
        # block numerically equivalent to the previous nn.MultiheadAttention; set larger to raise
        # the scene->decoder readout capacity without widening the decoder residual stream.
        self.cross_attn_dim = cross_attn_dim if cross_attn_dim is not None else embed_dim
        self.num_layers = num_layers
        self.use_camera_self_attention = use_camera_self_attention
        self.use_temporal_self_attention = use_temporal_self_attention
        self.cross_attn_rope = cross_attn_rope
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
        # Soft-argmax averaging space for the 3D decode (only meaningful with log_3d_output):
        #   "head"   = average the head-unit bin centres directly (out_3d = Σ p_k * grid_1d_k).
        #              The centres are convex-spaced (signed_expm1), so a broad distribution
        #              OVERSHOOTS through the warp (Jensen bias at large motion). DEFAULT — the
        #              original behaviour, bit-for-bit when "head".
        #   "warped" = average in the uniform WARPED space first, then expm1
        #              (out_3d = signed_expm1(Σ p_k * w_k)); overshoot-free. Adds no params/buffers
        #              (the warped centres are signed_log1p(grid_1d), recovered exactly), so it is
        #              load-compatible with any checkpoint. Enable for a future retrain.
        assert grid_decode_space in ("head", "warped"), grid_decode_space
        self.grid_decode_space = grid_decode_space
        # Optional sub-pixel/sub-bin 3D refinement (supervised by the EXISTING coords regression
        # loss, no new term). Zero-init offset head -> starts as an exact no-op; disabled by default
        # (no head at all). When enabled, the decode is a SHARPENED ARGMAX: a learnable high
        # temperature (subpixel_temperature) peaks the per-bin weights toward one-hot at the argmax
        # bin, so the decode approaches centre[argmax] + scale * bin_width * o_{argmax} (~one position
        # + one offset) while staying fully differentiable. See forward().
        #
        # subpixel_scale is measured in BINS: the offset is multiplied by subpixel_bin_width (the
        # adjacent-centre spacing in the space the offset is applied in), so subpixel_scale=0.5 means
        # up to half a bin of correction. Under log_3d_output the offset is applied in WARPED space
        # (uniform bins), else in head space; subpixel_bin_width below matches that space.
        self.enable_subpixel_refinement = enable_subpixel_refinement
        self.subpixel_scale = float(subpixel_scale)
        # Warped (log_3d) or head bin spacing between adjacent centres, so subpixel_scale -> bins.
        if log_3d_output:
            _bin_width = 2.0 * self.log_3d_c_range / max(head_3d_grid_size - 1, 1)   # warped units/bin
        else:
            _bin_width = 2.0 * head_3d_grid_radius / max(head_3d_grid_size - 1, 1)   # head units/bin
        self.subpixel_bin_width = float(_bin_width)

        # Grid (classification) output modes, mirroring TrackerTapNext:
        #   "grid"      = absolute ray-local position via per-axis marginal soft-argmax.
        #   "gridresid" = the grid head emits a motion residual offset (added to the
        #                 per-track query anchor); depth stays the absolute log grid.
        # Both supervise the bins with cross-entropy (losses.py grid_softmax_loss).
        self.is_grid = output_mode in ('grid', 'gridresid', 'gridnorm')
        self.is_resid = output_mode in ('residual', 'resdirect', 'gridresid')
        # gridnorm: the grid works in a gauge-free frame; a per-camera (scale, offset)
        # is solved from the query correspondences downstream (tracker_encoder), so the
        # decoder emits raw soft-argmax values (no cube_scale/f_eff, linear depth gauge).
        self.is_gridnorm = output_mode == 'gridnorm'

        # learnable_scale: decode a positive per-token scale from the latent and multiply
        # the metric 3D output (and depth) by it (a bounded, adaptive correction on the
        # mode's base scale). s = scale_init * exp(clamp(head(x), ±scale_delta)); head init
        # 0 -> s = scale_init (no-op at scale_init=1). Composes with any mode; redundant for
        # gridnorm (which already SOLVES the gauge), so it is asserted off there upstream.
        self.learnable_scale = bool(learnable_scale)
        self.learnable_scale_depth = bool(learnable_scale_depth)
        self.scale_init = float(scale_init)
        self.scale_delta = float(scale_delta)
        # scale = scale_init * exp(clamp(head, ±delta)); the head is zero-init so the scale
        # STARTS at scale_init (a no-op multiply when scale_init=1), and exp(clamp) floors it
        # at scale_init*exp(-delta) > 0 forever. scale_init MUST be > 0: it also divides the
        # grid CE target, so scale_init=0 would zero the residual and blow up the target.
        if self.learnable_scale or self.learnable_scale_depth:
            assert self.scale_init > 0, f'scale_init must be > 0, got {self.scale_init}'

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
            if self.is_gridnorm:
                # gridnorm: LINEAR depth gauge over the same fixed box as the 3D grid; a
                # per-camera affine (scale_d, offset_d) solved from query depths maps it to
                # metric downstream (no exp/log, no cube_scale/f_eff).
                self.register_buffer("depth_grid",
                                     torch.linspace(-head_3d_grid_radius, head_3d_grid_radius, G))
                self.gd_lo, self.gd_hi = -head_3d_grid_radius, head_3d_grid_radius
            else:
                # Depth grid is linear-in-log over [depth_log_min, depth_log_max]
                # (its own representation; never touched by log_3d_output).
                self.register_buffer("depth_grid", torch.linspace(depth_log_min, depth_log_max, G))
                self.gd_lo, self.gd_hi = depth_log_min, depth_log_max
            # 2D pixel bin centres at i+0.5 to match coordinate_softmax_loss's
            # round(target-0.5) quantization (losses.py:25-30).
            P = image_size
            self.register_buffer("pix_grid", torch.arange(P, dtype=torch.float32) + 0.5)
            self.g3d_lo, self.g3d_hi = -head_3d_grid_radius, head_3d_grid_radius

        if self.use_camera_self_attention:
            self.camera_attns = nn.ModuleList([
                CameraSelfAttention(embed_dim=embed_dim,
                                    num_heads=num_heads)
                for _ in range(num_layers)
            ])
        else:
            self.camera_attns = None

        # Cross-attention: pose queries attend to scene tokens. With cross_attn_rope the
        # scene-token positions are injected as 1-D temporal RoPE inside the QK product
        # (length-agnostic, relative time) instead of via an additive scene pos_embed; pair
        # it with scene_pos_embed_mode='none'. Default = the original additive nn.MultiheadAttention.
        if self.cross_attn_rope:
            self.cross_attns = nn.ModuleList([
                RoPECrossAttention(embed_dim=embed_dim, num_heads=num_heads,
                                   kv_dim=encoder_dim, cross_attn_dim=self.cross_attn_dim,
                                   dropout=dropout, rope_base=cross_attn_rope_base)
                for _ in range(num_layers)
            ])
        else:
            self.cross_attns = nn.ModuleList([
                DecoupledCrossAttention(
                    latent_dim=embed_dim,
                    kv_dim=encoder_dim,
                    cross_attn_dim=self.cross_attn_dim,
                    num_heads=num_heads,
                    dropout=dropout,
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

        # Per-point memory cross-attention: each query token attends to its OWN point's
        # small bank of context-frame memory vectors (see MemoryEncoder). The bank has
        # M+1 keys (M context entries + a learned null), independent of clip length, so
        # this block is much cheaper than the scene cross-attention.
        self.use_memory = use_memory
        if self.use_memory:
            self.memory_cross_attns = nn.ModuleList([
                DecoupledCrossAttention(embed_dim, embed_dim, self.cross_attn_dim,
                                        num_heads, dropout)
                for _ in range(num_layers)
            ])
            self.norm_mems = nn.ModuleList([AdaptiveLayerNorm(embed_dim)
                                            for _ in range(num_layers)])
            # Zero-init the output projections so the memory path starts as an exact
            # no-op: a checkpoint warm-started into a memory-enabled model reproduces its
            # predictions until this learns to use the bank.
            for mca in self.memory_cross_attns:
                nn.init.zeros_(mca.out_proj.weight)
                nn.init.zeros_(mca.out_proj.bias)

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

        # Decoded per-token scale head(s). Zero-init (weight & bias) -> head output 0 ->
        # scale = scale_init at start (no-op when scale_init=1), so learnable_scale is a
        # clean superset of the base mode. The metric regression loss learns the scale
        # through the product; the grid CE (with the scale detached in its target) shapes
        # the normalized grid — jointly identifiable (see plan / Toy J).
        if self.learnable_scale:
            self.scale_3d_head = _make_heads(1)
            for m in range(2):
                nn.init.zeros_(self.scale_3d_head[m][1].weight)
                nn.init.zeros_(self.scale_3d_head[m][1].bias)
        else:
            self.scale_3d_head = None
        if self.learnable_scale_depth:
            self.scale_depth_head = _make_heads(1)
            for m in range(2):
                nn.init.zeros_(self.scale_depth_head[m][1].weight)
                nn.init.zeros_(self.scale_depth_head[m][1].bias)
        else:
            self.scale_depth_head = None

        if self.enable_subpixel_refinement and self.is_grid:
            # Per-bin (3*G) sub-bin offset map: one learned offset per grid value per axis,
            # soft-argmax-weighted at decode so the correction is conditioned on the predicted
            # bin (motion location), not just the features. See the decode in forward().
            self.heads_subpixel_3d = _make_heads(3 * head_3d_grid_size)
            for m in range(2):                             # zero-init -> whole offset map = 0 -> no-op at start
                nn.init.zeros_(self.heads_subpixel_3d[m][1].weight)
                nn.init.zeros_(self.heads_subpixel_3d[m][1].bias)
        else:
            self.heads_subpixel_3d = None

        if self.enable_subpixel_refinement and self.is_grid:
            # Learnable sharpening temperature (beta) for the subpixel selection. High init ->
            # weights peak at the argmax bin (~one position + one offset). Loaded fresh under
            # strict=False; absent unless refinement is on, so it never perturbs old checkpoints.
            self.log_subpixel_temp = nn.Parameter(torch.tensor(math.log(float(subpixel_temperature))))
        else:
            self.log_subpixel_temp = None

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

    def _grid_softmax(self, logits, temp=None):
        """Per-axis masked soft-argmax WEIGHTS over the bin grid (threshold 20 bins,
        temperature 0.5), mirroring tracker_tapnext.py:379-392. ``logits`` (..., K) ->
        normalized probs (..., K). Shared by the decode and the per-bin offset head.
        ``temp`` overrides the softmax temperature (used by the sharpened subpixel path)."""
        soft_argmax_threshold = self.soft_argmax_threshold
        softmax_temperature = temp if temp is not None else (
                               torch.exp(self.log_soft_argmax_temp)
                               if self.log_soft_argmax_temp is not None
                               else self._soft_argmax_temp_fixed)
        # Bound the (learnable) softmax sharpness. exp(log_temp) is unclamped, so an upward
        # drift makes softmax(logits * T) near one-hot and its gradient at a near-tied argmax
        # explodes (isolated grad-norm spikes with normal loss). Clamp the effective temp of
        # BOTH the soft-argmax path and the subpixel path (passed in via `temp`). T_MAX is a
        # few x the subpixel init (10.0) so intentional sharpness is preserved; the min is
        # cheap insurance against collapse to a near-uniform softmax. Differentiable clamp
        # keeps the parameter learnable inside the range.
        if torch.is_tensor(softmax_temperature):
            softmax_temperature = softmax_temperature.clamp(min=1e-2, max=30.0)
        K = logits.shape[-1]
        argmax = logits.argmax(dim=-1, keepdim=True)
        index = torch.arange(K, device=logits.device)
        mask = (torch.abs(argmax - index) <= soft_argmax_threshold).float()
        probs = F.softmax(logits * softmax_temperature, dim=-1) * mask
        return probs / probs.sum(dim=-1, keepdim=True)

    def _grid_decode(self, logits, grid_values):
        """Soft-argmax decode: weighted sum of bin centres. ``logits`` (..., K),
        ``grid_values`` (K,) -> decoded value (...)."""
        probs = self._grid_softmax(logits)
        return (probs * grid_values.to(probs.dtype)).sum(dim=-1)

    def _memory_read(self, layer_idx, x_normed, memory_bank, T, N_cams):
        """Each query token attends to its OWN point's memory entries.

        x_normed: ((cams b), (t k), dim) normalized decoder stream.
        memory_bank: (B, K, M, dim) -- entries with nothing to remember already carry
        MemoryEncoder's learned null token, so no validity mask is needed here.
        Returns the residual, same shape as x_normed.
        """
        bank = memory_bank.to(x_normed.dtype)
        # Each token (cam, b, t, k) reads bank[b, k]; expand over cams and frames to match
        # the flattened (cams b) (t k) query layout (t-major, so (t k) pairs with k tiled).
        keys = repeat(bank, 'b k m d -> (cams b) (t k) m d', cams=N_cams, t=T)

        # Flatten every query token into its own attention problem: 1 query, M keys.
        q = rearrange(x_normed, 'cb tk d -> (cb tk) 1 d')
        kv = rearrange(keys, 'cb tk m d -> (cb tk) m d')
        out = self.memory_cross_attns[layer_idx](q, kv)                   # (BQ, 1, dim)
        return rearrange(out[:, 0], '(cb tk) d -> cb tk d', cb=x_normed.shape[0])

    def forward(self, scene_features, query_embeds, rays, mode_idx,
                scene_frame_pos=None, memory_bank=None):
        """
        Args:
            scene_features: [N_cams, B, N_tokens, encoder_dim] from SceneRepresentation
            query_embeds: [B, T, K, N_cams, embed_dim]  T=frames, K=points
            rays: [B, T, K, N_cams, 4, 4]
            mode_idx: LongTensor of shape [1] — 0 for 2D queries, 1 for 3D queries
            scene_frame_pos: [N_tokens] frame index (in query-frame units) of each scene
                token, required when cross_attn_rope is on (the key positions for temporal
                RoPE). Ignored otherwise.
            memory_bank: [B, K, M, embed_dim] per-point context-frame memory (see
                MemoryEncoder), or None -> the memory cross-attention is skipped.

        Returns:
            dict with per-head tensors [B, T, K, N_cams, ·]: 'out_3d','out_2d','out_vis',
            'out_conf','out_depth','out_conf_3d'; 'grid_logits' (dict or None); 'latent'
            [B,T,K,N_cams,embed_dim] (the per-point decoder latent); and 'scale_3d'/
            'scale_depth' [B,T,K,N_cams,1] when learnable_scale[_depth].
        """
        B, T, K, N_cams, embed_dim = query_embeds.shape
        assert embed_dim == self.embed_dim

        kv = rearrange(scene_features, 'cams b tokens dim -> (cams b) tokens dim')
        x = rearrange(query_embeds, 'b t k cams dim -> (cams b) (t k) dim')
        rays_r = rearrange(rays, 'b t k cams d e -> (b t k) cams d e')

        # Temporal-RoPE cross-attention positions (built once, reused every layer). Query
        # tokens flatten t-major as (t k), so token (t,k) has frame t -> arange(T) each
        # repeated K times. Keys use the per-scene-token frame passed in.
        if self.cross_attn_rope:
            assert scene_frame_pos is not None, \
                "cross_attn_rope=True requires scene_frame_pos (scene-token frame indices)"
            q_pos = torch.arange(T, device=x.device).repeat_interleave(K)  # (T*K,)
            k_pos = scene_frame_pos.to(x.device)                           # (N_tokens,)

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
            if self.cross_attn_rope:
                attn_out = self.cross_attns[layer_idx](x_normed, kv, q_pos, k_pos)
            else:
                attn_out = self.cross_attns[layer_idx](x_normed, kv)
            x = x + attn_out

            # Per-point memory: what this point looked like at the context frames.
            # (scene cross-attn above = where it is now; this = what it has been.)
            if self.use_memory and memory_bank is not None:
                x = x + self._memory_read(
                    layer_idx, self.norm_mems[layer_idx](x, mode_idx),
                    memory_bank, T, N_cams)

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

            if self.heads_subpixel_3d is not None:
                # SHARPENED ARGMAX refinement: a learnable high temperature peaks the per-bin weights
                # toward one-hot at the argmax bin, so the decode approaches centre[argmax] +
                # offset[argmax] (~one position + one offset) while staying fully differentiable. Both
                # the base position and the offset use these sharp weights.
                w = self._grid_softmax(logits_3d, temp=torch.exp(self.log_subpixel_temp))  # (..., 3, G) sharp
                offsets = rearrange(self.heads_subpixel_3d[m](x), '... (d k) -> ... d k', d=3, k=G)
                offset_sel = (w * offsets).sum(-1) * self.subpixel_scale * self.subpixel_bin_width  # bins
                if self.log_3d_output:
                    # Select centre + add offset in WARPED (uniform-bin) space; clamp bounds expm1.
                    cr = self.log_3d_c_range
                    warped_centre = signed_log1p(self.grid_1d.to(w.dtype), self.log_3d_eps)
                    warped_sel = (w * warped_centre).sum(-1)                # (..., 3) ~ warped centre[argmax]
                    out_3d = signed_expm1((warped_sel + offset_sel).clamp(-cr, cr), self.log_3d_eps).to(w.dtype)
                else:
                    out_3d = (w * self.grid_1d.to(w.dtype)).sum(-1) + offset_sel  # (..., 3) head-space
            else:
                p3d = self._grid_softmax(logits_3d)                        # (..., 3, G) soft-argmax weights
                if self.log_3d_output and self.grid_decode_space == "warped":
                    # Average in the uniform WARPED space, then expm1 — overshoot-free (see __init__).
                    # grid_1d == signed_expm1(linspace(-cr,cr,G)), so signed_log1p recovers the warped
                    # centres exactly; no extra buffer, load-compatible with "head" checkpoints.
                    cr = self.log_3d_c_range
                    warped_centres = signed_log1p(self.grid_1d.to(p3d.dtype), self.log_3d_eps)
                    warped_mean = (p3d * warped_centres).sum(-1).clamp(-cr, cr)  # (..., 3) warped units
                    out_3d = signed_expm1(warped_mean, self.log_3d_eps).to(p3d.dtype)
                else:
                    out_3d = (p3d * self.grid_1d.to(p3d.dtype)).sum(-1)    # (..., 3) head-space decode (default)

            logits_depth = self.heads_depth[m](x)                          # (..., G)
            # grid/gridresid: decode log-depth then exp -> normalized depth (>0);
            # tracker_encoder multiplies by cube_scale [* f_eff] and must NOT softplus this.
            # gridnorm: decode a LINEAR gauge value (affine-solved to metric downstream).
            _depth_dec = self._grid_decode(logits_depth, self.depth_grid)
            out_depth = (_depth_dec if self.is_gridnorm else torch.exp(_depth_dec))[..., None]

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

        def _to_btkc(t):
            return rearrange(t, '(cams b) (t k) d -> b t k cams d',
                             cams=N_cams, b=B, t=T, k=K)

        # Return a dict of named per-head tensors (b,t,k,cams,·) instead of a packed cat +
        # positional split. `grid_logits` is None for non-grid modes; scale heads present
        # only when learnable_scale[_depth].
        dec = {
            'out_3d': _to_btkc(out_3d),          # (b,t,k,cams,3)
            'out_2d': _to_btkc(out_2d),          # (b,t,k,cams,2)
            'out_vis': _to_btkc(out_vis),        # (b,t,k,cams,1)
            'out_conf': _to_btkc(out_conf),      # (b,t,k,cams,1)
            'out_depth': _to_btkc(out_depth),    # (b,t,k,cams,1)
            'out_conf_3d': _to_btkc(out_conf_3d),  # (b,t,k,cams,1)
            'grid_logits': grid_logits,
            'latent': _to_btkc(x),               # (b,t,k,cams,embed_dim)
        }
        if self.learnable_scale:
            s3d = self.scale_init * torch.exp(
                self.scale_3d_head[m](x).clamp(-self.scale_delta, self.scale_delta))
            dec['scale_3d'] = _to_btkc(s3d)      # (b,t,k,cams,1)
        if self.learnable_scale_depth:
            sdep = self.scale_init * torch.exp(
                self.scale_depth_head[m](x).clamp(-self.scale_delta, self.scale_delta))
            dec['scale_depth'] = _to_btkc(sdep)  # (b,t,k,cams,1)
        return dec


class AttentionPooling(nn.Module):
    """Multihead attention-pooling (MAP) head over the (time, camera) set of a point.

    Pools the per-(t, cam) decoder latents of each tracked point into ONE latent per
    point: a learned seed query cross-attends over the flattened (t * cams) tokens
    (Set-Transformer PMA, Lee et al. 2019). Permutation-invariant over cameras (a
    variable 1..N_cams are present, no camera embedding) but order-aware over time via
    an optional learned per-frame embedding, linearly resampled to the actual T (mirrors
    QueryEncoder's t-embed resize / train_utils._interp_res_params).

        x: [b, t, k, cams, dim]  ->  [b, k, dim]
    """

    def __init__(self, dim, num_heads=8, n_frames=16, use_time_embedding=True,
                 mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.n_frames = n_frames
        self.use_time_embedding = use_time_embedding

        # learned seed query (the single "pooled" token that reads out the set)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        if use_time_embedding:
            self.time_embed = nn.Parameter(torch.zeros(n_frames, dim))
            nn.init.trunc_normal_(self.time_embed, std=0.02)

        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # post-attention feed-forward (rFF in PMA), residual on the pooled token
        self.norm_out = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _resize_time_embed(self, T, device, dtype):
        """[n_frames, dim] -> [T, dim] by linear resampling of the frame axis."""
        te = self.time_embed.to(dtype)
        if T == self.n_frames:
            return te
        w = te.t().unsqueeze(0)                                   # [1, dim, n_frames]
        w = F.interpolate(w, size=T, mode='linear', align_corners=False)
        return w.squeeze(0).t()                                   # [T, dim]

    def forward(self, x, key_padding_mask=None):
        """x: [b, t, k, cams, dim]; key_padding_mask: [b, k, t, cams] bool (True = drop).
        Returns [b, k, dim]."""
        b, T, k, cams, dim = x.shape

        if self.use_time_embedding:
            te = self._resize_time_embed(T, x.device, x.dtype)   # [T, dim]
            x = x + te[None, :, None, None, :]

        kv = rearrange(x, 'b t k cams d -> (b k) (t cams) d')
        kv = self.norm_kv(kv)

        kpm = None
        if key_padding_mask is not None:
            kpm = rearrange(key_padding_mask, 'b k t cams -> (b k) (t cams)')

        q = self.query.expand(kv.shape[0], -1, -1)               # [(b k), 1, dim]
        pooled, _ = self.attn(q, kv, kv, key_padding_mask=kpm, need_weights=False)
        pooled = pooled + self.ff(self.norm_out(pooled))         # [(b k), 1, dim]
        return rearrange(pooled[:, 0], '(b k) d -> b k d', b=b, k=k)


class MemoryEncoder(nn.Module):
    """Builds a small per-point appearance memory bank from a few CONTEXT frames.

    Motivation: the decoder reads the scene per frame, which tells it *where a point is
    now* but carries no history of *what it looked like before*. This module encodes a
    handful of context frames sampled from anywhere in the clip (non-causal) into ONE
    vector per (point, context frame); the decoder then cross-attends to that bank
    (``Decoder.memory_cross_attns``) for identity / appearance history.

    Pipeline, per context frame m:
      1. Encode the frame ONCE per camera with the (shared, usually frozen) scene ViT --
         one encode serves every point.
      2. Seed a per-point query with the existing ``QueryEncoder``, evaluated at the
         point's position *at frame m* (image patch + pixel position + depth + visibility
         + occlusion). Cheap and reuses trained machinery -- no new query encoder.
      3. Cross-attend that seed to the frame's ViT tokens -> one vector per (point, cam).
      4. Pool over cameras into a SINGLE vector per point (permutation-invariant), so the
         memory is viewpoint-robust rather than tied to one camera.
      5. Add a Fourier embedding of the context frame index so the decoder can tell the
         entries apart / know how far away in time each one is.

    Returns ``memory_bank [B, N, M, dim]``. An entry whose point was out of frame in
    EVERY camera at that context frame carries no information, so it is replaced by a
    learned ``null_entry`` -- a uniform "nothing remembered here" token. That keeps the
    bank a single dense tensor (no companion mask) and keeps attention well-defined even
    when every entry is invalid.
    """

    def __init__(self, scene_encoder, query_encoder, dim, num_heads=8,
                 cross_attn_dim=None, max_freq=10, dropout=0.0):
        super().__init__()
        # Shared references -- NOT submodules we own. Assigning a module that is already a
        # child of the parent model would double-register its parameters (breaking the
        # optimizer's param grouping and the checkpoint), so hide them from nn.Module's
        # registration via object.__setattr__ and rely on the parent for ownership.
        object.__setattr__(self, 'scene_encoder', scene_encoder)
        object.__setattr__(self, 'query_encoder', query_encoder)

        self.dim = dim
        self.max_freq = max_freq
        cross_attn_dim = cross_attn_dim or dim

        # Per-point read of the context frame's ViT tokens. NOT zero-init: this is the only
        # path by which appearance enters a memory vector.
        self.read_norm_q = nn.LayerNorm(dim)
        self.read_attn = DecoupledCrossAttention(
            latent_dim=dim, kv_dim=scene_encoder.embed_dim,
            cross_attn_dim=cross_attn_dim, num_heads=num_heads, dropout=dropout)

        # Cross-camera fusion (Set-Transformer PMA): permutation-invariant over a variable
        # number of visible cameras. t=1 here -- we pool over cameras only, not time.
        self.camera_pool = AttentionPooling(dim, num_heads=num_heads,
                                            use_time_embedding=False)

        # Absolute context-frame index -> embedding, added to the pooled vector so each
        # bank entry is tagged with its frame. Fixed normalizer (matches QueryEncoder's
        # _fourier_time) so a given frame index maps identically at any clip length.
        self.time_norm = float(getattr(query_encoder, 'time_norm', 16.0))
        self.temporal_embed = nn.Linear(2 * max_freq + 1, dim)

        # Substituted for entries whose point is out of frame in every camera: a learned
        # "nothing remembered at this frame" token, so the bank needs no validity mask.
        self.null_entry = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.null_entry, std=0.02)

    def _frame_embed(self, ctx_idx):
        """ctx_idx: (B, M) int -> (B, M, dim)."""
        s = (ctx_idx.to(torch.float32) / self.time_norm)[..., None, None]   # (B, M, 1, 1)
        feat = torch.cat([s, get_fourier_encoding(s, min_freq=0, max_freq=self.max_freq)],
                         dim=-1)
        return self.temporal_embed(feat)[..., 0, :]                         # (B, M, dim)

    def _encode_context(self, views_norm, ctx_idx):
        """ViT-encode the context frames once per (camera, context frame).

        Returns [n_cams, B*M, n_tokens, encoder_dim].

        NOTE the tubelet: the scene encoder consumes videos with tubelet_size=2, and
        SceneRepresentation.forward computes its positional grid as T // tubelet_size --
        a single frame (T=1) would give gT=0 and corrupt the pos-embed. So each context
        frame is duplicated into a 2-frame clip, yielding gT=1 and reusing the whole
        SceneRepresentation path (pos-embed, hierarchical levels, kv_proj/kv_norm)
        unchanged, so memory tokens live in the same KV space the decoder already reads.
        """
        B, M = ctx_idx.shape
        bidx = torch.arange(B, device=ctx_idx.device)[:, None].expand(B, M)
        clips = []
        for view in views_norm:                      # view: [B, T, C, H, W]
            ctx = view[bidx, ctx_idx]                # [B, M, C, H, W]
            ctx = rearrange(ctx, 'b m c h w -> (b m) c h w')
            clips.append(repeat(ctx, 'bm c h w -> bm t c h w', t=2))   # duplicate-to-2
        return self.scene_encoder(clips)             # [cams, (B M), tokens, enc_dim]

    def forward(self, views_norm, camera_group, coords_traj, ctx_idx,
                cube_scale, occlusion_ctx=None, visible_ctx=None):
        """
        Args:
            views_norm: list of [B, T, C, H, W] normalized clips (one per camera).
            camera_group: list of camera dicts.
            coords_traj: [B, T, N, R] the points' positions at EVERY frame (R=3 world or
                R=2 pixel). Ground truth at training; predicted trajectory at inference.
                A moving point must be localized at its position *at each context frame*,
                which is why the full trajectory (not the query position) is required.
            ctx_idx: [B, M] absolute frame indices of the context frames.
            cube_scale: [n_cams, B] scene scale, as in TrackerEncoder.
            occlusion_ctx: [B, M, N, n_cams] per-camera occlusion state
                {occluded=0, visible=1, unknown=-1} at each context frame, or None
                (all-unknown). Cameras that are KNOWN occluded are excluded from the
                camera pool: their image patch shows the occluder, not the point, so
                folding them in would poison the appearance memory. Unknown (-1) is kept.
            visible_ctx: [B, M, N, n_cams] bool, True where the point is geometrically in
                frame in that camera at that context frame. None -> computed from the
                projection.

        Returns:
            memory_bank: [B, N, M, dim] -- entries with no valid camera are the learned
            null token.
        """
        B, M = ctx_idx.shape
        N = coords_traj.shape[2]
        R = coords_traj.shape[3]
        n_cams = len(views_norm)
        device = coords_traj.device

        # --- 1. scene tokens for each context frame (one encode per (cam, ctx)) ---
        mem_tokens = self._encode_context(views_norm, ctx_idx)      # [cams,(B M),tok,enc]

        # --- 2. per-point seed at the point's position IN that context frame ---
        bidx = torch.arange(B, device=device)[:, None].expand(B, M)
        ctx_coords = coords_traj[bidx, ctx_idx]                     # [B, M, N, R]
        ctx_coords_flat = rearrange(ctx_coords, 'b m n r -> b (m n) r')
        # query_time == target_time == the context frame: QueryEncoder samples its image
        # patch at `query_time`, so this reads the point's appearance AT frame m (gap 0).
        ctx_t_flat = repeat(ctx_idx, 'b m -> b (m n)', n=N).to(torch.int32)
        occ_flat = (rearrange(occlusion_ctx, 'b m n c -> b (m n) c')
                    if occlusion_ctx is not None else None)

        seed = self.query_encoder(
            views_norm, camera_group, ctx_coords_flat, ctx_t_flat, ctx_t_flat,
            cube_scale, occlusion=occ_flat)                         # [B,(M N),cams,dim]
        seed = rearrange(seed, 'b (m n) cams d -> cams (b m) n d', m=M, n=N)

        # --- 3. read the frame's tokens, per point ---
        q = rearrange(self.read_norm_q(seed), 'cams bm n d -> (cams bm) n d')
        kv = rearrange(mem_tokens, 'cams bm tok d -> (cams bm) tok d')
        read = self.read_attn(q, kv)                                # [(cams B M), N, dim]
        read = rearrange(read, '(cams bm) n d -> cams bm n d', cams=n_cams)
        mem_cam = seed + read                                       # residual on the seed

        # --- 4. which cameras actually SHOW the point at this context frame ---
        # Geometrically in frame, and not known to be occluded there: an occluded view's
        # patch is the occluder's appearance, which would corrupt the pooled memory.
        if visible_ctx is None:
            visible_ctx = self._compute_visible(camera_group, ctx_coords, R)
        if occlusion_ctx is not None:
            visible_ctx = visible_ctx & (occlusion_ctx != 0)
        vis_c = rearrange(visible_ctx, 'b m n cams -> (b m) n cams')   # [(B M), N, cams]

        # --- 5. fuse across cameras -> one vector per (point, ctx) ---
        x = rearrange(mem_cam, 'cams bm n d -> bm 1 n cams d')      # t=1: pool cams only
        kpm = rearrange(~vis_c, 'bm n cams -> bm n 1 cams')         # True = drop
        # A point out of frame in EVERY camera would give an all-masked attention row
        # (NaN). Unmask it so the pool stays finite; the entry is replaced by null_entry
        # below anyway.
        all_masked = kpm.all(dim=-1, keepdim=True)
        kpm = kpm & ~all_masked
        fused = self.camera_pool(x, key_padding_mask=kpm)           # [(B M), N, dim]
        fused = rearrange(fused, '(b m) n d -> b m n d', b=B, m=M)

        # --- 6. tag each entry with its frame, null out the empty ones ---
        fused = fused + self._frame_embed(ctx_idx)[:, :, None, :]
        valid = rearrange(vis_c.any(dim=-1), '(b m) n -> b m n', b=B, m=M)
        fused = torch.where(valid[..., None],
                            fused,
                            self.null_entry.to(fused.dtype).expand_as(fused))
        return rearrange(fused, 'b m n d -> b n m d')

    @staticmethod
    def _compute_visible(camera_group, ctx_coords, R):
        """[B,M,N,R] -> [B,M,N,n_cams] bool: point geometrically inside the frame."""
        B, M, N, _ = ctx_coords.shape
        flat = rearrange(ctx_coords, 'b m n r -> (b m n) r')
        if R == 3:
            vis = torch.stack([is_point_visible(cam, flat, margin=2)
                               for cam in camera_group])            # [cams, (B M N)]
            vis = rearrange(vis, 'cams (b m n) -> b m n cams', b=B, m=M, n=N)
        else:
            # 2D: single camera, coords are already pixels -- in-frame test against size.
            cam = camera_group[0]
            size = cam['size'].to(flat.dtype).to(flat.device)
            inside = ((flat >= 0) & (flat < size)).all(dim=-1)
            vis = rearrange(inside, '(b m n) -> b m n 1', b=B, m=M, n=N)
        return vis & torch.isfinite(ctx_coords).all(dim=-1, keepdim=True)
