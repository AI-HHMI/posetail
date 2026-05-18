#!/usr/bin/env python3

import torch
import torch.nn as nn 
import torch.nn.functional as F

from posetail.posetail.networks import EmbedV2V
from posetail.posetail.cube import is_point_visible, project_points_torch
from posetail.posetail.cube import CameraSelfAttention
from posetail.posetail.utils import get_fourier_encoding

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
                 patch_size=9, use_volume_embedding=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.corr_radius = corr_radius
        self.max_freq = max_freq
        self.patch_size = patch_size
        self.n_frames = n_frames
        self.use_volume_embedding = use_volume_embedding

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

        self.n_fusion_terms = 7 if self.use_volume_embedding else 6
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

        # Time embeddings (shared)
        embed_query_time  = repeat(self.t_query_embed(query_time),
                                   'b t d -> b t cams d', cams=n_cams)
        embed_target_time = repeat(self.t_target_embed(target_time),
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
                 hierarchical_features=True, decoder_dim=None):
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

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.freeze_encoder = freeze_encoder

        self.n_frames = n_frames
        self.image_size = image_size

        n_tokens = (n_frames // self.tubelet_size) * (image_size // self.patch_size) * (image_size // self.patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_tokens, self.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02, a=-2 * 0.02, b=2 * 0.02)

        if decoder_dim is not None:
            self.kv_proj = nn.Linear(self.embed_dim, decoder_dim)
            nn.init.xavier_uniform_(self.kv_proj.weight)
            nn.init.zeros_(self.kv_proj.bias)
            self.kv_norm = nn.LayerNorm(decoder_dim)
            self.embed_dim = decoder_dim
        else:
            self.kv_proj = None
            self.kv_norm = None

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
            feat = feat + self.pos_embed
            if self.kv_proj is not None:
                feat = self.kv_proj(feat)
                feat = self.kv_norm(feat)
            encoded_list.append(feat)
        encoded = torch.stack(encoded_list)  # [cams, B, n_tokens, embed_dim]

        return encoded


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
                 output_mode="direct",
                 head_3d_grid_size=8,
                 head_3d_grid_radius=1.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers
        self.use_camera_self_attention = use_camera_self_attention
        self.output_mode = output_mode
        self.head_3d_grid_size = head_3d_grid_size
        self.head_3d_grid_radius = head_3d_grid_radius

        if output_mode == 'grid':
            grid_1d = torch.linspace(-head_3d_grid_radius, head_3d_grid_radius, head_3d_grid_size)
            grid_offsets = torch.cartesian_prod(grid_1d, grid_1d, grid_1d)  # [G**3, 3]
            self.register_buffer("grid_offsets_3d", grid_offsets)

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

        # Per-mode output heads: index 0 = 2D, index 1 = 3D
        head_3d_out = head_3d_grid_size ** 3 if output_mode == 'grid' else 3

        def _make_heads(out_dim):
            return nn.ModuleList([
                nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_dim))
                for _ in range(2)
            ])

        self.heads_3d      = _make_heads(head_3d_out)
        self.heads_2d      = _make_heads(2)
        self.heads_vis     = _make_heads(1)
        self.heads_conf    = _make_heads(1)
        self.heads_depth   = _make_heads(1)
        self.heads_conf_3d = _make_heads(1)

        # Weight init — applied to both mode heads
        for m in range(2):
            for head in [self.heads_2d[m][1], self.heads_depth[m][1]]:
                nn.init.normal_(head.weight, std=0.001)
                nn.init.zeros_(head.bias)

            if output_mode == 'grid':
                nn.init.zeros_(self.heads_3d[m][1].weight)
                nn.init.zeros_(self.heads_3d[m][1].bias)
            else:
                nn.init.normal_(self.heads_3d[m][1].weight, std=0.001)
                nn.init.zeros_(self.heads_3d[m][1].bias)

            for head in [self.heads_vis[m][1], self.heads_conf[m][1], self.heads_conf_3d[m][1]]:
                nn.init.normal_(head.weight, mean=0.0, std=0.01)
                nn.init.zeros_(head.bias)

        # Learnable output scales (shared across modes)
        if self.output_mode == 'direct':
            self.scale_3d = nn.Parameter(torch.tensor([500.0]))
            self.scale_2d = nn.Parameter(torch.tensor([128.0]))
        elif self.output_mode == 'residual':
            self.scale_3d = nn.Parameter(torch.tensor([1.0]))
            self.scale_2d = nn.Parameter(torch.tensor([1.0]))
        else:  # grid
            self.scale_3d = nn.Parameter(torch.tensor([500.0]))
            self.scale_2d = nn.Parameter(torch.tensor([128.0]))

        self.scale_depth = nn.Parameter(torch.tensor([500.0]))

    def forward(self, scene_features, query_embeds, rays, mode_idx):
        """
        Args:
            scene_features: [N_cams, B, N_tokens, encoder_dim] from SceneRepresentation
            query_embeds: [B, T_query, N_cams, embed_dim] from QueryEncoder
            rays: [B, T_query, N_cams, 4, 4]
            mode_idx: LongTensor of shape [1] — 0 for 2D queries, 1 for 3D queries
        Returns:
            outputs: [B, T_query, N_cams, 9]
        """
        B, T_query, N_cams, embed_dim = query_embeds.shape
        assert embed_dim == self.embed_dim

        kv = rearrange(scene_features, 'cams b tokens dim -> (cams b) tokens dim')
        x = rearrange(query_embeds, 'b t cams dim -> (cams b) t dim')
        rays_r = rearrange(rays, 'b t cams d e -> (b t) cams d e')

        for layer_idx in range(self.num_layers):
            if self.use_camera_self_attention:
                x_cam = rearrange(x, '(cams b) t dim -> (b t) cams dim', b=B, cams=N_cams, t=T_query)
                attn_out = self.camera_attns[layer_idx](self.norm0s[layer_idx](x_cam, mode_idx), rays_r)
                x_cam = x_cam + attn_out
                x = rearrange(x_cam, '(b t) cams dim -> (cams b) t dim', b=B, cams=N_cams, t=T_query)

            x_normed = self.norm1s[layer_idx](x, mode_idx)
            attn_out, _ = self.cross_attns[layer_idx](
                query=x_normed, key=kv, value=kv, need_weights=False
            )
            x = x + attn_out

            x = x + self.mlps[layer_idx](self.norm2s[layer_idx](x, mode_idx))

        m = mode_idx.item()
        if self.output_mode == 'grid':
            logits_3d = self.heads_3d[m](x)
            prob_3d = F.softmax(logits_3d, dim=-1)
            out_3d = (prob_3d @ self.grid_offsets_3d) * self.scale_3d
        else:
            out_3d = self.heads_3d[m](x) * self.scale_3d
        out_2d      = self.heads_2d[m](x) * self.scale_2d
        out_vis     = self.heads_vis[m](x)
        out_conf    = self.heads_conf[m](x)
        out_depth   = self.heads_depth[m](x) * self.scale_depth
        out_conf_3d = self.heads_conf_3d[m](x)
        output = torch.cat([out_3d, out_2d, out_vis, out_conf, out_depth, out_conf_3d], dim=-1)

        return rearrange(output, '(cams b) t dim -> b t cams dim', cams=N_cams, b=B)
