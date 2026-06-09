"""TrackerTapNext: native TAPNext++ multi-camera 3D point tracker.

This mirrors ``TrackerEncoder`` (``tracker_encoder.py``): same
``forward(views, coords, camera_group, query_times) -> result_dict`` interface,
same result-dict keys, same 3D-lifting/fusion tail — so the dataset, training
loop, losses, and eval all work unchanged. The difference is the trunk: instead
of a frozen V-JEPA encoder + cross-attention decoder, we run DeepMind's
pretrained TAPNext++ recurrent TRecViT backbone (re-implemented natively in
``tapnext.py``) and interleave cross-camera PROPE attention on the point tokens
to lift 2D tracks into a consistent 3D prediction.

Key properties:
  * Causal in time (RG-LRU SSM): a point queried at frame ``t`` is tracked
    forward only; pre-query frames are ``unknown`` and unsupervised (handled by
    the dataset's ``causal_masking`` flag NaN-ing pre-query targets).
  * PROPE ``CameraSelfAttention`` is zero-initialized → at init the model is
    bit-identical to per-camera TAPNext, preserving pretrained behavior.
  * Input video must be in [-1, 1] (NOT ImageNet-normalized).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, einsum, repeat

from posetail.posetail.cube import get_camera_scale, from_homogeneous, to_homogeneous
from posetail.posetail.cube import undistort_points, triangulate_simple_batch, project_points_torch
from posetail.posetail.cube import points_to_rays, _invert_SE3
from posetail.posetail.cube import CameraSelfAttention
from posetail.posetail.utils import PadToSize, count_parameters
from posetail.posetail.tapnext import TapNextBackbone


class TrackerTapNext(nn.Module):
    """TAPNext++ backbone + cross-camera PROPE attention + 3D geometry tail."""

    def __init__(self,
                 image_size=256,
                 stride_length=16,
                 # TAPNext backbone knobs
                 width=768,
                 patch_size=8,
                 num_heads=12,
                 lru_width=768,
                 depth=12,
                 prope_insert_positions=(2, 5, 8, 11),
                 pretrained_ckpt_path=None,
                 freeze_backbone=False,
                 use_checkpointing=False,
                 # geometry knobs (shared with TrackerEncoder semantics)
                 per_camera_cube_scale=False,
                 metric_ray_translation=False,
                 f_eff_scale=False,
                 mode_3d='tapnext',
                 **_ignored):
        super().__init__()

        self.mode_3d = mode_3d
        self.image_size = image_size
        self.S = stride_length
        self.n_frames = stride_length
        self.width = width
        self.num_heads = num_heads
        self.use_checkpointing = use_checkpointing

        self.per_camera_cube_scale = per_camera_cube_scale
        self.metric_ray_translation = metric_ray_translation
        self.f_eff_scale = f_eff_scale

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.prope_insert_positions = set(int(p) for p in prope_insert_positions)

        # backbone freeze: bool (whole-run) or int (unfreeze iteration), matching
        # TrackerEncoder's video_encoder_requires_grad convention.
        if isinstance(freeze_backbone, bool):
            self.backbone_unfreeze_iter = None
            self.backbone_frozen = freeze_backbone
        else:
            self.backbone_unfreeze_iter = int(freeze_backbone)
            self.backbone_frozen = True

        # [-1, 1] normalization (NOT ImageNet). Pad to the 256 canvas first so the
        # padded border becomes -1 (black) after the affine map.
        self.pad = PadToSize(self.image_size)

        # --- TAPNext backbone (native re-implementation, upstream-matching names) ---
        self.backbone = TapNextBackbone(
            image_size=(self.image_size, self.image_size),
            width=width,
            patch_size=self.patch_size,
            num_heads=num_heads,
            lru_width=lru_width,
            depth=depth,
        )
        self.grid_h = self.backbone.grid_height
        self.grid_w = self.backbone.grid_width
        self.HW = self.grid_h * self.grid_w

        if pretrained_ckpt_path is not None:
            self._load_pretrained(pretrained_ckpt_path)

        if self.backbone_frozen:
            self._set_backbone_requires_grad(False)

        # --- cross-camera PROPE attention (new, zero-init -> no-op at init) ---
        self.camera_attns = nn.ModuleList([
            CameraSelfAttention(embed_dim=width, num_heads=num_heads)
            for _ in range(len(self.prope_insert_positions))
        ])

        # --- new 3D heads (trained from scratch), mirroring Decoder head style ---
        def _head(out_dim):
            return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_dim))

        self.head_3d_direct = _head(3)
        self.head_depth = _head(1)
        self.head_conf2d = _head(1)
        self.head_conf3d = _head(1)

        # Variance-matched, dimension-invariant head init (see encoder_decoder.py).
        HEAD_OUT_STD_REG = 0.01
        HEAD_OUT_STD_LOGIT = 0.25
        reg_std = HEAD_OUT_STD_REG / (width ** 0.5)
        logit_std = HEAD_OUT_STD_LOGIT / (width ** 0.5)
        for head in [self.head_3d_direct[1], self.head_depth[1]]:
            nn.init.normal_(head.weight, std=reg_std)
            nn.init.zeros_(head.bias)
        for head in [self.head_conf2d[1], self.head_conf3d[1]]:
            nn.init.normal_(head.weight, mean=0.0, std=logit_std)
            nn.init.zeros_(head.bias)

        # Learnable output scales. Absolute outputs (direct 3D, depth) regress
        # ~depth/cube_scale == f_eff; with f_eff_scale on the per-camera f_eff
        # multiply in the forward collapses the learnable residual to ~1.
        abs_scale = 1.0 if self.f_eff_scale else 1000.0
        self.scale_3d = nn.Parameter(torch.tensor([abs_scale]))
        self.scale_depth = nn.Parameter(torch.tensor([abs_scale]))

    # ------------------------------------------------------------------ utils
    def _load_pretrained(self, path):
        import os
        path = os.path.expanduser(path)
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {k.replace('tapnext.', ''): v for k, v in state.items()}
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        # The backbone is fully covered by the pretrained TAPNext++ checkpoint.
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"[TrackerTapNext] checkpoint load: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected keys")
            if missing:
                print("  missing:", missing[:8], "..." if len(missing) > 8 else "")
            if unexpected:
                print("  unexpected:", unexpected[:8], "..." if len(unexpected) > 8 else "")
        else:
            print(f"[TrackerTapNext] loaded pretrained backbone from {path}")

    def _set_backbone_requires_grad(self, requires_grad):
        for p in self.backbone.parameters():
            p.requires_grad = requires_grad

    def unfreeze_video_encoder(self, iteration):
        """Unfreeze the TAPNext backbone once `iteration` reaches the configured
        switch-on point. Safe to call every iteration (no-op otherwise).
        Named to match TrackerEncoder so train.py's maybe_unfreeze call works."""
        if self.backbone_unfreeze_iter is None:
            return False
        if not self.backbone_frozen:
            return False
        if iteration < self.backbone_unfreeze_iter:
            return False
        self._set_backbone_requires_grad(True)
        self.backbone_frozen = False
        return True

    def print_summary(self):
        print("Hey! PARAMETERS (TrackerTapNext)")
        print("  total parameters: {:,d}".format(count_parameters(self)))
        print("  backbone params: {:,d}".format(count_parameters(self.backbone)))
        print("  camera-attn params: {:,d}".format(count_parameters(self.camera_attns)))
        print("  PROPE insert positions: {}".format(sorted(self.prope_insert_positions)))

    # ---------------------------------------------------------------- forward
    def forward(self, views, coords, camera_group, query_times=None):
        device = coords.device

        B, N, R = coords.shape
        _, T, H, W, C = views[0].shape
        n_cams = len(views)

        assert len(views) == len(camera_group), "views should match number of cameras"
        if R == 2:
            assert len(views) == 1, "should only have 1 view for 2d input"

        # ----- geometry scales (mirrors TrackerEncoder) -----
        if R == 3:
            cube_scale = get_camera_scale(camera_group, coords)  # (n_cams, B)
        else:
            cube_scale = torch.ones((n_cams, B), device=device)
        if not self.per_camera_cube_scale:
            med = torch.median(cube_scale, dim=0).values
            cube_scale = med[None, :].expand(n_cams, B).contiguous()

        if self.f_eff_scale:
            if R == 3:
                f_eff = torch.stack([
                    0.5 * (cam['mat'][0, 0] + cam['mat'][1, 1]) for cam in camera_group
                ]).to(device)
                if not self.per_camera_cube_scale:
                    f_eff = torch.full((n_cams,), torch.median(f_eff).item(), device=device)
            else:
                f_eff = torch.ones((n_cams,), device=device)

        cube_scale_shared = torch.median(cube_scale, dim=0).values  # (B,)

        if self.metric_ray_translation:
            centers_w = torch.stack([cam['center'] for cam in camera_group])
            if R == 3:
                scene_center = torch.nanmean(coords.to(torch.float32), dim=1)  # (B,3)
                dist = (centers_w[:, None, :] - scene_center[None, :, :]).norm(dim=-1)
                scene_radius = torch.median(dist, dim=0).values  # (B,)
            else:
                scene_center = centers_w[0][None].expand(B, 3)
                scene_radius = torch.ones(B, device=device)

        if query_times is None:
            query_times = torch.zeros((B, N), dtype=torch.int32, device=device)
        assert query_times.shape[0] == B and query_times.shape[1] == N

        # ----- per-camera query pixel coords (seed the point tokens) -----
        if R == 3:
            p2d_query = project_points_torch(camera_group, coords.to(torch.float32))  # (cams,B,N,2)
        else:
            p2d_query = coords.to(torch.float32)[None]  # (1,B,N,2)

        # ----- normalize frames to [-1,1] and build per-camera token streams -----
        x_cams = []
        for cam_i, frames in enumerate(views):
            frames = frames.to(device)
            frames = rearrange(frames, 'b t h w c -> b t c h w')
            frames = self.pad(frames)                      # pad H,W to image_size
            frames = 2.0 * frames - 1.0                    # [0,1] -> [-1,1], pad -> -1
            frames = rearrange(frames, 'b t c h w -> b t h w c')

            video_tokens = self.backbone.patch_embed(frames)         # (B,T,HW,c)
            qp = torch.cat([query_times[..., None].to(torch.float32),
                            p2d_query[cam_i]], dim=-1)               # (B,N,3) = (t,x,y)
            point_tokens = self.backbone.embed_queries(T, qp)        # (B,T,N,c)
            x_cams.append(torch.cat([video_tokens, point_tokens], dim=2))  # (B,T,HW+N,c)

        x = torch.stack(x_cams, dim=0)                                # (cam,B,T,HW+N,c)
        x = rearrange(x, 'cam b t n c -> (cam b) t n c')

        # ----- PROPE viewmats: per-camera query rays (constant over T) -----
        query_rays_per_cam = []
        for i in range(n_cams):
            rays_per_b = []
            for b in range(B):
                p2d_ib = p2d_query[i, b]  # (N,2)
                if self.metric_ray_translation:
                    rays_per_b.append(points_to_rays(
                        camera_group[i], p2d_ib, cube_scale_shared[b],
                        scene_center=scene_center[b], scene_radius=scene_radius[b]))
                else:
                    rays_per_b.append(points_to_rays(
                        camera_group[i], p2d_ib, cube_scale_shared[b]))
            query_rays_per_cam.append(torch.stack(rays_per_b, dim=0))   # (B,N,4,4)
        query_rays = torch.stack(query_rays_per_cam, dim=0)             # (cam,B,N,4,4)
        viewmats = repeat(query_rays, 'cam b q d e -> (b t q) cam d e', t=T)

        # ----- block loop with interleaved cross-camera PROPE attention -----
        use_linear_scan = not self.training
        HW = self.HW
        prope_idx = 0
        for i, blk in enumerate(self.backbone.blocks):
            if self.use_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(
                    blk, x, None, use_linear_scan, use_reentrant=False)
            else:
                x, _ = blk(x, cache=None, use_linear_scan=use_linear_scan)
            if i in self.prope_insert_positions:
                pts = x[:, :, HW:, :]                                   # (cam b) t q c
                pts = rearrange(pts, '(cam b) t q c -> (b t q) cam c', cam=n_cams, b=B)
                pts = pts + self.camera_attns[prope_idx](pts, viewmats)
                pts = rearrange(pts, '(b t q) cam c -> (cam b) t q c',
                                cam=n_cams, b=B, t=T, q=N)
                x = torch.cat([x[:, :, :HW, :], pts], dim=2)
                prope_idx += 1

        x = self.backbone.encoder_norm(x)

        # ----- split point tokens -----
        point_tokens = x[:, :, HW:, :]                                  # (cam b) t q c
        point_tokens = rearrange(point_tokens, '(cam b) t q c -> cam b t q c',
                                 cam=n_cams, b=B)

        # ----- 2D branch (pretrained heads): absolute pixel tracks -----
        tracks, _track_logits, vis_logits = self.backbone.prediction_heads(point_tokens)
        points_pred_scaled = tracks                                    # (cam,b,t,n,2) absolute px
        vis_pred_2d_logits = vis_logits                                # (cam,b,t,n,1)

        # ----- 3D branch (new heads) -----
        points_3d_raw = self.head_3d_direct(point_tokens) * self.scale_3d   # (cam,b,t,n,3)
        depth_pred = self.head_depth(point_tokens) * self.scale_depth       # (cam,b,t,n,1)
        conf_pred_2d_logits = self.head_conf2d(point_tokens)                # (cam,b,t,n,1)
        conf_3d_logits = self.head_conf3d(point_tokens)                     # (cam,b,t,n,1)

        vis_pred_2d = F.sigmoid(vis_pred_2d_logits)
        conf_pred_2d = F.sigmoid(conf_pred_2d_logits)
        conf_3d = torch.softmax(conf_3d_logits[..., 0], dim=0)              # (cam,b,t,n)

        depth_pred_scaled = F.softplus(depth_pred[..., 0]) * rearrange(cube_scale, 'cams b -> cams b 1 1')
        if self.f_eff_scale:
            depth_pred_scaled = depth_pred_scaled * rearrange(f_eff, 'cams -> cams 1 1 1').to(depth_pred_scaled.dtype)

        # ----- 3D lifting + fusion (ported from tracker_encoder.py:300-428) -----
        points_und = torch.stack([
            undistort_points(camera_group[i], points_pred_scaled[i])
            for i in range(n_cams)
        ])

        # ray-lift each camera
        rays_norm = to_homogeneous(points_und)
        rot_mats = torch.stack([cam['ext'][:3, :3] for cam in camera_group])
        rays_world = einsum(rays_norm, rot_mats, 'cams b t n r, cams r x -> cams b t n x')
        rays_world = F.normalize(rays_world, dim=-1)

        centers = torch.stack([cam['center'] for cam in camera_group])
        cadd = repeat(centers, 'cams r -> cams 1 1 1 r')
        points_3d_all_rays = cadd + einsum(rays_world, depth_pred_scaled,
                                           'cams b t n r, cams b t n -> cams b t n r')
        points_3d_rays = einsum(points_3d_all_rays, conf_pred_2d[..., 0],
                                'cams b t n r, cams b t n -> b t n r')

        # triangulate
        if n_cams > 1:
            points_und_flat = rearrange(points_und, 'cams b t n r -> cams (b t n) r')
            camera_mats = torch.stack([cam['ext'] for cam in camera_group])
            weights = rearrange(conf_pred_2d, 'cams b t n 1 -> cams (b t n)')
            points_und_flat = torch.clip(points_und_flat, -2, 2)
            points_3d_flat = triangulate_simple_batch(points_und_flat.to(torch.float32),
                                                      camera_mats.to(torch.float32),
                                                      weights.to(torch.float32)).to(points_und_flat.dtype)
            points_3d_tri = rearrange(points_3d_flat, '(b t n) r -> b t n r', b=B, t=T, n=N)
        else:
            points_3d_tri = None

        # direct ray-local predictions
        center = torch.tensor([self.image_size // 2, self.image_size // 2],
                              device=device, dtype=torch.float32).reshape(1, 2)
        rays_c = torch.stack([points_to_rays(cam, center, normalize_t=False)[0] for cam in camera_group])
        rays_c_inv = _invert_SE3(rays_c)

        p3d_cams = points_3d_raw * rearrange(cube_scale, 'cams b -> cams b 1 1 1')
        if self.f_eff_scale:
            p3d_cams = p3d_cams * rearrange(f_eff, 'cams -> cams 1 1 1 1').to(p3d_cams.dtype)

        points_3d_all_direct = from_homogeneous(
            einsum(rays_c_inv, to_homogeneous(p3d_cams),
                   'cams x r, cams b t n r -> cams b t n x')
        )
        points_3d_direct = einsum(points_3d_all_direct, conf_3d,
                                  'cams b t n r, cams b t n -> b t n r')

        vis_pred = torch.amax(vis_pred_2d_logits, dim=0)               # (b,t,n,1)
        conf_pred = torch.amax(conf_3d_logits[..., 0], dim=0)          # (b,t,n)

        result_dict = {
            'coords_pred': points_3d_direct,
            '3d_pred_cams_direct': points_3d_all_direct,
            '3d_pred_cams_rays': points_3d_all_rays,
            'conf_3d': conf_3d_logits[..., 0],
            '3d_pred_direct': points_3d_direct,
            '3d_pred_rays': points_3d_rays,
            '3d_pred_triangulate': points_3d_tri,
            '2d_pred': points_pred_scaled,
            'vis_pred': vis_pred,
            'conf_pred': conf_pred,
            'vis_pred_2d': vis_pred_2d_logits[..., 0],
            'conf_pred_2d': conf_pred_2d_logits[..., 0],
            'depth_pred': depth_pred_scaled,
        }
        return result_dict
