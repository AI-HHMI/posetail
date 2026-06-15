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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, einsum, repeat

from posetail.posetail.cube import get_camera_scale, from_homogeneous, to_homogeneous
from posetail.posetail.cube import undistort_points, triangulate_simple_batch, project_points_torch
from posetail.posetail.cube import points_to_rays, _invert_SE3
from posetail.posetail.cube import CameraSelfAttention, is_point_visible
from posetail.posetail.cube import signed_log1p, signed_expm1
from posetail.posetail.cube import noisy_or_logit
from posetail.posetail.utils import PadToSize, count_parameters, get_fourier_encoding
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
                 # query-geometry signals (same knob names/semantics as QueryEncoder)
                 max_freq=10,
                 principal_point_embedding=False,
                 intrinsic_embedding=False,
                 # 3D/depth output representation. "direct" = unbounded regression
                 # (today's behavior); "grid" = per-dimension marginal soft-argmax
                 # over `head_grid_size` bins (2D-prediction style) for BOTH the 3D
                 # head (3 dims) and the depth head (1 dim, in log-normalized space).
                 output_mode='direct',
                 head_grid_size=256,
                 head_3d_grid_radius=1.0,
                 depth_log_min=-3.0,
                 depth_log_max=3.0,
                 # `log_3d_output`: represent the 3D head's output in a signed-log
                 # compressed space (denser resolution near 0, where motion residuals
                 # live). Affects ONLY the 3D output (2D + depth untouched; depth is
                 # already log). Default False keeps the linear behavior bit-identical.
                 log_3d_output=False,
                 log_3d_eps=0.1,
                 mode_3d='tapnext',
                 **_ignored):
        super().__init__()

        self.mode_3d = mode_3d
        self.output_mode = output_mode
        # `is_grid`: the 3D/depth heads emit soft-argmax bin logits (vs direct
        # regression). `is_resid`: the 3D head predicts a motion *offset* added to
        # a per-track query anchor (ported from tracker_encoder.py's `residual`
        # mode); depth stays absolute in every mode. `gridresid` = both.
        self.is_grid = output_mode in ('grid', 'gridresid')
        self.is_resid = output_mode in ('residual', 'gridresid')
        # signed-log warp of the 3D output. `c_range = log1p(radius/eps)` is the
        # compressed half-width: the warp maps [-c_range, c_range] <-> [-radius,
        # radius]. Grid modes place bin centres in warped space; continuous modes
        # apply the warp (clamped) in the forward. See cube.signed_log1p/expm1.
        self.log_3d_output = log_3d_output
        self.log_3d_eps = log_3d_eps
        self.log_3d_c_range = float(math.log1p(head_3d_grid_radius / log_3d_eps))
        self.head_grid_size = head_grid_size
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

        # NOTE: the early freeze is applied at the END of __init__ (not here),
        # because it now also covers the PROPE attention + aux projections created
        # below. Freezing those during the early phase keeps the 2D heads' inputs
        # pristine (they read the same point tokens PROPE/aux would perturb).

        # --- cross-camera PROPE attention (new, zero-init -> no-op at init) ---
        self.camera_attns = nn.ModuleList([
            CameraSelfAttention(embed_dim=width, num_heads=num_heads)
            for _ in range(len(self.prope_insert_positions))
        ])

        # --- new 3D heads (trained from scratch), mirroring Decoder head style ---
        def _head(out_dim):
            return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_dim))

        # In "grid" mode the 3D head emits 3*G marginal-per-axis logits and the
        # depth head emits G logits over a fixed bin grid; otherwise they regress
        # 3 and 1 values directly (today's behavior). Register the bin centers:
        # 3D in normalized ray-local units [-radius, radius]; depth in log of the
        # normalized depth (distance / cube_scale [/ f_eff]) over [log_min, log_max].
        if self.is_grid:
            G = head_grid_size
            self.head_3d_direct = _head(3 * G)
            self.head_depth = _head(G)
            # 3D bin centres. With log_3d_output the centres are signed-log spaced
            # (denser near 0) but still span exactly [-radius, radius]; otherwise
            # linear. The depth grid is ALWAYS linear-in-log (its own representation),
            # never touched by log_3d_output.
            if self.log_3d_output:
                cr = self.log_3d_c_range
                grid_1d = signed_expm1(torch.linspace(-cr, cr, G), log_3d_eps)
            else:
                grid_1d = torch.linspace(-head_3d_grid_radius, head_3d_grid_radius, G)
            self.register_buffer('grid_1d', grid_1d)
            self.register_buffer('depth_grid',
                                 torch.linspace(depth_log_min, depth_log_max, G))
            self.g3d_lo, self.g3d_hi = -head_3d_grid_radius, head_3d_grid_radius
            self.gd_lo, self.gd_hi = depth_log_min, depth_log_max
        else:
            self.head_3d_direct = _head(3)
            self.head_depth = _head(1)
        self.head_conf2d = _head(1)
        self.head_conf3d = _head(1)

        # Variance-matched, dimension-invariant head init (see encoder_decoder.py).
        # Grid heads are zero-init (uniform softmax -> starts at the grid centre,
        # i.e. 0 ray-local / mid log-depth), matching encoder_decoder.py:695.
        HEAD_OUT_STD_REG = 0.01
        HEAD_OUT_STD_LOGIT = 0.25
        reg_std = HEAD_OUT_STD_REG / (width ** 0.5)
        logit_std = HEAD_OUT_STD_LOGIT / (width ** 0.5)
        if self.is_grid:
            for head in [self.head_3d_direct[1], self.head_depth[1]]:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        else:
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
        # The `residual` direct head emits a metric motion offset (motion/cube ~ a few
        # px-units, NOT f_eff-scaled -> ortho-safe); the encoder inits its residual
        # scale at 8.0. Absolute direct/grid keep abs_scale. Grid/gridresid ignore
        # scale_3d (the bins carry the magnitude); under log_3d_output the continuous
        # path also drops scale_3d (the warp's eps/clamp set slope/reach instead).
        scale_3d_init = 8.0 if output_mode == 'residual' else abs_scale
        self.scale_3d = nn.Parameter(torch.tensor([scale_3d_init]))
        self.scale_depth = nn.Parameter(torch.tensor([abs_scale]))

        # --- query-geometry aux embeddings (zero-init additive residual) ---
        # Reinstate the rich QueryEncoder query signals lost in the native
        # TAPNext token (depth/distance, per-camera frustum visibility, focal +
        # principal point). Each is Fourier+linear'd (same math as QueryEncoder,
        # encoder_decoder.py:329-395) and summed onto the query token at the
        # query frame. ALL projections are zero-initialized so the residual is
        # exactly 0 at init -> the model stays bit-identical to pretrained
        # TAPNext (mirrors the zero-init PROPE out_proj). Depth + visibility are
        # always on in 3D (as in QueryEncoder); focal/pp are flag-gated.
        self.max_freq = max_freq
        self.principal_point_embedding = principal_point_embedding
        self.intrinsic_embedding = intrinsic_embedding
        mf = max_freq
        self.q_depth_proj = nn.Linear(2 * mf + 1, width)   # depth scalar (always)
        self.q_vis_embed = nn.Embedding(2, width)          # in/out frustum (always)
        self.q_depth_norm_scale = nn.Parameter(torch.tensor([1.0]))
        if intrinsic_embedding:
            self.q_focal_proj = nn.Linear(4 * mf + 2, width)   # (fx, fy)
        if principal_point_embedding:
            self.q_pp_proj = nn.Linear(4 * mf + 2, width)      # principal point
        zero_linears = [self.q_depth_proj]
        if intrinsic_embedding:
            zero_linears.append(self.q_focal_proj)
        if principal_point_embedding:
            zero_linears.append(self.q_pp_proj)
        for m in zero_linears:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.q_vis_embed.weight)

        # Early freeze (backbone + PROPE + aux). Applied here, after every
        # early-frozen module exists. Both PROPE and the aux residual are zero-init,
        # so freezing them keeps the 2D path bit-identical to pretrained TAPNext
        # through the frozen phase while the from-scratch 3D heads warm up.
        if self.backbone_frozen:
            self._set_backbone_requires_grad(False)

    # ------------------------------------------------------------------ utils
    def _load_pretrained(self, path):
        import os
        path = os.path.expanduser(path)
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {k.replace('tapnext.', ''): v for k, v in state.items()}
        # query_pos_embed is a deterministic sincos constant recomputed in
        # TapNextBackbone (held as a plain attribute, not a buffer, so DDP does
        # not try to broadcast it); drop it from the load so it isn't flagged.
        sd.pop('query_pos_embed', None)
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

    def _early_frozen_parameters(self):
        """Params held frozen during the early phase: the pretrained backbone PLUS
        the new PROPE attention and aux-query projections. Freezing the latter two
        (both zero-init) keeps the point tokens the frozen 2D heads read pristine,
        so 2D/occlusion stay at pretrained quality until the 3D heads warm up."""
        params = list(self.backbone.parameters())
        params += list(self.camera_attns.parameters())
        aux = [self.q_depth_proj, self.q_vis_embed, self.q_depth_norm_scale]
        if self.intrinsic_embedding:
            aux.append(self.q_focal_proj)
        if self.principal_point_embedding:
            aux.append(self.q_pp_proj)
        for a in aux:
            params += list(a.parameters()) if isinstance(a, nn.Module) else [a]
        return params

    def _set_backbone_requires_grad(self, requires_grad):
        for p in self._early_frozen_parameters():
            p.requires_grad = requires_grad

    def unfreeze_video_encoder(self, iteration):
        """Unfreeze the backbone + PROPE + aux projections once `iteration` reaches
        the configured switch-on point. Safe to call every iteration (no-op
        otherwise). Named to match TrackerEncoder so train.py's maybe_unfreeze call
        works."""
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

    # ----------------------------------------------------------- aux query embed
    def _query_aux_embed(self, cam_i, camera_group, coords, cube_scale, view_hw):
        """Zero-init additive residual for one camera's query tokens -> [B,N,c].

        Mirrors QueryEncoder's depth/visibility/focal/pp math
        (encoder_decoder.py:329-395). In 3D: depth + frustum visibility (+ focal
        if ``intrinsic_embedding`` + principal point if
        ``principal_point_embedding``). In 2D (R==2): only a pixel in-bounds
        visibility check (depth/focal/pp need a real 3D point + cube scale).
        Returns exactly 0 at init because every projection is zero-initialized.
        """
        B, N, R = coords.shape
        H, W = view_hw
        mf = self.max_freq
        coords_f = coords.to(torch.float32)

        if R == 2:
            # 2D: visibility is a plain pixel-bounds check (cf. encoder_decoder.py:409).
            margin = 2
            in_bounds = ((coords_f[..., 0] >= margin) & (coords_f[..., 0] < W - margin) &
                         (coords_f[..., 1] >= margin) & (coords_f[..., 1] < H - margin))
            return self.q_vis_embed(in_bounds.to(torch.int32))      # [B,N,c]

        cam = camera_group[cam_i]
        # --- depth: log distance to camera center / per-camera cube scale ---
        center = cam['center'].to(torch.float32)                    # (3,)
        raw = (coords_f - center).norm(dim=-1) / cube_scale[cam_i][:, None]  # (B,N)
        depths = torch.log(raw + 1e-6) * self.q_depth_norm_scale    # (B,N)
        dr = depths.reshape(B, N, 1, 1)                             # 'bsnr', r=1
        fourier_depth = get_fourier_encoding(dr, min_freq=0, max_freq=mf)
        fourier_depth = torch.cat([dr, fourier_depth], dim=-1)      # (B,N,1,2mf+1)
        extra = self.q_depth_proj(fourier_depth).squeeze(2)        # [B,N,c]

        # --- visibility: frustum + in-front test ---
        visible = is_point_visible(cam, coords_f.reshape(B * N, 3), margin=2)
        extra = extra + self.q_vis_embed(visible.reshape(B, N).to(torch.int32))

        # --- focal length (fx, fy), normalized by the canvas the net sees ---
        if self.intrinsic_embedding:
            focal = torch.stack([cam['mat'][0, 0], cam['mat'][1, 1]]).to(torch.float32)
            focal = focal / self.image_size                         # (2,)
            fn = focal.reshape(1, 1, 1, 2).expand(B, N, 1, 2)       # 'bsnr', r=2
            fourier_f = get_fourier_encoding(fn, min_freq=0, max_freq=mf)
            fourier_f = torch.cat([fn, fourier_f], dim=-1)          # (B,N,1,4mf+2)
            extra = extra + self.q_focal_proj(fourier_f).squeeze(2)

        # --- principal point, normalized to [-1, 1] ---
        if self.principal_point_embedding:
            pp = (cam['mat'][:2, 2] - cam['offset']).to(torch.float32)
            pp = pp / self.image_size * 2.0 - 1.0                   # (2,)
            pn = pp.reshape(1, 1, 1, 2).expand(B, N, 1, 2)
            fourier_pp = get_fourier_encoding(pn, min_freq=0, max_freq=mf)
            fourier_pp = torch.cat([pn, fourier_pp], dim=-1)        # (B,N,1,4mf+2)
            extra = extra + self.q_pp_proj(fourier_pp).squeeze(2)

        return extra

    # ------------------------------------------------------------- grid decode
    def _grid_decode(self, logits, grid_values):
        """Per-axis masked soft-argmax over a fixed grid of bin centres, mirroring
        ``TapNextBackbone.prediction_heads`` (threshold 20 bins, temperature 0.5).
        ``logits`` (..., K), ``grid_values`` (K,) -> decoded value (...,). Used for
        both the grid 3D head (called per-axis) and the grid depth head."""
        soft_argmax_threshold = 20
        softmax_temperature = 0.5
        K = logits.shape[-1]
        argmax = logits.argmax(dim=-1, keepdim=True)
        index = torch.arange(K, device=logits.device)
        mask = (torch.abs(argmax - index) <= soft_argmax_threshold).float()
        probs = F.softmax(logits * softmax_temperature, dim=-1) * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return (probs * grid_values).sum(dim=-1)

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
            extra = self._query_aux_embed(cam_i, camera_group, coords,
                                          cube_scale, (H, W))         # (B,N,c) zero at init
            point_tokens = self.backbone.embed_queries(
                T, qp, extra_query_embed=extra)                       # (B,T,N,c)
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
        tracks, track_logits, vis_logits = self.backbone.prediction_heads(point_tokens)
        points_pred_scaled = tracks                                    # (cam,b,t,n,2) absolute px
        vis_pred_2d_logits = vis_logits                                # (cam,b,t,n,1)

        # ----- 3D branch (new heads) -----
        # `points_3d_raw` is the normalized ray-local 3D point and `depth_norm` the
        # normalized depth (distance / cube_scale [/ f_eff]); both get scaled back to
        # metric below. In grid mode they are masked soft-argmax over a fixed bin grid
        # (bounded, self-balancing); in direct mode they are unbounded regressions.
        logits_3d = logits_depth = None
        if self.is_grid:
            logits_3d = rearrange(self.head_3d_direct(point_tokens),
                                  'cam b t n (d k) -> cam b t n d k', d=3, k=self.head_grid_size)
            points_3d_raw = self._grid_decode(logits_3d, self.grid_1d)       # (cam,b,t,n,3)
            logits_depth = self.head_depth(point_tokens)                     # (cam,b,t,n,K)
            depth_norm = torch.exp(self._grid_decode(logits_depth, self.depth_grid))  # (cam,b,t,n)
        else:
            if self.log_3d_output:
                # Continuous (direct/residual) warp: the head predicts a compressed
                # coordinate; the value = signed_expm1(.). Clamp to c_range + fp32
                # keep expm1 (and its gradient) bounded; scale_3d is dropped (eps sets
                # the slope, the clamp the reach). Init head ~0 -> output 0 (parity).
                cr = self.log_3d_c_range
                head_out = self.head_3d_direct(point_tokens).clamp(-cr, cr)
                points_3d_raw = signed_expm1(head_out, self.log_3d_eps).to(point_tokens.dtype)
            else:
                points_3d_raw = self.head_3d_direct(point_tokens) * self.scale_3d  # (cam,b,t,n,3)
            depth_pred = self.head_depth(point_tokens) * self.scale_depth       # (cam,b,t,n,1)
            depth_norm = F.softplus(depth_pred[..., 0])                         # (cam,b,t,n)
        conf_pred_2d_logits = self.head_conf2d(point_tokens)                # (cam,b,t,n,1)
        conf_3d_logits = self.head_conf3d(point_tokens)                     # (cam,b,t,n,1)

        vis_pred_2d = F.sigmoid(vis_pred_2d_logits)
        conf_pred_2d = F.sigmoid(conf_pred_2d_logits)
        conf_3d = torch.softmax(conf_3d_logits[..., 0], dim=0)              # (cam,b,t,n)

        depth_pred_scaled = depth_norm * rearrange(cube_scale, 'cams b -> cams b 1 1')
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
        if self.f_eff_scale and not self.is_resid:
            # f_eff-scale ABSOLUTE outputs only (direct/grid): position/(cube*f_eff)
            # is a dimensionless depth ratio -> O(1) and ortho-safe. A residual is
            # NOT f_eff-scaled (f_eff is the wrong normalizer for motion and breaks
            # ortho cameras) — matches tracker_encoder.py's `not add_residual` gate.
            p3d_cams = p3d_cams * rearrange(f_eff, 'cams -> cams 1 1 1 1').to(p3d_cams.dtype)
        if self.output_mode == 'gridresid':
            # The gridresid bins encode motion / (cube * image_size) (~pixels, in
            # [0,1], camera-invariant). Map back to metric motion by * image_size.
            p3d_cams = p3d_cams * self.image_size

        # residual modes: add the per-track query anchor in metric ray-local space.
        # R==3 anchor = the query's GT world coords; R==2 anchor = the model's own
        # ray-fused 3D prediction at the query frame (gather over points_3d_rays).
        query_local = None
        if self.is_resid:
            if R == 3:
                query_world = repeat(coords.to(torch.float32),
                                     'b n r -> cams b t n r', cams=n_cams, t=T)
            else:
                t_idx = repeat(query_times.long(), 'b n -> b 1 n r', r=3)
                query_3d = torch.gather(points_3d_rays, dim=1, index=t_idx)  # (b,1,n,3)
                query_world = repeat(query_3d, 'b 1 n r -> cams b t n r', t=T, cams=n_cams)
            query_local = from_homogeneous(
                einsum(rays_c, to_homogeneous(query_world),
                       'cams x r, cams b t n r -> cams b t n x'))
            p3d_cams = p3d_cams + query_local

        points_3d_all_direct = from_homogeneous(
            einsum(rays_c_inv, to_homogeneous(p3d_cams),
                   'cams x r, cams b t n r -> cams b t n x')
        )
        points_3d_direct = einsum(points_3d_all_direct, conf_3d,
                                  'cams b t n r, cams b t n -> b t n r')

        vis_pred = noisy_or_logit(vis_pred_2d_logits, dim=0)           # (b,t,n,1) noisy-OR logit
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
            # Raw 2D position logits (256 x-bins | 256 y-bins) for direct
            # coordinate-softmax supervision (upstream TAPNext++'s primary 2D loss).
            '2d_logits': track_logits,                                  # (cam,b,t,n,512)
        }
        # In grid mode, hand the loss the raw 3D/depth bin logits plus the geometry
        # it needs to discretize GT into bin targets (it never sees the model): the
        # per-camera ray-local SE3, cube_scale, f_eff, and the grid ranges.
        if self.is_grid:
            result_dict['grid'] = {
                'logits_3d': logits_3d,            # (cam,b,t,n,3,K)
                'logits_depth': logits_depth,      # (cam,b,t,n,K)
                'rays_c': rays_c,                  # (cams,4,4) world -> ray-local
                'cube_scale': cube_scale,          # (cams,B)
                'f_eff': f_eff if self.f_eff_scale else None,  # (cams,) or None
                'g3d_lo': self.g3d_lo, 'g3d_hi': self.g3d_hi,
                'gd_lo': self.gd_lo, 'gd_hi': self.gd_hi,
                'K': self.head_grid_size,
                # gridresid: the 3D grid bins encode a residual offset, so the loss
                # forms its bin target as (GT_raylocal - anchor)/(cube*image_size)
                # (no f_eff; ortho-safe). Absolute `grid` mode passes is_resid=False,
                # no anchor, and divides by cube*f_eff instead.
                'is_resid': self.is_resid,
                'anchor_local': query_local.detach() if query_local is not None else None,
                'image_size': self.image_size,
                # log_3d_output: the loss must quantize the 3D bin target in the same
                # signed-log warped space as the (warped) bin centres.
                'log_warp': self.log_3d_output,
                'eps': self.log_3d_eps,
                'c_range': self.log_3d_c_range,
            }
        return result_dict
