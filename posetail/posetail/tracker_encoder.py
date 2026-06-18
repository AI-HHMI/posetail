import itertools
# import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange, einsum, reduce, repeat

from posetail.posetail.cube import get_camera_scale, from_homogeneous, to_homogeneous
from posetail.posetail.cube import undistort_points, triangulate_simple_batch, project_points_torch
from posetail.posetail.cube import points_to_rays, _invert_SE3
from posetail.posetail.cube import noisy_or_logit
from posetail.posetail.utils import PadToMultiple, PadToSize, count_parameters
from posetail.posetail.encoder_decoder import SceneRepresentation, QueryEncoder, Decoder

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class TrackerEncoder(nn.Module): 

    def __init__(self, image_size = 256,
                 stride_length = 16, stride_overlap = None,
                 unroll_windows = False,
                 video_encoder_version = 'giant',
                 video_encoder_requires_grad = False,
                 video_encoder_hierarchical = True,
                 video_encoder_finetune_last_n_layers = None,
                 corr_radius = 3, 
                 max_freq = 10, n_iters = 4, embedding_dim = 256,
                 query_patch_size = 9,
                 use_volume_embedding = False,
                 per_camera_cube_scale = False,
                 principal_point_embedding = False,
                 intrinsic_embedding = False,
                 metric_ray_translation = False,
                 latent_dim = 1024, n_heads = 8,
                 n_time_space_blocks = 6, embedding_factor = 4,
                 use_camera_self_attention = False,
                 use_temporal_self_attention = False,
                 mode_3d = 'encoder',
                 output_mode = 'direct',
                 scene_encoder_proj = False,
                 scene_proj_dim = None,
                 scene_proj_prenorm = False,
                 scene_proj_mlp = False,
                 head_3d_grid_size = 8,
                 head_3d_grid_radius = 1.0,
                 log_3d_output = False,
                 log_3d_eps = 0.1,
                 depth_log_min = -2.5,
                 depth_log_max = 2.0,
                 f_eff_scale = False):
        super().__init__()

        self.mode_3d = mode_3d
            
        # video processing
        self.S = stride_length
        self.n_frames = stride_length
        self.image_size = image_size

        
        if stride_overlap is None:
            self.stride_overlap = self.S // 2
        else:
            self.stride_overlap = stride_overlap

        # Training-only: when windowing is active, carry cross-window state (re-anchored
        # query + decoder latent) with the autograd graph connected (BPTT through the
        # window chain) instead of detaching it. Default False = detached (cheap)
        # variant. Inert at inference -- detach is a no-op without a backward, so
        # predictions are identical either way.
        self.unroll_windows = unroll_windows
        
        # encoder params
        # video_encoder_requires_grad may be a bool (freeze/unfreeze for the whole
        # run) or an int (iteration at which to switch gradients on). When it's an
        # int the encoder starts frozen and is unfrozen later via
        # maybe_unfreeze_video_encoder().
        # NOTE: bool is a subclass of int, so check bool first.
        if isinstance(video_encoder_requires_grad, bool):
            self.video_encoder_unfreeze_iter = None
            initial_requires_grad = video_encoder_requires_grad
        else:
            self.video_encoder_unfreeze_iter = int(video_encoder_requires_grad)
            initial_requires_grad = False
        self.video_encoder_requires_grad = initial_requires_grad
        self.video_encoder_version = video_encoder_version
        self.video_encoder_hierarchical = video_encoder_hierarchical
        self.video_encoder_finetune_last_n_layers = video_encoder_finetune_last_n_layers
        

        # query encoder params
        self.corr_radius = corr_radius 
        self.corr_dim = 2 * self.corr_radius + 1
        self.max_freq = max_freq     
        self.embedding_dim = embedding_dim
        self.use_volume_embedding = use_volume_embedding
        self.per_camera_cube_scale = per_camera_cube_scale
        self.principal_point_embedding = principal_point_embedding
        self.intrinsic_embedding = intrinsic_embedding
        self.metric_ray_translation = metric_ray_translation

        # decoder params
        self.latent_dim = latent_dim 
        self.n_iters = n_iters
        self.n_heads = n_heads
        self.n_time_space_blocks = n_time_space_blocks
        self.embedding_factor = embedding_factor
        self.use_camera_self_attention = use_camera_self_attention
        self.use_temporal_self_attention = use_temporal_self_attention
        self.output_mode = output_mode
        self.scene_encoder_proj = scene_encoder_proj
        self.scene_proj_dim = scene_proj_dim
        self.scene_proj_prenorm = scene_proj_prenorm
        self.scene_proj_mlp = scene_proj_mlp
        self.f_eff_scale = f_eff_scale

        assert output_mode in ['direct', 'residual', 'grid', 'gridresid', 'resdirect'], 'output_mode should be "direct", "residual", "grid", "gridresid", or "resdirect"'
        # Grid (classification) modes: per-axis marginal soft-argmax + cross-entropy,
        # mirroring TrackerTapNext. "gridresid" predicts a motion residual offset.
        self.is_grid = output_mode in ('grid', 'gridresid')
        
        # self.transform_norm = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.transform_norm = transforms.Compose([
            PadToSize(self.image_size),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])


        self.scene_encoder = SceneRepresentation(
            version = self.video_encoder_version,
            freeze_encoder = not initial_requires_grad,
            n_frames = self.n_frames,
            image_size = self.image_size,
            hierarchical_features = self.video_encoder_hierarchical,
            decoder_dim = (scene_proj_dim or latent_dim) if scene_encoder_proj else None,
            proj_prenorm = scene_proj_prenorm,
            proj_mlp = scene_proj_mlp,
            video_encoder_finetune_last_n_layers = self.video_encoder_finetune_last_n_layers,
        )
        
        self.query_encoder = QueryEncoder(
            embed_dim=embedding_dim,
            decoder_dim=latent_dim,
            n_frames=self.n_frames, 
            corr_radius=corr_radius, 
            max_freq=max_freq,
            patch_size=query_patch_size,
            use_volume_embedding=use_volume_embedding,
            principal_point_embedding=principal_point_embedding,
            intrinsic_embedding=intrinsic_embedding,
        )
        self.decoder = Decoder(
            embed_dim=latent_dim,
            encoder_dim=self.scene_encoder.embed_dim,
            num_heads=n_heads,
            num_layers=n_time_space_blocks,
            mlp_ratio=embedding_factor,
            use_camera_self_attention=self.use_camera_self_attention,
            use_temporal_self_attention=self.use_temporal_self_attention,
            output_mode=self.output_mode,
            head_3d_grid_size=head_3d_grid_size,
            head_3d_grid_radius=head_3d_grid_radius,
            log_3d_output=log_3d_output,
            log_3d_eps=log_3d_eps,
            depth_log_min=depth_log_min,
            depth_log_max=depth_log_max,
            image_size=self.image_size,
            f_eff_scale=f_eff_scale,
        )

    def unfreeze_video_encoder(self, iteration):
        """Unfreeze the video encoder once `iteration` reaches the configured
        switch-on point (video_encoder_requires_grad given as an int).

        Returns True the iteration the encoder is unfrozen, False otherwise.
        Safe to call every iteration -- it is a no-op when there is nothing
        scheduled or once the encoder is already trainable.
        """
        if self.video_encoder_unfreeze_iter is None:
            return False
        if self.video_encoder_requires_grad:
            return False
        if iteration < self.video_encoder_unfreeze_iter:
            return False

        self.scene_encoder.set_encoder_requires_grad(True)
        self.video_encoder_requires_grad = True
        return True

    def print_summary(self):
        print("Hey! PARAMETERS")
        print("  total parameters: {:,d}".format(count_parameters(self)))
        print("  query encoder params: {:,d}".format(count_parameters(self.query_encoder)))
        print("  scene representation params: {:,d}".format(count_parameters(self.scene_encoder)))
        print("  decoder params: {:,d}".format(count_parameters(self.decoder)))
        
    def forward(self, views, coords, camera_group, query_times=None, init_latent=None):
        '''
        B: batch size
        T: number of frames in video
        C: number of channels 
        H: height of image
        W: width of image
        D: latent dimension
        '''
        
        device = coords.device

        B, N, R = coords.shape
        B, T, H, W, C = views[0].shape

        n_cams = len(views)

        assert len(views) == len(camera_group), "views should match number of cameras"
        
        if R == 2:
            assert len(views) == 1, "should only have 1 view for 2d input"
        
        # assert self.n_frames == T

        if R == 3:
            cube_scale = get_camera_scale(camera_group, coords)  # (n_cams, B)
        else:
            cube_scale = torch.ones((n_cams, B), device=device)
        if not self.per_camera_cube_scale:
            med = torch.median(cube_scale, dim=0).values  # (B,)
            cube_scale = med[None, :].expand(n_cams, B).contiguous()

        # Effective focal per camera (cropped+resized intrinsics). cube_scale only converts
        # world->pixels, leaving a leftover f_eff factor in the absolute-depth outputs; scaling
        # those by f_eff (~= scene depth Z) makes the head targets O(1) uniformly across datasets.
        # Gated on R==3 like cube_scale (the 2D-only path keeps cube_scale==1, so f_eff==1).
        f_eff = None
        if self.f_eff_scale:
            if R == 3:
                f_eff = torch.stack([
                    0.5 * (cam['mat'][0, 0] + cam['mat'][1, 1]) for cam in camera_group
                ]).to(device)  # (n_cams,)
                if not self.per_camera_cube_scale:
                    f_eff = torch.full((n_cams,), torch.median(f_eff).item(), device=device)
            else:
                f_eff = torch.ones((n_cams,), device=device)

        # Ray translations must share a scale across cameras so that PROPE-style
        # CameraSelfAttention sees consistent inter-camera geometry.
        cube_scale_shared = torch.median(cube_scale, dim=0).values  # (B,)

        # Metric ray-translation: recenter camera origins to the scene centroid and
        # divide by a shared metric radius (median cam->centroid distance). This makes
        # the encoded camera positions origin- and focal-invariant and O(1) across
        # datasets, while preserving relative camera-rig geometry. Gated by config;
        # default off keeps the legacy cube_scale/200 normalization bit-identical.
        scene_center = None
        scene_radius = None
        if self.metric_ray_translation:
            centers_w = torch.stack([cam['center'] for cam in camera_group])  # (cams, 3)
            if R == 3:
                scene_center = torch.nanmean(coords.to(torch.float32), dim=1)  # (B, 3)
                dist = (centers_w[:, None, :] - scene_center[None, :, :]).norm(dim=-1)  # (cams, B)
                scene_radius = torch.median(dist, dim=0).values  # (B,)
            else:
                # single-camera 2d: translation is irrelevant to self-attention.
                scene_center = centers_w[0][None].expand(B, 3)  # (B, 3) -> origin 0
                scene_radius = torch.ones(B, device=device)

        if query_times is None:
            query_times = torch.zeros((B, N), dtype=torch.int32, device=device)
        
        assert query_times.shape[0] == B
        assert query_times.shape[1] == N
            
        # normalize frames
        views_norm = []
        for i, frames in enumerate(views): 
            # frames = 2 * (frames / 255.0) - 1
            frames = frames.to(device)
            frames = rearrange(frames, 'b t h w c -> b t c h w')
            frames = self.transform_norm(frames)
            views_norm.append(frames)

        return self._forward_windows(
            views_norm, coords, query_times, camera_group,
            cube_scale, cube_scale_shared, f_eff, scene_center, scene_radius,
            init_latent=init_latent)

    def _forward_windows(self, views_norm, coords, query_times, camera_group,
                         cube_scale, cube_scale_shared, f_eff,
                         scene_center, scene_radius, init_latent=None):
        """Run the encoder/decoder over the clip, optionally as a sliding window.

        When self.S >= T (window covers the whole clip) this is a single pass,
        identical to the original non-windowed forward. Otherwise we slide a window
        of length self.S with step (self.S - self.stride_overlap), re-anchoring each
        new window's query on the previous window's prediction at the first frame of
        the next window. That re-anchoring warm-starts each window from the previous
        one's state and is what curbs long-horizon drift.

        Tracking is forward-only: each track is seeded in the window containing its
        query frame and propagated forward, so query_anytime queries land in-range
        per window. Frames before a track's query frame are not produced and must be
        dropped from the loss (causal_masking=true) when query_anytime=true; with all
        queries at frame 0 this is moot and the path reduces to the original behavior.

        Cross-window state (re-anchored query + carried latent) is detached by default,
        so windowed training is the cheap variant (each window's backward is
        independent, same memory as a single pass) yet still learns to use the carry.
        Set unroll_windows=True to keep the graph connected across windows (BPTT through
        the window chain) so the model also learns to produce good carry states; peak
        memory then scales ~n_windows x.
        """
        device = coords.device
        B, N, R = coords.shape
        T = views_norm[0].shape[1]
        S = self.S

        # Window >= clip: single pass, bit-identical to the original forward.
        # (prev_latent stays None -> latent_carry is a no-op; latent return discarded.)
        if S >= T:
            result, latent = self._forward_window(
                views_norm, coords, query_times, camera_group,
                cube_scale, cube_scale_shared, f_eff, scene_center, scene_radius,
                prev_latent=init_latent)
            # Expose the latent so a caller can thread it into a following chunk
            # (cross-chunk carry); harmless for the non-windowed model whose
            # latent_carry is untrained (a no-op).
            result['final_latent'] = latent
            return result

        stride_remainder = S - self.stride_overlap
        assert stride_remainder >= 1, "stride_overlap must be < stride_length"
        # ceil((T - S) / stride_remainder) + 1 windows to cover all T frames.
        n_windows = (T - S + stride_remainder - 1) // stride_remainder + 1
        T_full = stride_remainder * (n_windows - 1) + S
        n_pad = T_full - T

        # Pad the (already normalized) frames by repeating the last frame so the
        # final window is full length; outputs are trimmed back to T at the end.
        if n_pad > 0:
            views_norm = [
                torch.cat([v, v[:, -1:].expand(-1, n_pad, *([-1] * (v.dim() - 2)))], dim=1)
                for v in views_norm
            ]

        # Frame (t) axis of every time-indexed output, used to stitch windows.
        t_axis = {
            'coords_pred': 1, '3d_pred_direct': 1, '3d_pred_rays': 1,
            '3d_pred_triangulate': 1, 'vis_pred': 1, 'conf_pred': 1,
            '3d_pred_cams_direct': 2, '3d_pred_cams_rays': 2, 'conf_3d': 2,
            '2d_pred': 2, 'vis_pred_2d': 2, 'conf_pred_2d': 2, 'depth_pred': 2,
            '2d_logits': 2,
        }
        grid_t_axis = {'logits_3d': 2, 'logits_depth': 2, 'anchor_local': 2}

        def _put(store, key, src, axis, start, full):
            # Write a window's output slice into the full-length accumulator across
            # many keys with non-uniform time axes: lazily allocate the full-length
            # output on first sight, then write this window's slice. Later windows
            # overwrite earlier ones in the overlap region.
            if src is None:
                store[key] = None
                return
            if store.get(key) is None:
                shape = list(src.shape)
                shape[axis] = full
                store[key] = src.new_zeros(shape)
            idx = [slice(None)] * src.dim()
            idx[axis] = slice(start, start + src.shape[axis])
            store[key][tuple(idx)] = src

        acc = {}
        q = query_times                 # (B, N) absolute query-frame index per track
        reanchor = None                 # (B, N, R) predicted coords at this window's first frame
        carry = init_latent             # (B, S, N, cams, D) carried latent; init_latent threads
                                        # state in from a previous chunk (cross-chunk carry)
        for w in range(n_windows):
            ix = stride_remainder * w
            before = q < ix                       # (B, N) track appeared in an earlier window
            in_win = (q >= ix) & (q < ix + S)     # query frame lands in this window (seed here)

            # Per-track forward-only seeding: a track tracked from an earlier window is
            # re-anchored on the carried prediction; a track whose query frame is in (or
            # still after) this window uses its original query. Query time is the
            # in-window offset for seeds, else 0. Tracks whose query frame is after this
            # window are placeholders -- their pre-query frames are dropped by the loss
            # (requires causal_masking). Reduces exactly to the uniform frame-0 case
            # when every query is at frame 0.
            if reanchor is None:
                coords_w = coords
            else:
                coords_w = torch.where(before.unsqueeze(-1), reanchor, coords)
            times_w = torch.where(in_win, (q - ix).clamp(0, S - 1),
                                  torch.zeros_like(q)).to(torch.int32)
            # Carried latent only feeds re-anchored (already-active) tracks; zero it for
            # seeds / not-yet-appeared tracks so their carry is a no-op. Exception:
            # window 0's carry comes from a previous chunk (init_latent) and represents
            # continuations, so feed it to all tracks (no `before` mask).
            if carry is None:
                carry_w = None
            elif w == 0:
                carry_w = carry
            else:
                carry_w = carry * before.to(carry.dtype).view(B, 1, N, 1, 1)

            views_w = [v[:, ix:ix + S] for v in views_norm]
            res, latent = self._forward_window(
                views_w, coords_w, times_w, camera_group,
                cube_scale, cube_scale_shared, f_eff, scene_center, scene_radius,
                prev_latent=carry_w)

            for key, ax in t_axis.items():
                if key in res:
                    _put(acc, key, res[key], ax, ix, T_full)
            if 'grid' in res:
                acc.setdefault('grid', {})
                for key, val in res['grid'].items():
                    if key in grid_t_axis:
                        _put(acc['grid'], key, val, grid_t_axis[key], ix, T_full)
                    else:
                        acc['grid'][key] = val  # window-independent metadata

            # Cross-window state is detached unless unroll_windows is set, in which
            # case the graph stays connected for BPTT through the window chain.
            detach = not self.unroll_windows

            # Build the latent carry from this window: the overlap frames
            # [stride_remainder:S] become the first `stride_overlap` frames of the next
            # window; the remaining future frames are seeded with the last overlap frame.
            # Built every window so the last one's carry is returned as final_latent (to
            # thread into a following chunk). No overlap -> nothing to carry.
            if self.stride_overlap >= 1:
                overlap_latent = latent[:, stride_remainder:S]  # (b, overlap, n, cams, d)
                pad = overlap_latent[:, -1:].expand(
                    -1, S - overlap_latent.shape[1], -1, -1, -1)
                carry = torch.cat([overlap_latent, pad], dim=1)
                if detach:
                    carry = carry.detach()
            else:
                carry = None

            if w < n_windows - 1:
                # Re-anchor the next window's query on this window's prediction at its
                # first frame (relative index = stride_remainder).
                rel = min(stride_remainder, S - 1)
                if R == 3:
                    reanchor = res['coords_pred'][:, rel]
                else:
                    reanchor = res['2d_pred'][0, :, rel]
                if detach:
                    reanchor = reanchor.detach()

        # The carry from the last window is the state to thread into a following chunk.
        acc['final_latent'] = carry

        # Trim the padded frames back to the true clip length T.
        if n_pad > 0:
            for key, ax in t_axis.items():
                if acc.get(key) is not None:
                    idx = [slice(None)] * acc[key].dim()
                    idx[ax] = slice(0, T)
                    acc[key] = acc[key][tuple(idx)]
            if 'grid' in acc:
                for key, ax in grid_t_axis.items():
                    if acc['grid'].get(key) is not None:
                        idx = [slice(None)] * acc['grid'][key].dim()
                        idx[ax] = slice(0, T)
                        acc['grid'][key] = acc['grid'][key][tuple(idx)]

        return acc

    def _forward_window(self, views_norm, coords, query_times, camera_group,
                        cube_scale, cube_scale_shared, f_eff,
                        scene_center, scene_radius, prev_latent=None):
        """Single encoder/decoder pass over one window (or the whole clip).

        views_norm frames are already normalized/padded ('b t c h w'); coords are the
        per-window query anchor (B, N, R) and query_times their frame index within the
        window. Scene-level scalars (cube_scale, f_eff, scene_center/radius) are
        precomputed once in forward() and passed through unchanged. prev_latent is the
        previous window's final decoder latent (frame-aligned, B,T,N,cams,D) or None.

        Returns (result_dict, latent) where latent is this window's final decoder
        latent, to be carried into the next window.
        """
        device = coords.device
        B, N, R = coords.shape
        T = views_norm[0].shape[1]
        n_cams = len(views_norm)

        scene_features = self.scene_encoder(views_norm)

        # Hey, coords start at 0
        query_coords = repeat(coords, 'b n r -> b (t n) r', t=T).to(torch.float32)
        # query_time = torch.zeros((B, T * N), dtype=torch.int32, device=device)
        query_times_rep = repeat(query_times, 'b n -> b (t n)', t=T)
        target_time = repeat(torch.arange(T, device=device), 't -> b (t n)', b=B, t=T, n=N)
        
        query_embeds = self.query_encoder(
            views_norm, camera_group,
            query_coords = query_coords,
            query_time = query_times_rep,
            target_time = target_time,
            cube_scale = cube_scale
        )
        # Reshape from flat (b, t*n, cams, d) → explicit (b, t, n, cams, d) for Decoder
        query_embeds = rearrange(query_embeds, 'b (t n) cams d -> b t n cams d', t=T, n=N)

        if R == 3:
            p2d_query = project_points_torch(camera_group, query_coords) # [cams, b, (t n), 2]
            p2d_query = rearrange(p2d_query, 'cams b (t n) r -> cams b t n r', t=T, n=N)
        else:
            p2d_query = rearrange(query_coords, 'b (t n) r -> 1 b t n r', t=T, n=N)

        query_rays_per_cam = []
        for i in range(len(camera_group)):
            rays_per_b = []
            for b in range(B):
                p2d_ib = rearrange(p2d_query[i, b], 't n r -> (t n) r')
                if self.metric_ray_translation:
                    rays_per_b.append(points_to_rays(
                        camera_group[i], p2d_ib, cube_scale_shared[b],
                        scene_center=scene_center[b], scene_radius=scene_radius[b]))
                else:
                    rays_per_b.append(points_to_rays(camera_group[i], p2d_ib, cube_scale_shared[b]))
            query_rays_per_cam.append(torch.stack(rays_per_b, dim=0))  # (B, T*N, 4, 4)
        query_rays_flat = torch.stack(query_rays_per_cam, dim=0)  # (cams, B, T*N, 4, 4)
        query_rays = rearrange(query_rays_flat, 'cams b (t n) d e -> b t n cams d e', t=T, n=N)

        mode_idx = torch.tensor([1 if R == 3 else 0], dtype=torch.long, device=query_embeds.device)
        outputs, grid_logits, latent = self.decoder(
            scene_features, query_embeds, query_rays, mode_idx, prev_latent=prev_latent)
        outputs = rearrange(outputs, 'b t n cams outdim -> cams b t n outdim')

        points_3d_raw, points_pred, vis_pred_2d_logits, conf_pred_2d_logits, depth_pred, conf_3d_logits = torch.split(
            outputs, [3, 2, 1, 1, 1, 1], dim=-1
        )


        vis_pred_2d = F.sigmoid(vis_pred_2d_logits)
        conf_pred_2d = F.sigmoid(conf_pred_2d_logits)

        conf_3d = torch.softmax(conf_3d_logits[..., 0], dim=0)


        # qc = rearrange(query_coords, 'b (t n) r -> b t n 1 r', t=T, n=N)
        # centers = torch.stack([cam['center'] for cam in camera_group])
        # depths_query = torch.linalg.norm(qc - centers, dim=-1)
        # depths_query_shaped = rearrange(depths_query, 'b t n cams -> cams b t n')
        # depth_pred_scaled = depths_query_shaped + depth_pred[..., 0] * cube_scale * self.depth_scale

        # Normalized depth -> metric. In grid mode the decoder already decoded
        # depth_norm = exp(soft-argmax(log-depth bins)) (>0), so skip the softplus.
        if self.is_grid:
            depth_norm = depth_pred[..., 0]
        else:
            depth_norm = F.softplus(depth_pred[..., 0])
        depth_pred_scaled = depth_norm * rearrange(cube_scale, 'cams b -> cams b 1 1')
        if self.f_eff_scale:
            # depth is absolute in every output mode -> always f_eff-scaled
            depth_pred_scaled = depth_pred_scaled * rearrange(f_eff, 'cams -> cams 1 1 1').to(depth_pred_scaled.dtype)


        if self.is_grid:
            # grid 2D head decodes absolute pixel positions directly (soft-argmax).
            points_pred_scaled = points_pred
        elif self.output_mode in ('residual', 'resdirect'):
            # Predict offsets instead of absolute bounded coordinates
            points_pred_scaled = p2d_query + points_pred
        elif self.output_mode == 'direct':
            # Predict absolute coordinates
            points_pred_scaled = points_pred + self.image_size // 2

  
      
        
        points_und = torch.stack([
            undistort_points(camera_group[i], points_pred_scaled[i])
            for i in range(len(camera_group))
        ])

        # # get 3d points from each cameras using rays
        rays_norm = to_homogeneous(points_und)
        rot_mats = torch.stack([cam['ext'][:3,:3] for cam in camera_group])
        rays_world = einsum(rays_norm, rot_mats, 'cams b t n r, cams r x -> cams b t n x')
        rays_world = F.normalize(rays_world, dim=-1)

        centers = torch.stack([cam['center'] for cam in camera_group])
        cadd = repeat(centers, 'cams r -> cams 1 1 1 r')
        points_3d_all_rays = cadd + einsum(rays_world, depth_pred_scaled,
                                      'cams b t n r, cams b t n -> cams b t n r')
        points_3d_rays = einsum(points_3d_all_rays, conf_pred_2d[..., 0],
                                'cams b t n r, cams b t n -> b t n r')


        # triangulate points
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

        
        # direct residual predictions
        center = torch.tensor([self.image_size // 2, self.image_size//2],
                              device=device, dtype=torch.float32).reshape(1, 2)
        rays_c = torch.stack([points_to_rays(cam, center, normalize_t=False)[0] for cam in camera_group])
        rays_c_inv = _invert_SE3(rays_c)  # [cams, 4, 4], ray-local → world
        
        add_residual = (self.output_mode in ('residual', 'gridresid')) or \
                       (self.output_mode == 'resdirect' and R == 3)

        query_local = None
        p3d_cams = points_3d_raw * rearrange(cube_scale, 'cams b -> cams b 1 1 1')
        if self.output_mode == 'gridresid':
            # gridresid bins encode motion / (cube * image_size) (~pixels, NO f_eff ->
            # ortho-safe); map back to metric motion by * image_size (mirror
            # tracker_tapnext.py:597-600).
            p3d_cams = p3d_cams * self.image_size
        if self.f_eff_scale and not add_residual:
            # direct 3D output is an absolute ray-local position -> f_eff-scaled.
            # The residual branch (motion offset) is NOT scaled (handled by scale_3d only).
            p3d_cams = p3d_cams * rearrange(f_eff, 'cams -> cams 1 1 1 1').to(p3d_cams.dtype)

        if add_residual:
            if R == 3:
                query_world = repeat(
                    rearrange(query_coords, 'b (t n) r -> b t n r', t=T, n=N),
                    'b t n r -> cams b t n r', cams=n_cams)
            elif R == 2:
                # only reachable for output_mode == 'residual'
                t_idx = repeat(query_times, 'b n -> b 1 n r', r=3)
                query_3d = torch.gather(points_3d_rays, dim=1, index=t_idx)  # (b, 1, n, 3)
                query_world = repeat(query_3d, 'b 1 n r -> cams b t n r', t=T, cams=n_cams)

            # convert query from world → ray-local space, then add as residual before world transform
            query_local = from_homogeneous(
                einsum(rays_c, to_homogeneous(query_world),
                       'cams x r, cams b t n r -> cams b t n x')
            )
            p3d_cams = p3d_cams + query_local

        points_3d_all_direct = from_homogeneous(
            einsum(rays_c_inv, to_homogeneous(p3d_cams),
                   'cams x r, cams b t n r -> cams b t n x')
        )
        
        points_3d_direct = einsum(points_3d_all_direct, conf_3d,
                                  'cams b t n r, cams b t n -> b t n r')
            
        # # zero out 3d points with no confidence
        # bad_pred = torch.amax(conf_pred_2d[..., 0], dim=0) <= 1e-5
        # points_3d = einsum(points_3d, ~bad_pred, 'b t n r, b t n -> b t n r') 
        
        vis_pred = noisy_or_logit(vis_pred_2d_logits, dim=0)  # noisy-OR logit
        conf_pred = torch.amax(conf_3d_logits[..., 0], dim=0)
        
        # assemble outputs 
        result_dict = {
            'coords_pred': points_3d_direct, # (b, t, n, 3)
            # 
            '3d_pred_cams_direct': points_3d_all_direct, # (cams, b, t, n, 3)
            '3d_pred_cams_rays': points_3d_all_rays, # (cams, b, t, n, 3)
            'conf_3d': conf_3d_logits[..., 0], # (cams, b, t, n)
            # 
            '3d_pred_direct': points_3d_direct, # (b, t, n, 3)
            '3d_pred_rays': points_3d_rays, # (b, t, n, 3)
            '3d_pred_triangulate': points_3d_tri, # (b, t, n, 3)
            # 
            '2d_pred': points_pred_scaled, # (cams, b, t, n, 2)
            'vis_pred': vis_pred, # (b, t, n, 1)
            'conf_pred': conf_pred, # (b, t, n, 1)
            'vis_pred_2d': vis_pred_2d_logits[..., 0], # (cams, b, t, n)
            'conf_pred_2d': conf_pred_2d_logits[..., 0], # (cams, b, t, n)
            'depth_pred': depth_pred_scaled # (cams, b, t, n)
        }

        # Grid (classification) supervision: hand the loss the raw 2D/3D/depth bin
        # logits plus the geometry it needs to discretize GT into bin targets (the
        # exact inverse of the forward decode). Mirrors tracker_tapnext.py:645-672 so
        # the model-agnostic CE paths in losses.py fire identically for the encoder.
        if self.is_grid:
            result_dict['2d_logits'] = rearrange(
                grid_logits['logits_2d'], 'b t n cams p -> cams b t n p')  # (cams,b,t,n,2P)
            result_dict['grid'] = {
                'logits_3d': rearrange(grid_logits['logits_3d'],
                                       'b t n cams d g -> cams b t n d g'),  # (cams,b,t,n,3,K)
                'logits_depth': rearrange(grid_logits['logits_depth'],
                                          'b t n cams g -> cams b t n g'),   # (cams,b,t,n,K)
                'rays_c': rays_c,                  # (cams,4,4) world -> ray-local
                'cube_scale': cube_scale,          # (cams,B)
                'f_eff': f_eff if self.f_eff_scale else None,  # (cams,) or None
                'g3d_lo': self.decoder.g3d_lo, 'g3d_hi': self.decoder.g3d_hi,
                'gd_lo': self.decoder.gd_lo, 'gd_hi': self.decoder.gd_hi,
                'K': self.decoder.head_3d_grid_size,
                # gridresid: 3D bins encode the motion offset from the per-track query
                # anchor, normalized by cube*image_size (no f_eff -> ortho-safe).
                'is_resid': (self.output_mode == 'gridresid'),
                'anchor_local': query_local.detach() if query_local is not None else None,
                'image_size': self.image_size,
                # log_3d_output: the loss quantizes the 3D bin target in the same
                # signed-log warped space as the (warped) bin centres.
                'log_warp': self.decoder.log_3d_output,
                'eps': self.decoder.log_3d_eps,
                'c_range': self.decoder.log_3d_c_range,
            }

        # if self.training: 
        #     train_dict = {
        #         'coords_pred_iters': coords_pred_iters,
        #         'vis_pred_iters': vis_pred_iters, 
        #         'conf_pred_iters': conf_pred_iters}
            
        #     result_dict.update(train_dict)

        return result_dict, latent
