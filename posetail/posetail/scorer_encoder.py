"""Track-quality scorer built on the posetail tracker encoder/decoder.

`ScorerEncoder` subclasses `TrackerEncoder` so every reused submodule (scene_encoder,
query_encoder, decoder, norms, transform_norm) keeps its name and shape -> the trained
tracker checkpoint loads by name (strict=False). It adds an attention-pooling head + a
score / precision head on top of the decoder's per-point latents.

The only structural change vs the tracker is **query = target**: the tracker repeats one
query position across all frames ("where is point n at every frame?"); the scorer feeds
the FULL trajectory and evaluates each point in place at its own frame, then pools over
(time, cameras) into one latent per point and reads out a scalar quality score.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from posetail.posetail.cube import (get_camera_scale, project_points_torch,
                                    points_to_rays)
from posetail.posetail.tracker_encoder import TrackerEncoder
from posetail.posetail.encoder_decoder import AttentionPooling


class ScorerEncoder(TrackerEncoder):

    def __init__(self, *, pool_num_heads=8, score_hidden=64, use_precision=True,
                 **tracker_model_config):
        # tracker_model_config carries video_encoder_requires_grad=False -> the V-JEPA
        # backbone is built frozen; query_encoder + decoder are trainable as usual.
        super().__init__(**tracker_model_config)
        d = self.decoder.embed_dim                                # = latent_dim

        self.attn_pool = AttentionPooling(d, num_heads=pool_num_heads, n_frames=self.S)

        # Learnable token substituted for (frame, point) slots whose track coord is NaN
        # (missing/occluded). Sized to the query-encoder output dim, since it replaces the
        # fused query embedding before the decoder. Keeps those slots finite so the decoder's
        # temporal/camera self-attention can't be poisoned by NaN; the slot still flows through
        # pooling (with its time embedding), so missingness informs the score.
        self.missing_point = nn.Parameter(torch.zeros(self.query_encoder.decoder_dim))
        nn.init.normal_(self.missing_point, std=0.02)

        self.score_feature = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, score_hidden), nn.SiLU())
        self.score_head = nn.Linear(score_hidden, 1, bias=False)  # no bias (miss-alignment)
        # emits a logit -> sigmoid -> per-point confidence in (0,1) used to weight triplets
        self.precision_head = nn.Linear(score_hidden, 1) if use_precision else None

    # ----------------------------------------------------------------------------------
    # Scene encoding (frozen backbone) -- split out so good+bad reuse the same features.
    # ----------------------------------------------------------------------------------
    def encode_scene(self, views):
        """Normalize frames and run the (frozen) scene encoder.

        Mirrors TrackerEncoder.forward's normalize loop. The backbone is frozen via config
        (SceneRepresentation freeze_encoder=True), so its blocks retain no activations;
        any trainable scene projection still receives gradients. Returns (views_norm,
        scene_features).
        """
        device = next(self.parameters()).device
        views_norm = []
        for frames in views:
            frames = frames.to(device)
            frames = rearrange(frames, 'b t h w c -> b t c h w')
            frames = self.transform_norm(frames)
            views_norm.append(frames)
        scene_features = self.scene_encoder(views_norm)
        return views_norm, scene_features

    # ----------------------------------------------------------------------------------
    # Scene-level scalars (cube_scale, ray-translation centering) from the full track.
    # Copied from TrackerEncoder.forward (239-282) but over the flattened (t*k) points.
    # ----------------------------------------------------------------------------------
    def _scene_scalars(self, coords_flat, camera_group, device):
        """coords_flat: [B, t*k, R]. Returns cube_scale, cube_scale_shared, f_eff,
        scene_center, scene_radius (same semantics as the tracker)."""
        B, _, R = coords_flat.shape
        n_cams = len(camera_group)

        if R == 3:
            cube_scale = get_camera_scale(camera_group, coords_flat)   # (n_cams, B)
        else:
            cube_scale = torch.ones((n_cams, B), device=device)
        if not self.per_camera_cube_scale:
            med = torch.median(cube_scale, dim=0).values
            cube_scale = med[None, :].expand(n_cams, B).contiguous()

        f_eff = None
        if self.f_eff_scale:
            if R == 3:
                f_eff = torch.stack([
                    0.5 * (cam['mat'][0, 0] + cam['mat'][1, 1]) for cam in camera_group
                ]).to(device)
                if not self.per_camera_cube_scale:
                    f_eff = torch.full((n_cams,), torch.median(f_eff).item(), device=device)
            else:
                f_eff = torch.ones((n_cams,), device=device)

        cube_scale_shared = torch.median(cube_scale, dim=0).values     # (B,)

        scene_center = None
        scene_radius = None
        if self.metric_ray_translation:
            centers_w = torch.stack([cam['center'] for cam in camera_group])
            if R == 3:
                scene_center = torch.nanmean(coords_flat.to(torch.float32), dim=1)
                dist = (centers_w[:, None, :] - scene_center[None, :, :]).norm(dim=-1)
                scene_radius = torch.median(dist, dim=0).values
            else:
                scene_center = centers_w[0][None].expand(B, 3)
                scene_radius = torch.ones(B, device=device)

        return cube_scale, cube_scale_shared, f_eff, scene_center, scene_radius

    # ----------------------------------------------------------------------------------
    # Score one sample from precomputed scene features. query = target.
    # ----------------------------------------------------------------------------------
    def score(self, views_norm, scene_features, coords_full, camera_group):
        """coords_full: [B, T, K, R] full trajectory. Returns scores, precision: [B, K]."""
        device = coords_full.device
        B, T, K, R = coords_full.shape
        n_cams = len(camera_group)

        # Support missing (NaN) track points. Sanitize NaN coords to a finite placeholder
        # (each point's mean over its observed frames) BEFORE projection/rays/patches, so no
        # NaN reaches the decoder's temporal/camera self-attention (which would poison every
        # other frame/cam of the point). The valid mask drives the missing-point token below.
        valid = torch.isfinite(coords_full).all(dim=-1)                 # [b, t, k]
        mean_pos = torch.nanmean(coords_full.to(torch.float32), dim=1, keepdim=True)  # [b,1,k,R]
        coords_full = torch.where(valid.unsqueeze(-1), coords_full.to(torch.float32),
                                  mean_pos.expand_as(coords_full))
        coords_full = torch.nan_to_num(coords_full, nan=0.0)            # guard all-NaN points

        coords_flat = rearrange(coords_full, 'b t k r -> b (t k) r').to(torch.float32)
        cube_scale, cube_scale_shared, f_eff, scene_center, scene_radius = \
            self._scene_scalars(coords_flat, camera_group, device)

        # query == target: each (t, k) token lives at its own frame t, evaluated at t.
        query_coords = coords_flat
        query_time = repeat(torch.arange(T, device=device), 't -> b (t k)', b=B, k=K)
        target_time = query_time

        query_embeds = self.query_encoder(
            views_norm, camera_group,
            query_coords=query_coords, query_time=query_time,
            target_time=target_time, cube_scale=cube_scale)
        query_embeds = rearrange(query_embeds, 'b (t k) cams d -> b t k cams d', t=T, k=K)

        # Replace embeddings of missing (NaN) slots with the learnable missing-point token
        # (broadcast over cameras). Their sanitized coords only served to keep projection/rays
        # finite; the token is what the decoder + pooling actually see for those slots.
        query_embeds = torch.where(
            (~valid)[..., None, None], self.missing_point.to(query_embeds.dtype), query_embeds)

        if R == 3:
            p2d_query = project_points_torch(camera_group, query_coords)       # [cams,b,(t k),2]
            p2d_query = rearrange(p2d_query, 'cams b (t k) r -> cams b t k r', t=T, k=K)
        else:
            p2d_query = rearrange(query_coords, 'b (t k) r -> 1 b t k r', t=T, k=K)

        query_rays_per_cam = []
        for i in range(n_cams):
            rays_per_b = []
            for b in range(B):
                p2d_ib = rearrange(p2d_query[i, b], 't k r -> (t k) r')
                if self.metric_ray_translation:
                    rays_per_b.append(points_to_rays(
                        camera_group[i], p2d_ib, cube_scale_shared[b],
                        scene_center=scene_center[b], scene_radius=scene_radius[b]))
                else:
                    rays_per_b.append(points_to_rays(camera_group[i], p2d_ib, cube_scale_shared[b]))
            query_rays_per_cam.append(torch.stack(rays_per_b, dim=0))
        query_rays = rearrange(torch.stack(query_rays_per_cam, dim=0),
                               'cams b (t k) d e -> b t k cams d e', t=T, k=K)

        mode_idx = torch.tensor([1 if R == 3 else 0], dtype=torch.long, device=device)

        # third decoder return is the per-point latent [b, t, k, cams, dim]
        latents = self.decoder(scene_features, query_embeds, query_rays, mode_idx)[2]

        pooled = self.attn_pool(latents)                            # [b, k, d]
        feats = self.score_feature(pooled)
        scores = self.score_head(feats)[..., 0]                     # [b, k]
        if self.precision_head is not None:
            precision = torch.sigmoid(self.precision_head(feats)[..., 0])  # [b, k], (0,1)
        else:
            precision = torch.ones_like(scores)  # full confidence -> weighting is a no-op
        return scores, precision

    def forward(self, views, coords, camera_group, query_times=None):
        """Single-sample inference path. views: list of [b,t,h,w,c];
        coords: full trajectory [b,t,k,R]. Returns scores, precision: [b, k]."""
        views_norm, scene_features = self.encode_scene(views)
        return self.score(views_norm, scene_features, coords, camera_group)

    def score_triplet(self, trip):
        """Score a (good, bad, anchor) triplet in ONE forward pass (DDP-safe: a single
        entry/exit of the module's forward, one backward).

        trip is the dict from datasets.scorer_corruption.make_triplet. Returns
        scores, precision, labels each [N, 3] (N = b*k; columns good, bad, anchor).
        """
        gv, gc, gcg = trip['good']
        _, bc, bcg = trip['bad']                       # bad shares good's pixels
        av, ac, acg = trip['anchor']

        vn, sf = self.encode_scene(gv)
        good_s, good_p = self.score(vn, sf, gc, gcg)
        bad_s, bad_p = self.score(vn, sf, bc, bcg)
        if trip['reuse_scene_for_anchor']:
            avn, asf = vn, sf
        else:
            avn, asf = self.encode_scene(av)
        anc_s, anc_p = self.score(avn, asf, ac, acg)

        scores = torch.stack([good_s, bad_s, anc_s], dim=-1).reshape(-1, 3)
        precision = torch.stack([good_p, bad_p, anc_p], dim=-1).reshape(-1, 3)
        al = float(trip['anchor_label'])
        labels = torch.tensor([1.0, -1.0, al], device=scores.device).expand_as(scores)
        return scores, precision, labels

    def print_summary(self):
        from posetail.posetail.utils import count_parameters
        print("ScorerEncoder PARAMETERS")
        print("  total parameters: {:,d}".format(count_parameters(self)))
        print("  query encoder params: {:,d}".format(count_parameters(self.query_encoder)))
        print("  scene representation params: {:,d}".format(count_parameters(self.scene_encoder)))
        print("  decoder params: {:,d}".format(count_parameters(self.decoder)))
        print("  attn_pool params: {:,d}".format(count_parameters(self.attn_pool)))
