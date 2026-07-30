import itertools
# import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange, einsum, reduce, repeat

from posetail.posetail.cube import UnprojectViews, project_volumes, project_points_torch, get_camera_scale
from posetail.posetail.transformer import TimeSpaceTransformer, MLP
from posetail.posetail.networks import ResidualFeatureExtractor, TriplaneFeatureExtractor
from posetail.posetail.networks import MinicubesV2V, SimpleV2V, ViewAttentionV2V, QueryViewAttentionV2V
from posetail.posetail.networks import HieraFeatureExtractor, SAM2HieraFeatureExtractor 
from posetail.posetail.utils import get_pos_encoding, get_fourier_encoding, PadToMultiple, count_parameters

from torchvision import transforms

# ImageNet normalization stats (inlined from timm.data.constants to drop the dep).
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Tracker(nn.Module): 

    def __init__(self, track_3d = True, stride_length = 8, 
                 stride_overlap = None, downsample_factor = 4, cnn_type = 'sam2',
                 hiera_requires_grad = False, vc_head_requires_grad = True,
                 cube_dim = 20, cube_extent = None, upsample_factor = 1, 
                 corr_levels = 4, corr_radius = 3, 
                 corr_hidden_dim = 384, corr_output_dim = 256, 
                 max_freq = 10, n_iters = 4, embedding_dim = 256, 
                 latent_dim = 128, n_virtual = 64, n_heads = 8, 
                 n_time_space_blocks = 6, embedding_factor = 4,
                 mode_3d = 'triplane'): 
        super().__init__()

        if track_3d: 
            self.R = 3 
        else: 
            self.R = 2

        self.mode_3d = mode_3d
            
        # video processing
        self.S = stride_length
        
        if stride_overlap is None: 
            self.stride_overlap = self.S // 2
        else:
            self.stride_overlap = stride_overlap 
        
        # cnn params
        self.cnn_type = cnn_type
        self.downsample_factor = downsample_factor
        self.latent_dim = latent_dim 
        self.hiera_requires_grad = hiera_requires_grad

        # cube params
        self.cube_dim = cube_dim 
        self.cube_extent = cube_extent

        # this gives us the indices for xy, xz, and yz planes for 3d 
        # tracking, or xy planes for 2d tracking
        #
        if self.R == 3 and self.mode_3d == 'minicubes':
            self.plane_ixs = [(0,1)] # not used, but this makes input_dim work out later
        else:
            self.plane_ixs = list(itertools.combinations(range(self.R), 2))
        self.upsample_factor = upsample_factor

        # correlation params
        self.corr_levels = corr_levels 
        self.corr_radius = corr_radius 
        self.corr_dim = 2 * self.corr_radius + 1
        self.corr_hidden_dim = corr_hidden_dim
        self.corr_output_dim = corr_output_dim

        # transformer params
        self.max_freq = max_freq     
        self.n_iters = n_iters
        self.n_virtual = n_virtual
        self.embedding_dim = embedding_dim   
        self.n_heads = n_heads
        self.n_time_space_blocks = n_time_space_blocks
        self.embedding_factor = embedding_factor
        self.activation_kwargs = {'approximate': 'tanh'}
        self.vc_head_requires_grad = vc_head_requires_grad

        # add up the dimensions for transformer input
        self.input_dim = (2 + 2 * self.R + 4 * self.R * self.max_freq +
                          self.corr_levels * self.corr_output_dim * len(self.plane_ixs))
        # print(f'transformer input dimension: {self.input_dim}') 

        # networks
        if self.cnn_type == 'resnet': 

            self.cnn = ResidualFeatureExtractor(
                input_dim = 3, # RGB 
                output_dim = self.latent_dim,
                n_blocks = 4,
                kernel_size = 3,
                downsample_factor = self.downsample_factor,
                spatial_res_factor = 2 
            )

            self.transform_norm = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)


        else: # self.cnn_type == 'sam2'
            # self.cnn = HieraFeatureExtractor(output_dim=self.latent_dim)
            freeze_nonlast_fpn = not (self.R == 3 and self.mode_3d == 'minicubes')
            # freeze_nonlast_fpn = True
            self.cnn = SAM2HieraFeatureExtractor(output_dim = self.latent_dim,
                                                 requires_grad = self.hiera_requires_grad,
                                                 freeze_nonlast_fpn = freeze_nonlast_fpn)
            
            self.transform_norm = transforms.Compose([
                PadToMultiple(32),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

        
        if self.R == 3:
            if self.mode_3d == 'triplane':
                # triplane features
                self.triplane_cnn = TriplaneFeatureExtractor(
                    input_dim = self.latent_dim * len(self.plane_ixs), 
                    n_hidden_layers = 2, 
                    kernel_size = 3, 
                    padding = 1, 
                    upsample_factor = self.upsample_factor
                )
            elif self.mode_3d == 'minicubes':
                # self.minicube_v2v = MinicubesV2V(self.latent_dim)
                # self.minicube_v2v = nn.Identity()
                # self.minicube_v2v = SimpleV2V(self.latent_dim)
                self.minicube_v2v = nn.ModuleList([SimpleV2V(self.latent_dim) for _ in range(self.corr_levels)])
                # self.minicube_v2v = DepthwiseSeparableV2V(self.latent_dim)
                # self.minicube_v2v = PlanesV2V(self.latent_dim)
                # self.view_attention = nn.ModuleList([ QueryViewAttentionV2V(self.latent_dim) for _ in range(self.corr_levels) ])
                
        # correlation features
        if self.R == 3 and self.mode_3d == 'minicubes':
            # mlp_input_dim = (2 * self.corr_radius + 1) ** 6
            mlp_input_dim = (2 * self.corr_radius + 1) ** 3
            # mlp_input_dim = (2 * self.corr_radius + 1) ** 3 * (3**3)
        else:
            mlp_input_dim = (2 * self.corr_radius + 1) ** 4

        self.corr_mlp = MLP(
            input_dim = mlp_input_dim, 
            embedding_dim = self.corr_hidden_dim, 
            output_dim = self.corr_output_dim
        )

        # time embeddings
        t = torch.arange(self.S)

        self.register_buffer(
            'time_encoding', 
            get_pos_encoding(t, self.input_dim).float()
        )

        # transformer
        self.tsformer = TimeSpaceTransformer(
            input_dim = self.input_dim, 
            embedding_dim = self.embedding_dim, 
            output_dim = self.R + 2, 
            n_time_space_blocks = self.n_time_space_blocks, 
            n_heads = self.n_heads,
            n_virtual = self.n_virtual, 
            embedding_factor = self.embedding_factor, 
            vc_head = True, 
            **self.activation_kwargs
        )

        # freeze modules (e.g. for fine-tuning)
        self.freeze_modules() 

    def print_summary(self):
        print("PARAMETERS")
        print("  total parameters: {:,d}".format(count_parameters(self)))
        if self.mode_3d == 'triplane':
            print("  triplane cnn params: {:,d}".format(count_parameters(self.triplane_cnn)))
        elif self.mode_3d == 'minicubes':
            print("  minicube v2v params: {:,d}".format(count_parameters(self.minicube_v2v)))
        print("  corr mlp params: {:,d}".format(count_parameters(self.corr_mlp)))
        print("  timesformer params: {:,d}".format(count_parameters(self.tsformer)))

    def freeze_modules(self): 

        # freeze vis conf head in the timesformer
        if not self.vc_head_requires_grad: 
            self._freeze_modules('tsformer.vc_head')


    def _freeze_modules(self, module_name):

        module = self
        attributes = module_name.split('.') 

        # get the module
        for attr in attributes: 
            if hasattr(module, attr): 
                module = getattr(module, attr)
            else: 
                print(f'{module} has no attribute {attr}')

        # freeze the params (remove grad)
        for param in module.parameters(): 
            print(f'freezing params for module {module}')
            param.requires_grad = False


    def init_stride(self, pred, prev_ix):

        stride_remainder = self.S - self.stride_overlap
        expansion = [1 for x in pred.shape]
        expansion[1] = stride_remainder

        first = pred[:, prev_ix:prev_ix + self.stride_overlap, ...]
        last = first[:, -1:, ...].repeat(expansion)
        init = torch.cat((first, last), dim = 1)

        return init

    def get_feature_planes_levels(self, feature_planes, corr_levels=None):

        if corr_levels is None:
            corr_levels = self.corr_levels
        
        B, T, D, V1, V2, R = feature_planes.shape
        feature_planes_levels = []
        
        for i, corr_level in enumerate(range(1, corr_levels + 1)): 

            # downsize image features according to corr level
            if corr_level > 1: 

                feature_planes_scaled = rearrange(
                    feature_planes_scaled, 
                    'b t d v1 v2 r -> (b t) (d r) v1 v2' 
                )

                feature_planes_scaled = F.avg_pool2d(
                    input = feature_planes_scaled, 
                    kernel_size = 2, 
                    stride = 2
                )

                feature_planes_scaled = rearrange(
                    feature_planes_scaled, 
                    '(b t) (d r) v1 v2 -> b t d v1 v2 r', 
                    b = B, t = T, r = R
                )
            else: 
                feature_planes_scaled = feature_planes

            feature_planes_levels.append(feature_planes_scaled)

        return feature_planes_levels
    

    def init_levels(self, coords, feature_planes):

        B, T, D, V1, V2, R = feature_planes.shape
        
        feature_planes_levels = self.get_feature_planes_levels(feature_planes)
        track_features_levels = []
        
        for i in range(1, self.corr_levels + 1): 

            feature_planes_scaled = feature_planes_levels[i]
            track_features = []

            for k, ixs in enumerate(self.plane_ixs):

                tf = self.get_track_features(
                    coords = coords[..., ixs] / 2**i, 
                    feature_planes = feature_planes_scaled[..., k]
                )
                track_features.append(tf)

            feature_planes_levels.append(feature_planes_scaled)
            track_features = torch.stack(track_features, axis = -1)
            track_features_levels.append(track_features)

        return feature_planes_levels, track_features_levels

    
    def get_coord_regions(self, coords):
        
        device = coords.device
        B, _, N, R = coords.shape
        coords = rearrange(coords, 'b 1 n r -> b n 1 1 r')

        offsets = torch.arange(self.corr_dim, device = device) - self.corr_radius

        offset_coords = torch.tensor(list(itertools.product(offsets, offsets)), device = device)
        offset_coords = offset_coords.T.reshape((R - 1, self.corr_dim, self.corr_dim))

        time_offsets = torch.zeros((1, self.corr_dim, self.corr_dim), device = device)
        offset_coords = torch.cat((offset_coords, time_offsets), dim = 0)
        offset_coords = rearrange(offset_coords, 'r ch cw -> ch cw r')

        coord_regions = coords + offset_coords
        coord_regions = rearrange(coord_regions, 'b n ch cw r -> b (ch cw) n r')

        return coord_regions


    def get_track_features(self, coords, feature_planes, 
                           padding_mode = 'border'): 

        device = coords.device
        _, N, R = coords.shape
        B, S, D, V1, V2 = feature_planes.shape

        feature_planes = rearrange(feature_planes, 'b s d vh vw -> b d s vh vw')

        t = torch.arange(B, device = device) * torch.ones((N, B), device = device) 
        t = rearrange(t, 'n b -> b 1 n 1')

        coords = rearrange(coords, 'b n r -> b 1 n r')
        coords = torch.cat((coords, t), dim = -1)
        coord_regions = self.get_coord_regions(coords)
    
        scale, _ = (torch.max(torch.tensor([V2, V1, S], device = device)
                          .unsqueeze(dim = -1) - 1,
                          dim = 1))

        coord_regions = rearrange(coord_regions, 'b corr n r -> b corr n 1 r')
        coord_regions = coord_regions * (2 / scale) - 1

        track_features = F.grid_sample(
            input = feature_planes.float(),
            grid = coord_regions.float(), 
            align_corners = True,
            padding_mode = padding_mode
        )

        track_features = rearrange(track_features, 'b d corr n 1 -> b corr n d')

        return track_features


    def get_corr_features_2d(self, coords, feature_planes, 
                             padding_mode = 'border'): 

        device = coords.device
        _, N, R = coords.shape # B * S, N, R
        B, S, D, V1, V2 = feature_planes.shape

        times = torch.zeros((B * S, N, 1), device = device)
        coords = torch.cat((coords, times), dim = -1)

        coord_regions = self.get_coord_regions(coords.unsqueeze(1))

        coord_regions = rearrange(
            coord_regions, 
            'bs (corr1 corr2) n r -> bs n corr1 corr2 r', 
            corr1 = self.corr_dim, corr2 = self.corr_dim
        )

        scale, _ = (torch.max(torch.tensor([V2, V1, S], device = device)
                          .unsqueeze(dim = -1) - 1,
                          dim = 1))
        coord_regions = coord_regions * (2 / scale) - 1

        feature_planes = rearrange(
            feature_planes, 
            'b s d vh vw -> (b s) d 1 vh vw'
        )

        corr_features = F.grid_sample(
            input = feature_planes.float(),
            grid = coord_regions.float(), 
            align_corners = True,
            padding_mode = padding_mode
        )

        corr_features = rearrange(
            corr_features, 
            '(b s) d n corr1 corr2 -> b s n corr1 corr2 d', 
            b = B, s = S, corr1 = self.corr_dim, corr2 = self.corr_dim
        )

        return corr_features

    
    def get_corr_features_4d(self, coords, feature_planes, track_features): 

        corr_features_2d = self.get_corr_features_2d(
            coords = coords, 
            feature_planes = feature_planes
        ) 
        
        track_features = rearrange(
            track_features, 
            'b (ch cw) n d -> b n d ch cw', 
            ch = self.corr_dim, 
            cw = self.corr_dim
        )

        corr_features_4d = torch.einsum(
            'bsnhwd,bndxy->bsnhwxy', 
            corr_features_2d, 
            track_features
        )

        corr_features_4d = rearrange(
            corr_features_4d, 
            'b s n ch cw cx cy -> (b s n) (ch cw cx cy)'
        )
                                    
        return corr_features_4d

    # @torch.compile
    def get_corr_features(self, coords, feature_planes_levels, track_features_levels): 

        B, S, N, R = coords.shape
        corr_features_levels = []

        for i in range(self.corr_levels):

            corr_features = []

            for j, ixs in enumerate(self.plane_ixs):
                
                coords_scaled = rearrange(
                        coords[..., ixs] / 2**i, 
                        'b s n r -> (b s) n r'
                )

                corr_features_4d = self.get_corr_features_4d(
                    coords = coords_scaled, 
                    feature_planes = feature_planes_levels[i][..., j], 
                    track_features = track_features_levels[i][..., j]
                )

                corr_features.append(self.corr_mlp(corr_features_4d))

            # stack xy, xz, and yz correlation features
            cf = torch.stack(corr_features, dim = -1)
            cf = rearrange(cf, 'bsn d sr -> bsn (d sr)')
            corr_features_levels.append(cf)

        corr_features = rearrange(
            torch.cat(corr_features_levels,  dim = -1), 
            '(b s n) d -> b s n d', 
            b = B, s = S, n = N
        )
        
        return corr_features

    # @torch.compile
    def sample_feature_cubes(self, feature_planes, camera_group,
                             cube_centers, cube_interval,
                             corr_radius = None, downsample_ratio = None,
                             v2v = None, att_net = None, query = None):
        """Inputs:
         feature_planes: cams bt d h w
         cube_centers: bt k 3
         cube_interval: single float
        
        Returns:
          volume: bt d k total
        """
        if corr_radius is None:
            corr_radius = self.corr_radius
        if downsample_ratio is None:
            downsample_ratio = self.downsample_factor
        if v2v is None:
            v2v = self.minicube_v2v
            
        cube_size = corr_radius * 2 + 1
        n_cams = len(feature_planes)
        BT, K, _ = cube_centers.shape
        
        # get coordinates of each cube
        row = (torch.arange(cube_size) - corr_radius) * cube_interval
        xyz_s = torch.stack(torch.meshgrid(row, row, row, indexing='ij'))
        xyz = rearrange(xyz_s, 'r x y z -> (x y z) r')
        xyz = xyz.contiguous().to(device = cube_centers.device, dtype = cube_centers.dtype)
        # xyz = torch.as_tensor(xyz, device=cube_centers.device, dtype=cube_centers.dtype)    
    
        cube_coords = cube_centers[..., None, :] + xyz
        cube_coords_flat = rearrange(cube_coords, 'bt k total r -> (bt k total) r')
        p2d_flat = project_points_torch(
            camera_group = camera_group, 
            coords_3d = cube_coords_flat, 
            downsample_factor = downsample_ratio,
        )
    
        p2d = rearrange(p2d_flat, 'ncams (bt k total) r -> ncams bt k total r',
                        bt=BT, k=K)
    
        all_samples = []
        all_masks = []
        for ix_cam in range(n_cams):
            bt, d, h, w = feature_planes[ix_cam].shape
            scale = torch.tensor([w, h], device = p2d.device) 
            p2d_scaled = 2 * p2d[ix_cam] / scale - 1

            # Create visibility mask: True if within [-1, 1] bounds
            valid_mask = ((p2d_scaled[..., 0] >= -1) & (p2d_scaled[..., 0] <= 1) &
                          (p2d_scaled[..., 1] >= -1) & (p2d_scaled[..., 1] <= 1))
            
            samples = F.grid_sample(
                input=feature_planes[ix_cam],
                grid=p2d_scaled,
                align_corners=False,
                padding_mode="zeros")
            all_samples.append(samples)
            all_masks.append(valid_mask)

        # volumes: cams bt d k total            
        volumes = torch.stack(all_samples)
        masks = torch.stack(all_masks)  # ncams bt k total

        # if att_net is not None:
        #     mean_volume = att_net(volumes, masks, query)
        # else:
        # masks_expanded = masks.unsqueeze(2)  # ncams bt 1 k total
        masks_expanded = repeat(masks, "ncams bt k total -> ncams bt d k total",
                                d = volumes.shape[2])
        # valid_counts = masks_expanded.sum(dim=0).clamp(min=1)  # Avoid div by zero
        # mean_volume = volumes.sum(dim=0) / valid_counts
        # mean_volume = torch.mean(volumes, dim=0)

        # apply softmax from learnable triangulation
        volumes[~masks_expanded] = -1e3
        weights = F.softmax(volumes, dim=0)
        mean_volume = torch.sum(volumes * weights, dim=0)

        mv_flat = rearrange(mean_volume, 'bt d k (x y z) -> (bt k) d z y x',
                            x=cube_size, y=cube_size, z=cube_size)
        mv_flat = v2v(mv_flat)

        mv_flat = F.normalize(mv_flat, p=2, dim=1, eps=1e-6)
        
        mean_volume = rearrange(mv_flat, '(bt k) d z y x -> bt d k (x y z)',
                                bt=BT, k=K) 

        # # # normalize features
        # mv_norms = reduce(torch.square(mean_volume),
        #                   'bt d k total -> bt 1 k total', 'sum')

        # # handle 0 norm case
        # mv_norms = torch.sqrt(torch.maximum(
        #     mv_norms, torch.tensor(
        #         1e-6, device=mv_norms.device, dtype=mv_norms.dtype)))
        
        # mean_volume_normed = mean_volume / mv_norms
        
        # return mean_volume_normed
        return mean_volume
        
    
    def get_corr_features_minicubes(self, coords, feature_planes_levels, 
                                    track_features_levels, camera_group):
        # for each scale
        # sample_feature_cubes to get cubes from feature planes at coords
        # einsum for correlations
        B, S, D, H, W, R = feature_planes_levels[0][0].shape
        B, S, N, R = coords.shape

        coords_bs = rearrange(coords, 'b s n r -> (b s) n r')

        corr_features_levels = []
        for i in range(self.corr_levels):

            feature_planes = [ffl[i] for ffl in feature_planes_levels]
            feature_planes_bs = [rearrange(f, 'b s d h w 1 -> (b s) d h w')
                                 for f in feature_planes]
            
            cube_interval = self.cube_scale * (2**i)
            downsample_ratio = self.downsample_factor * (2**i)

            # query = repeat(track_features_levels[i], 'b t n d -> (b s) d n t', s = S)
                
            mv = self.sample_feature_cubes(
                feature_planes_bs,
                camera_group,
                coords_bs,
                cube_interval,
                corr_radius=self.corr_radius,
                downsample_ratio=downsample_ratio,
                v2v = self.minicube_v2v[i],
                # att_net = self.view_attention[i],
                # query = None
            )

            mv = rearrange(mv, '(b s) d n total -> b s total n d',
                           b=B, s=S)

            corr_features = einsum(mv, track_features_levels[i],
                                   'b s t1 n d, b t2 n d -> b s n t1 t2')
            
            
            corr_features = rearrange(corr_features,
                                      'b s n t1 t2 -> (b s n) (t1 t2)')

            mlp_corr_features = self.corr_mlp(corr_features) 
            
            corr_features_levels.append(mlp_corr_features)

        corr_features = rearrange(
            torch.cat(corr_features_levels,  dim = -1), 
            '(b s n) d -> b s n d', 
            b = B, s = S, n = N
        )
        return corr_features

    
    def get_track_features_minicubes(self, coords, feature_planes_levels,
                                     camera_group): 

        track_features_levels = []
        for i in range(self.corr_levels):
            feature_planes = [ffl[i] for ffl in feature_planes_levels]
            feature_planes_first = [ff[:, 0, ..., 0] for ff in feature_planes]
            cube_interval = self.cube_scale * (2**i)
            downsample_ratio = self.downsample_factor * (2**i)
            track_features_cube = self.sample_feature_cubes(
                feature_planes_first, camera_group, 
                coords, cube_interval,
                corr_radius=self.corr_radius,
                downsample_ratio=downsample_ratio,
                v2v = self.minicube_v2v[i],
                # att_net = self.view_attention[i]
            )
            track_features_cube = rearrange(track_features_cube,
                                            'b d n total -> b total n d')
            
            center = track_features_cube.shape[1] // 2
            track_features_cube = track_features_cube[:, center:center+1]

            track_features_levels.append(track_features_cube)        

        return track_features_levels

    # @torch.compile
    def forward_iteration(self, coords, vis, conf, 
                          feature_planes_levels, track_features_levels,
                          camera_group = None):

        device = coords.device 
        B, S, N, R = coords.shape

        if self.R == 3 and self.mode_3d == 'minicubes':
            B, S, D, V1, V2, R = feature_planes_levels[0][0].shape
        else:
            B, S, D, V1, V2, R = feature_planes_levels[0].shape

        # iterative updates
        coords_pred = []
        vis_pred = []
        conf_pred = []

        for i in range(self.n_iters):

            # correlation features for transformer input
            if self.R == 3 and self.mode_3d == 'minicubes':
                corr_features = self.get_corr_features_minicubes(
                    coords = coords, 
                    feature_planes_levels = feature_planes_levels, 
                    track_features_levels = track_features_levels,
                    camera_group = camera_group, 
                )
            else:
                corr_features = self.get_corr_features(
                    coords = coords, 
                    feature_planes_levels = feature_planes_levels, 
                    track_features_levels = track_features_levels,
                )

            # encodings for transformer input
            forward_flow = torch.diff(coords, dim = 1)
            backward_flow = (torch.diff(coords.flip(dims = [1]), dim = 1)
                                  .flip(dims = [1]))

            forward_flow = F.pad(
                forward_flow, 
                pad = (0, 0, 0, 0, 1, 0), 
                mode = 'constant', 
                value = 0
            ) 

            backward_flow = F.pad(
                backward_flow, 
                pad = (0, 0, 0, 0, 0, 1), 
                mode = 'constant', 
                value = 0
            ) 

            scale = V1 
            if self.R == 2: 
                scale = (torch.tensor([V1, V2, V1, V2])
                              .view(1, 1, 1, -1)
                              .to(device))
            elif self.mode_3d == 'minicubes':
                scale = self.cube_scale * 64

            coord_flow = torch.cat([forward_flow, backward_flow], dim = -1) / scale
            flow_encoding = get_fourier_encoding(coord_flow, min_freq = 0, max_freq = self.max_freq)

            # transformer time
            x = (torch.cat([vis, conf, corr_features, coord_flow, flow_encoding], dim = -1) 
                           + self.time_encoding)  

            updates = self.tsformer(x) # b s n d

            # update coords, vis, and conf
            delta_coords, delta_vis, delta_conf = torch.split(updates, [self.R, 1, 1], dim = -1)
            if self.R == 3 and self.mode_3d == 'minicubes':
                delta_coords = delta_coords * self.cube_scale * 16
                
            # delta_coords[:, 0] = 0 # initial coordinates should not change
            coords = coords + delta_coords # b s n 3
            vis = torch.sigmoid(vis + delta_vis) # b s n 1 
            conf = torch.sigmoid(conf + delta_conf) # b s n 1 
 
            coords_pred.append(coords)
            vis_pred.append(vis) 
            conf_pred.append(conf)
        
        return coords_pred, vis_pred, conf_pred


    def unscale(self, coords): 

        if self.R == 3:
            if self.mode_3d == 'minicubes':
                coords_unscaled = coords
            else:
                scale = (2 * self.cube_extent) / (self.cube_dim * self.upsample_factor)
                coords_unscaled = scale * coords + (self.cube_center - self.cube_extent)
        else: 
            coords_unscaled = coords * self.downsample_factor 

        return coords_unscaled


    def forward(self, views, coords, query_times = None, camera_group = None):
        '''
        B: batch size
        T: number of frames in video
        C: number of channels 
        H: height of image
        W: width of image
        D: latent dimension
        '''
        #TODO: actually handle query times
        
        device = coords.device

        B, N, R = coords.shape
        B, T, H, W, C = views[0].shape

        assert R == self.R

        # so we don't actually change the input
        views = list(views)

        if self.R == 3:
            self.cube_scale = ( get_camera_scale(camera_group, coords.reshape(-1, 3)) * 
                                self.downsample_factor * 2 )

            if self.mode_3d == 'triplane':
                # dynamically compute the cube center and extent based
                # on the coord from the first frame
                self.cube_center = torch.mean(coords, dim = (0, 1))
                # self.cube_extent = cgroup.cube_extent TODO: auto compute this later

                with torch.no_grad():
                    cgroup = UnprojectViews(
                        camera_group = camera_group,
                        cube_center = self.cube_center,
                        cube_extent = self.cube_extent,
                        downsample_factor = self.downsample_factor,
                        cube_dim = self.cube_dim,
                        device = device) 

        # normalize frames
        for i, frames in enumerate(views): 
            # frames = 2 * (frames / 255.0) - 1
            frames = frames.to(device)
            frames = rearrange(frames, 'b t h w c -> b t c h w')
            frames = self.transform_norm(frames)
            views[i] = frames

        # determine number of strides
        stride_remainder = self.S - self.stride_overlap
        n_windows = ((T - stride_remainder - 1) // stride_remainder) + 1
        n_pad = (self.S - T % self.S) % self.S

        # frames for each camera view
        if n_pad > 0: 
            for i, frames in enumerate(views):
                last_frame = frames[:, -1:, ...].repeat((1, n_pad, 1, 1, 1))
                frames = torch.cat((frames, last_frame), dim = 1)
                views[i] = frames

        # extract image features and downsize
        feature_maps = []

        for i, frames in enumerate(views):

            frames = rearrange(frames, 'b t c h w -> (b t) c h w')

            if self.R == 3 and self.mode_3d == 'minicubes' and self.cnn_type == 'sam2':
                feature_map_levels_flat = self.cnn(frames, return_all = True)
                ff = [rearrange(f, '(b t) d h2 w2 -> b t d h2 w2 1', b = B, t = T + n_pad)
                    for f in feature_map_levels_flat]
                
            else:
                feature_map = self.cnn(frames)
                _, D, H2, W2 = feature_map.shape
                ff = rearrange(feature_map, 
                               '(b t) d h2 w2 -> b t d h2 w2',
                               b = B, t = T + n_pad)

            if self.R == 2:
                # normalize feature maps for correlation
                ff_norms_sq = reduce(torch.square(ff), 'b s d h w -> b s 1 h w', 'sum')

                # handle 0 norm case
                ff_norms = torch.sqrt(
                    torch.maximum(
                        ff_norms_sq,
                        torch.tensor(1e-6, device=ff.device, dtype=ff.dtype))
                )

                ff = ff / ff_norms
            
            feature_maps.append(ff)

        if self.R == 3: 

            if self.mode_3d == 'triplane':

                # unproject views into volumes, normalize, then get average volume
                volumes = cgroup.unproject_to_volume(feature_maps) # (G, B, S, D, V1, V2, V3)
                volumes = F.normalize(volumes, p = 2, dim = 3, eps = 1e-6)
                volumes_avg = torch.mean(volumes, dim = 0) 

                # project volumes to get triplanes 
                xy_planes, xz_planes, yz_planes = project_volumes(volumes_avg)
                planes = torch.cat((xy_planes, xz_planes, yz_planes), dim = -3) 

                # extract features from planes 
                feature_planes = self.triplane_cnn(
                    rearrange(planes, 'b t d3 v1 v2 -> (b t) d3 v1 v2')
                )

                feature_planes = rearrange(
                    feature_planes, 
                    '(b t) (d r) v1 v2 -> b t d v1 v2 r', 
                    b = B, t = T + n_pad, r = 3
                )

                # normalize the triplanes 
                feature_planes = F.normalize(feature_planes, p = 2, dim = 2, eps = 1e-6)
                
                coords = ((coords - (self.cube_center - self.cube_extent)) *
                                    (self.cube_dim / (2 * self.cube_extent)) * 
                                     self.upsample_factor)
                
            elif self.mode_3d == 'minicubes':
                feature_planes = feature_maps

        else: # 2d case, no triplane
            coords = coords / self.downsample_factor
            feature_planes = rearrange(feature_maps, '1 b t d h w -> b t d h w 1')

        # initialize feature planes and track features for each correlation level
        
        if self.R == 3 and self.mode_3d == 'minicubes':

            feature_planes_levels = feature_planes
            # feature_planes_levels = [
            #     self.get_feature_planes_levels(rearrange(f, 'b s d h w -> b s d h w 1'))
            #     for f in feature_planes]
            track_features_levels = self.get_track_features_minicubes(
                coords, feature_planes_levels, camera_group
            )
        else:
            (feature_planes_levels, track_features_levels) = self.init_levels(
                coords = coords, 
                feature_planes = feature_planes)

        
        # track final predictions
        coords_pred = torch.zeros((B, T, N, R), device = device)
        vis_pred = torch.zeros((B, T, N, 1), device = device)
        conf_pred = torch.zeros((B, T, N, 1), device = device)

        # track iterative predictions
        coords_pred_iters = []
        vis_pred_iters = []
        conf_pred_iters = []

        for i in range(n_windows): 

            ix = stride_remainder * i
            prev_ix = ix - self.stride_overlap

            if i == 0:
                # initialize coords with the first frame in the window 
                # and the visibility of all tracks to 1 
                coords_init = coords.unsqueeze(1).repeat((1, self.S, 1, 1))
                vis_init = torch.zeros((B, self.S, N, 1), device = device)
                conf_init = torch.zeros((B, self.S, N, 1), device = device)

            else: 
                # copy overlapping coords, vis, and conf predictions from the
                # previous window to the the first frames in the current window 
                coords_init = self.init_stride(coords_pred, prev_ix)
                vis_init = self.init_stride(vis_pred, prev_ix)
                conf_init = self.init_stride(conf_pred, prev_ix)

            # get the feature planes and track features for the current stride
            if self.R == 3 and self.mode_3d == 'minicubes':
                # multicams
                feature_plane_levels_subset = [
                    [fp[:, ix:ix + self.S, ...] for fp in fp_cams]
                    for fp_cams in feature_planes_levels
                ]
            else:
                feature_plane_levels_subset = [fp[:, ix:ix + self.S, ...] 
                                               for fp in feature_planes_levels]

            # iterative updates with transformer
            (coords_pred_updates, 
            vis_pred_updates, 
            conf_pred_updates) = self.forward_iteration(
                coords = coords_init, 
                vis = vis_init,
                conf = conf_init,
                feature_planes_levels = feature_plane_levels_subset, 
                track_features_levels = track_features_levels,
                camera_group = camera_group, 
            )

            # remove excess padding and store final iteration
            if i == n_windows - 1 and n_pad > 0:
                coords_pred_updates = [update[:, :-n_pad, ...] for update in coords_pred_updates]
                vis_pred_updates = [update[:, :-n_pad, ...] for update in vis_pred_updates]
                conf_pred_updates = [update[:, :-n_pad, ...] for update in conf_pred_updates]

            # store final updates
            coords_pred[:, ix:ix + self.S, ...] = coords_pred_updates[-1] 
            vis_pred[:, ix:ix + self.S, ...] = vis_pred_updates[-1]
            conf_pred[:, ix:ix + self.S, ...] = conf_pred_updates[-1]

            # store all iterative updates
            coords_pred_iters.append([self.unscale(coords) for coords in coords_pred_updates])
            vis_pred_iters.append(vis_pred_updates)
            conf_pred_iters.append(conf_pred_updates)

        # adjust coordinates to match original scale
        coords_pred = self.unscale(coords_pred)

        # assemble outputs 
        result_dict = {
            'coords_pred': coords_pred,
            'vis_pred': vis_pred,
            'conf_pred': conf_pred, 
            'feature_planes_levels': feature_planes_levels}

        if self.training: 
            train_dict = {
                'coords_pred_iters': coords_pred_iters,
                'vis_pred_iters': vis_pred_iters, 
                'conf_pred_iters': conf_pred_iters}
            
            result_dict.update(train_dict)

        return result_dict 


    # @torch.compile
    def get_feature_loss(self, feature_planes_levels, coords_full, 
                         camera_group = None):

        coords = coords_full[:, 0] # first frame coords
        B, S, N, R = coords_full.shape        
        coords_bs = rearrange(coords_full, 'b s n r -> (b s) n r')

        if self.mode_3d == 'minicubes': 

            B, S, D, H, W, R = feature_planes_levels[0][0].shape
            corr_features_levels = []

            for i in range(self.corr_levels):
                feature_planes = [ffl[i] for ffl in feature_planes_levels]
                feature_planes_bs = [rearrange(f, 'b s d h w 1 -> (b s) d h w')
                                    for f in feature_planes]

                cube_interval = self.cube_scale * (2**i)
                downsample_ratio = self.downsample_factor * (2**i)
                
                mv = self.sample_feature_cubes(
                    feature_planes_bs,
                    camera_group,
                    coords_bs,
                    cube_interval,
                    corr_radius=self.corr_radius,
                    downsample_ratio=downsample_ratio,
                    v2v = self.minicube_v2v[i],
                    # att_net = self.view_attention[i]
                )

                mv = rearrange(mv, '(b s) d n total -> b s total n d', b = B, s = S)

                # only take the center
                center = mv.shape[2] // 2
                mv = mv[:, :, center]
                
                tv = mv[:, 0]
            
                # corr_features = einsum(mv, tv, 'b s t n d, b t n d -> b s n t')
                corr_features = einsum(mv, tv, 'b s n d, b n d -> b s n')
                corr_features_levels.append(torch.mean(corr_features))

            mean_corr = sum(corr_features_levels) / len(corr_features_levels)

        else: # self.mode_3d == 'triplanes':

            B, S, D, H, W, R = feature_planes_levels[0].shape
            corr_features_levels = []

            for i in range(self.corr_levels):

                corr_features_triplanes = []

                for j, ixs in enumerate(self.plane_ixs):
                    
                    coords_scaled = coords_bs[..., ixs] / 2**i

                    corr_features_2d = self.get_corr_features_2d(
                        coords = coords_scaled, 
                        feature_planes = feature_planes_levels[i][..., j]
                        )

                    # only take the center
                    corr_features_2d = corr_features_2d[:, :, :,
                                                        self.corr_radius,
                                                        self.corr_radius]

                    track_features = corr_features_2d[:, 0]

                    # corr_features = torch.einsum(
                    #     'bsnxyd,bnxyd->bsnxy', 
                    #     corr_features_2d, 
                    #     track_features
                    # )
                    corr_features = torch.einsum(
                        'bsnd,bnd->bsn', 
                        corr_features_2d, 
                        track_features
                    )

                    corr_features_triplanes.append(corr_features)

                # stack xy, xz, and yz correlation features, then average
                # average over xy, xz, yz
                corr_features_triplanes = torch.stack(corr_features_triplanes)
                cf = torch.mean(corr_features_triplanes, dim=0)
                corr_features_levels.append(torch.mean(cf))

            mean_corr = sum(corr_features_levels) / len(corr_features_levels)
        
        return 1 - mean_corr
