import torch
import torch.nn as nn 
import torch.nn.functional as F

from posetail.posetail.utils import init_kwargs


class ResUnit(nn.Module):

    def __init__(self, input_dim, output_dim, 
                 kernel_size = 3, spatial_res_factor = 1,
                 act_class = nn.ReLU, norm_class = nn.InstanceNorm2d,
                 act_kwargs = None, norm_kwargs = None):
        super().__init__()

        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.spatial_res_factor = spatial_res_factor

        self.act_class = act_class 
        self.norm_class = norm_class 
        self.act_kwargs = init_kwargs(act_kwargs)
        self.norm_kwargs = init_kwargs(norm_kwargs)

        self.conv1_branch1 = nn.Conv2d(
            in_channels = self.input_dim, 
            out_channels = self.output_dim, 
            kernel_size = self.kernel_size, 
            stride = self.spatial_res_factor, 
            padding = self.padding
        )

        self.norm1_branch1 = self.norm_class(num_features = self.output_dim,
                                             **self.norm_kwargs)

        self.act = self.act_class(**self.act_kwargs)

        self.conv2_branch1 = nn.Conv2d(
            in_channels = self.output_dim, 
            out_channels = self.output_dim, 
            kernel_size = self.kernel_size, 
            stride = 1, 
            padding = self.padding
        )

        self.norm2_branch1 = self.norm_class(num_features = self.output_dim, 
                                            **self.norm_kwargs)

        if self.spatial_res_factor > 1: 

            self.conv1_branch2 = nn.Conv2d(
                in_channels = self.input_dim, 
                out_channels = self.output_dim, 
                kernel_size = 1,
                stride = self.spatial_res_factor 
            )

            self.norm1_branch2 = self.norm_class(num_features = self.output_dim,
                                                 **self.norm_kwargs)


    def forward(self, x): 
        
        x_res = x
        
        x = self.act(self.norm1_branch1(self.conv1_branch1(x)))
        x = self.act(self.norm2_branch1(self.conv2_branch1(x)))

        if self.spatial_res_factor > 1: 
            x_res = self.norm1_branch2(self.conv1_branch2(x_res))

        x = self.act(x + x_res)

        return x


class ResBlock(nn.Module):

    def __init__(self, input_dim, output_dim, n_units = 2,
                 kernel_size = 3, spatial_res_factor = 1,
                 act_class = nn.ReLU, norm_class = nn.InstanceNorm2d,
                 act_kwargs = None, norm_kwargs = None):        
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.spatial_res_factor = spatial_res_factor
        self.n_units = n_units

        self.act_class = act_class
        self.norm_class = norm_class
        self.act_kwargs = init_kwargs(act_kwargs)
        self.norm_kwargs = init_kwargs(norm_kwargs)
        
        self.res_units = self._init_res_units()

    def _init_res_units(self): 

        res_units = nn.ModuleList()
        input_dim = self.input_dim

        for i in range(self.n_units): 

            if i == 0:
                stride = self.spatial_res_factor
            else: 
                stride = 1

            res_unit = ResUnit(
                input_dim = input_dim, 
                output_dim = self.output_dim, 
                kernel_size = self.kernel_size, 
                spatial_res_factor = stride,
                act_class = self.act_class, 
                norm_class = self.norm_class,
                act_kwargs = self.act_kwargs, 
                norm_kwargs = self.norm_kwargs
            )
            res_units.append(res_unit)
            input_dim = self.output_dim

        return res_units

    def forward(self, x): 
        
        for i in range(len(self.res_units)):
            x = self.res_units[i](x)

        return x


class ResidualFeatureExtractor(nn.Module):

    def __init__(self, input_dim, output_dim, n_blocks, 
                downsample_factor = 4, n_units = 2, interp_mode = 'bilinear', 
                kernel_size = 3, spatial_res_factor = 1, 
                act_class = nn.ReLU, norm_class = nn.InstanceNorm2d,
                act_kwargs = None, norm_kwargs = None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_blocks = n_blocks
        self.n_units = n_units
        self.downsample_factor = downsample_factor
        self.interp_mode = interp_mode
        self.kernel_size = kernel_size
        self.spatial_res_factor = spatial_res_factor

        self.act_class = act_class 
        self.norm_class = norm_class
        self.act_kwargs = init_kwargs(act_kwargs)
        self.norm_kwargs = init_kwargs(norm_kwargs)

        self.conv1 = nn.Conv2d(
            in_channels = self.input_dim, 
            out_channels = self.output_dim // 2, 
            kernel_size = 7, 
            stride = 2, 
            padding = 3)

        self.norm1 = self.norm_class(num_features = self.output_dim // 2,
                                    **self.norm_kwargs)
        self.act = self.act_class(**self.act_kwargs)

        # initialize residual blocks
        hidden_dims = [output_dim // 2, 
                       3 * (output_dim // 4), 
                       output_dim, 
                       output_dim]

        self.res_blocks = self._init_res_blocks(
            input_dim = self.output_dim // 2, 
            output_dims = hidden_dims
        )

        self.cat_dim = sum(hidden_dims)

        self.conv2 = nn.Conv2d(
            in_channels = self.cat_dim, 
            out_channels = self.output_dim * 2, 
            kernel_size = 3, 
            stride = 1,
            padding = 1
        )
        self.norm2 = self.norm_class(num_features = self.output_dim * 2,
                                    **self.norm_kwargs)

        self.conv3 = nn.Conv2d(
            in_channels = self.output_dim * 2, 
            out_channels = self.output_dim, 
            kernel_size = 1, 
            stride = 1,
            padding = 0
        )

        self._init_weights()

    def _init_weights(self): 

        for layer in self.modules():

            if isinstance(layer, nn.Conv2d): 
                nn.init.kaiming_uniform_(layer.weight, 
                    mode = 'fan_out', nonlinearity = 'relu')
            
            elif isinstance(layer, nn.InstanceNorm2d): 
                if layer.weight is not None and layer.bias is not None: 
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def _init_res_blocks(self, input_dim, output_dims): 

        assert len(output_dims) == self.n_blocks
        res_blocks = nn.ModuleList()

        for i in range(self.n_blocks): 

            if i == 0:
                stride = 1
            else: 
                stride = self.spatial_res_factor

            res_block = ResBlock(
                input_dim = input_dim, 
                output_dim = output_dims[i], 
                spatial_res_factor = stride,
                act_class = self.act_class, 
                norm_class = self.norm_class,
                **self.act_kwargs, 
                **self.norm_kwargs
            )
            input_dim = output_dims[i]
            res_blocks.append(res_block)
            
        return res_blocks

    def resize(self, feature_maps, res):

        spatial_res = (res[0] // self.downsample_factor, 
                       res[1] // self.downsample_factor)

        feature_maps = F.interpolate(
            input = feature_maps, 
            size = spatial_res,
            mode = self.interp_mode, 
            align_corners = True
        )

        return feature_maps

    def forward(self, x): 

        target_res = x.shape[-2:]

        x = self.act(self.norm1(self.conv1(x)))

        x_scales = []

        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
            x_resized = self.resize(x, res = target_res)
            x_scales.append(x_resized)

        x = torch.cat(x_scales, dim = 1)

        x = self.act(self.norm2(self.conv2(x)))
        x = self.conv3(x)

        return x



class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels = 128, 
                 stride = 1, downsample_factor = 2):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = 64,
            kernel_size = 3, 
            stride = stride, 
            padding = 1
        )

        self.pool = nn.MaxPool2d(
            kernel_size = downsample_factor, 
            stride = downsample_factor
        )

        self.conv2 = nn.Conv2d(
            in_channels = 64, 
            out_channels = out_channels,
            kernel_size = 3, 
            stride = 1, 
            padding = 1
        )

        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self): 

        for layer in self.modules():

            if isinstance(layer, nn.Conv2d): 
                nn.init.kaiming_uniform_(layer.weight, 
                    mode = 'fan_out', nonlinearity = 'relu')
            
            elif isinstance(layer, nn.InstanceNorm2d): 
                if layer.weight is not None and layer.bias is not None: 
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)

        return x


from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict
from einops import rearrange, reduce

class HieraFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128, pretrained_model='facebook/hiera_base_224.mae_in1k'):
        super().__init__()

        # Lazy import: hiera is only needed by the legacy Tracker path.
        # Install via `pip install posetail[legacy-tracker]`.
        try:
            import hiera
        except ImportError as e:
            raise ImportError(
                "HieraFeatureExtractor requires the 'hiera-transformer' package "
                "(legacy Tracker only). Install it with: pip install posetail[legacy-tracker]"
            ) from e

        self.model = hiera.Hiera.from_pretrained(pretrained_model)

        # Ensure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Remove unused Hiera head components
        self.model.head = nn.Identity()
        self.model.norm = nn.Identity()
            
        with torch.no_grad():  # Don't build computation graph during init
            test_input = torch.randn([1, 3, 224, 224])
            _, intermediates = self.model(test_input, return_intermediates=True)
        hiera_channels = [x.shape[-1] for x in intermediates]

        self.fpn = FeaturePyramidNetwork(hiera_channels, output_dim)

        # Freeze unused FPN layer_blocks (keep only layer_blocks[0])
        for name, param in self.fpn.named_parameters():
            if 'layer_blocks' in name and not name.startswith('layer_blocks.0'):
                param.requires_grad = False

    def forward(self, x):
        _, intermediates = self.model(x, return_intermediates=True)
        features = OrderedDict()
        for i, x in enumerate(intermediates):
            features[i] = rearrange(x, 'b h w c -> b c h w')
        out = self.fpn(features)
        return out[0]

class SAM2HieraFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128, pretrained_model="facebook/sam2.1-hiera-small",
                 requires_grad=False, freeze_nonlast_fpn=True):
        super().__init__()

        # Lazy import: sam2 is only needed by the legacy Tracker path.
        # Install via `pip install posetail[legacy-tracker]`.
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "SAM2HieraFeatureExtractor requires the 'sam-2' package "
                "(legacy Tracker only). Install it with: pip install posetail[legacy-tracker]"
            ) from e

        predictor = SAM2ImagePredictor.from_pretrained(pretrained_model)
        self.model = predictor.model.image_encoder.trunk
        
        for param in self.model.parameters():
            param.requires_grad = requires_grad
            device = param.device

  
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            # nn.GELU(),
            # nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, padding=1, stride=1)
        )
            
        # with torch.no_grad():  # Don't build computation graph during init
        #     test_input = torch.randn([1, 3, 400, 400]).to(device)
        #     intermediates = self.model(test_input)
        # hiera_channels = [x.shape[-1] for x in intermediates]

        if pretrained_model == 'facebook/sam2.1-hiera-base-plus':
            hiera_channels = [112, 224, 448, 896]
        elif pretrained_model in ['facebook/sam2.1-hiera-small', 'facebook/sam2.1-hiera-tiny']:
            hiera_channels = [96, 192, 384, 768]
            
        self.fpn = FeaturePyramidNetwork([32] + hiera_channels, output_dim)

        if freeze_nonlast_fpn:
            # Freeze unused FPN layer_blocks (keep only layer_blocks[0])
            for name, param in self.fpn.named_parameters():
                if 'layer_blocks' in name and not name.startswith('layer_blocks.0'):
                    param.requires_grad = False


        # self.stem_s2 = nn.Sequential(
        #     nn.Conv2d(output_dim // 4, output_dim, kernel_size=3, padding=1, stride=2),
        #     # nn.GELU()
        # )

        # self.upsample_blocks = nn.ModuleList([
        #     nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                   nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        #                   nn.GELU())
        #     for _ in range(1)
        # ])
        
        # self.fuse_blocks = nn.ModuleList([
        #     nn.Sequential(nn.Conv2d(output_dim * 2, output_dim, kernel_size=3, padding=1),
        #                   nn.GroupNorm(8, output_dim),
        #                   nn.GELU())
        #     for _ in range(1)
        # ])
        
        self._init_new_layers()

    def _init_new_layers(self):
        # Initialize only the new conv layers
        # for m in [self.stem_s1, self.stem_s2]: # + list(self.upsample_blocks) + list(self.fuse_blocks):
        for layer in self.stem.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None: nn.init.constant_(layer.bias, 0)

    def forward(self, inp, return_all=False):
        raw_s2 = self.stem(inp)
        intermediates = self.model(inp)
        
        features = OrderedDict()
        features[0] = raw_s2
        for i, x in enumerate(intermediates):
            features[i+1] = x
        out = self.fpn(features)

        # raw_s1 = self.stem_s1(inp)
        # raw_s2 = self.stem_s2(raw_s1)

        # raw_s2 = self.stem(inp)
        # raw_s2 = self.stem_s2(raw_s1)

        
        # up_s2 = self.upsample_blocks[0](out[0])
        # feat_s2 = self.fuse_blocks[0](torch.cat([up_s2, raw_s2], dim=1))

        # up_s1 = self.upsample_blocks[1](feat_s2)
        # feat_s1 = self.fuse_blocks[1](torch.cat([up_s1, raw_s1], dim=1))
        
        if return_all:
            # return [feat_s1, feat_s2] + list(out.values())
            # return [feat_s2] + list(out.values())
            # return [raw_s2] + list(out.values())
            return list(out.values())
        else:
            return out[0]

class VJEPAFeatureUpsampler(nn.Module):
    def __init__(
        self, 
        in_dim=4096, 
        out_dim=256,
        num_spatial_ups=3,
        patch_size=16,
        tubelet_size=2,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        
        self.channel_reduce = nn.Linear(in_dim, out_dim*2)  # operates on last dim, memory efficient
        self.temporal_up = nn.ConvTranspose1d(out_dim*2, out_dim, kernel_size=tubelet_size, stride=tubelet_size)

        self.out_dims = [out_dim] + [out_dim//2 for _ in range(num_spatial_ups)]
        
        self.spatial_ups = nn.ModuleList()
        for i in range(num_spatial_ups):
            self.spatial_ups.append(nn.Conv2d(self.out_dims[i], self.out_dims[i+1],
                                              kernel_size=3, padding=1))
                
        
        self.act = nn.SiLU()
    
    def forward(self, x, T_orig, H_orig, W_orig):
        T_patches = T_orig // self.tubelet_size
        H_patches = H_orig // self.patch_size
        W_patches = W_orig // self.patch_size
        B = x.shape[0]
        
        # Channel reduce first (on flat sequence)
        x = self.channel_reduce(x)  # (B, N, 4096) -> (B, N, 256)
        
        # Temporal upsample
        x = rearrange(x, 'b (t h w) c -> (b h w) c t', t=T_patches, h=H_patches, w=W_patches)
        x = self.temporal_up(x)
        x = rearrange(x, '(b h w) c t -> (b t) c h w', b=B, h=H_patches, w=W_patches)
        T = T_patches * self.tubelet_size
        
        outputs = [rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)]
        
        for up in self.spatial_ups:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.act(up(x))
            outputs.append(rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T))
        
        return outputs        
    
from posetail.posetail.vjepa2 import (
    vjepa2_1_vit_base_384,
    vjepa2_1_vit_large_384,
    vjepa2_1_vit_giant_384,
    vjepa2_1_vit_gigantic_384,
)

class VJEPAFeatureExtractor(nn.Module):
    def __init__(self, output_dim=256, version='large',
                 requires_grad=False):
        super().__init__()
        
        if version == 'base':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_base_384()
        elif version == 'large':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_large_384()
        elif version == 'giant':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_giant_384()
        elif version == 'gigantic':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_gigantic_384()

        self.encoder = vjepa_encoder
        # self.encoder.return_hierarchical = True

        for param in self.encoder.parameters():
            param.requires_grad = requires_grad
            device = param.device

        self.upsampler = VJEPAFeatureUpsampler(
            in_dim = self.encoder.embed_dim,
            out_dim = output_dim,
            num_spatial_ups = 3,
            patch_size = self.encoder.patch_size,
            tubelet_size = self.encoder.tubelet_size
        )

        self.out_dims = list(reversed(self.upsampler.out_dims))
        
    def forward(self, x, return_all=True):
        B, T, C, H, W = x.shape
        xr = rearrange(x, 'b t c h w -> b c t h w')
        features = self.encoder(xr)
        outputs = self.upsampler(features, T, H, W)
        if return_all:
            return list(reversed(outputs))
        else:
            return outputs[-1]
        
            
class TriplaneFeatureExtractor(nn.Module):

    def __init__(self, input_dim, n_hidden_layers, 
                 kernel_size = 3, padding = 1, 
                 upsample_factor = 4):
        super().__init__()

        self.upsample_factor = upsample_factor

        if self.upsample_factor > 1: 
            self.upsample = nn.Upsample(
                scale_factor = self.upsample_factor, 
                mode = 'bilinear',
                align_corners = True
            )

        # hidden layers
        self.hidden_conv_layers = nn.ModuleList()
        self.gn_layers = nn.ModuleList()

        for i in range(n_hidden_layers): 

            hidden_layer = nn.Conv2d(
                in_channels = input_dim, 
                out_channels = input_dim, 
                kernel_size = kernel_size,
                padding = padding
            )

            self.hidden_conv_layers.append(hidden_layer)

            gn = nn.GroupNorm(num_groups = 8, num_channels = input_dim)
            self.gn_layers.append(gn)

        self.relu = nn.ReLU()

        self.conv_out = nn.Conv2d(
            in_channels = input_dim, 
            out_channels = input_dim, 
            kernel_size = kernel_size, 
            padding = padding
        )

        self._init_weights()

    def _init_weights(self): 

        for layer in self.modules():

            if isinstance(layer, nn.Conv2d): 
                nn.init.kaiming_uniform_(layer.weight, 
                    mode = 'fan_out', nonlinearity = 'relu')

    def forward(self, x):

        if self.upsample_factor > 1:
            x = self.upsample(x)

        # for conv in self.hidden_conv_layers: 
        #     x = self.relu(conv(x))

        for gn, conv in zip(self.gn_layers, self.hidden_conv_layers): 
            x = self.relu(gn(conv(x)))

        x = self.conv_out(x)
        
        return x


class FeatureUpdater(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
 
        self.latent_dim = latent_dim

        self.norm = nn.GroupNorm(
            num_groups = 1,
            num_channels = self.latent_dim
        )

        self.fc1 = nn.Linear(
            in_features = latent_dim, 
            out_features = latent_dim
        )

        self.gelu = nn.GELU()

    def forward(self, x): 

        x = self.norm(x)
        x = self.fc1(x)
        x = self.gelu(x)

        return x

    
# from learnable triangulation
class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_planes),
            # nn.BatchNorm3d(out_planes),
            nn.ReLU(),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_planes),
            # nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(num_groups=8, num_channels=out_planes),
                # nn.BatchNorm3d(out_planes)
            )
        self.last_relu = nn.ReLU()

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        out = self.last_relu(res + skip)
        return out


class MinicubesV2V(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.res1 = Res3DBlock(latent_dim, latent_dim)
        self.conv_out = nn.Conv3d(latent_dim, latent_dim,
                                  kernel_size=3, stride=1, padding=1)
        # self.res2 = Res3DBlock(latent_dim, latent_dim)
        # self.skip1 = Res3DBlock(latent_dim, latent_dim)

        self._initialize_weights()
        
    def forward(self, x):
        # s1 = self.skip1(x)
        identity = x
        x = self.res1(x)
        # x = self.res2(x)
        # x = x + s1
        x = self.conv_out(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

class EmbedV2V(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_dim, latent_dim // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(latent_dim // 2, latent_dim, kernel_size=3, padding=1)

        self._initialize_weights()
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

                
class SimpleV2V(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1)

        self._initialize_weights()
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

class SimplerV2V(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1)

        self._initialize_weights()
        
    def forward(self, x):
        return self.conv1(x) + x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                

                
class ViewAttentionV2V(nn.Module):
    def __init__(self, latent_dim, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = latent_dim // 2

        # A small 3D CNN to determine how "good" the features look geometrically
        self.score_net = nn.Sequential(
            nn.Conv3d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, 1, kernel_size=3, padding=1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, volumes, masks):
        """
        Args:
            volumes: (n_cams, bt, d, k, total) - Raw sampled features
            masks: (n_cams, bt, k, total) - Boolean mask (True if inside image bounds)
        Returns:
            aggregated_volume: (bt, d, k, total)
        """
        n_cams, bt, d, k, total = volumes.shape
        # Infer spatial cube size (e.g. 27 -> 3x3x3)
        cube_size = int(round(total ** (1/3)))

        # 1. Reshape for 3D CNN: Treat (n_cams * bt * k) as batch
        # Input: [n_cams, bt, d, k, total] -> [(n_cams * bt * k), d, z, y, x]
        x = rearrange(volumes, 'nc bt d k (z y x) -> (nc bt k) d z y x', 
                      z=cube_size, y=cube_size, x=cube_size)

        # 2. Predict Scores (logits)
        # Output: [(n_cams * bt * k), 1, z, y, x]
        scores = self.score_net(x)

        # 3. Reshape back to isolate n_cams
        # -> [n_cams, bt, 1, k, total]
        scores = rearrange(scores, '(nc bt k) 1 z y x -> nc bt 1 k (z y x)', 
                           nc=n_cams, bt=bt)
 
        # 4. Masking:
        # If the point projected outside the image (mask=False), set score to -infinity
        # so Softmax results in 0 weight.
        masks_expanded = masks.unsqueeze(2) # [nc, bt, 1, k, total]
        scores = scores.masked_fill(~masks_expanded, -1e9)

        # 5. Attention (Softmax over cameras)
        # weights: [n_cams, bt, 1, k, total]
        attn_weights = F.softmax(scores, dim=0)

        # 6. Weighted Sum Aggregation
        # Sum over n_cams dimension
        aggregated_volume = (volumes * attn_weights).sum(dim=0)

        return aggregated_volume

class QueryViewAttentionV2V(nn.Module):
    def __init__(self, latent_dim, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = latent_dim // 2

        # Mode 1: Salience Network (Used when Query is None)
        self.salience_net = nn.Sequential(
            nn.Conv3d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, 1, kernel_size=3, padding=1)
        )
        
        # Mode 2: Query Projection (Optional linear layer to align spaces)
        # We assume query and volumes are in same space, so we can be parameter-free
        # or add a small transform here.
        self.query_proj = nn.Identity() 

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, volumes, masks, query=None):
        """
        Args:
            volumes: (nc, bt, d, k, total) - Candidate features from all cams
            masks:   (nc, bt, k, total) - Valid bounds mask
            query:   (bt, d, k, total) OR (bt, d, k, 1) - The target feature we are tracking
                     If None, runs in 'Salience Mode'.
        """
        n_cams, bt, d, k, total = volumes.shape
        cube_size = int(round(total ** (1/3)))

        # --- SCORE COMPUTATION ---
        if query is not None:
            # === MODE 2: QUERY MATCHING ===
            # query comes in as [bt, d, k, total] (or similar)
            # Expand query to match n_cams: [1, bt, d, k, total]
            q = query.unsqueeze(0) 
            
            scores = reduce(volumes * q, 'nc bt d k total -> nc bt k total', 'sum')
            scores = scores / (d ** 0.5)
            scores = scores.unsqueeze(2) # [nc, bt, 1, k, total]

        else:
            # === MODE 1: SALIENCE DETECTION (Init) ===
            # Use 3D CNN to find "interesting" features
            x = rearrange(volumes, 'nc bt d k (z y x) -> (nc bt k) d z y x', 
                          z=cube_size, y=cube_size, x=cube_size)
            
            scores = self.salience_net(x)
            
            scores = rearrange(scores, '(nc bt k) 1 z y x -> nc bt 1 k (z y x)', 
                               nc=n_cams, bt=bt)

        masks_expanded = masks.unsqueeze(2) # [nc, bt, 1, k, total]
        scores = scores.masked_fill(~masks_expanded, -1e9)

        attn_weights = F.softmax(scores, dim=0)
        
        aggregated_volume = (volumes * attn_weights).sum(dim=0)

        return aggregated_volume    
    
class DepthwiseSeparableResBlock(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.depthwise = nn.Conv3d(latent_dim, latent_dim, 3, padding=1, groups=latent_dim)
        self.pointwise = nn.Conv3d(latent_dim, latent_dim, 1)
        self.skip = nn.Conv3d(latent_dim, latent_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.pointwise(self.relu(self.depthwise(x))) +
                         self.skip(x))


class DepthwiseSeparableV2V(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.block1 = DepthwiseSeparableResBlock(latent_dim)
        self.block2 = DepthwiseSeparableResBlock(latent_dim)
        self.conv_out = nn.Conv3d(latent_dim, latent_dim, 1)

        self._initialize_weights()
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv_out(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    

class PlanesV2V(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(latent_dim, latent_dim*2, 3, padding=1)
        self.conv_2 = nn.Conv2d(latent_dim*2, latent_dim, 3, padding=1)
        self.conv_3d = nn.Conv3d(latent_dim, latent_dim, 3, padding=1)
        self.conv_out = nn.Conv3d(latent_dim, latent_dim, 1)
        self.relu = nn.ReLU()
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # xy planes (along z)
        x_xy = rearrange(x, 'b c d h w -> (b d) c h w')
        x_xy = self.relu(self.conv_1(x_xy))
        x_xy = self.relu(self.conv_2(x_xy))
        x_xy = rearrange(x_xy, '(b d) c h w -> b c d h w', b=b)
        
        # xz planes (along y)
        x_xz = rearrange(x, 'b c d h w -> (b h) c d w')
        x_xz = self.relu(self.conv_1(x_xz))
        x_xz = self.relu(self.conv_2(x_xz))
        x_xz = rearrange(x_xz, '(b h) c d w -> b c d h w', b=b)
        
        # yz planes (along x)
        x_yz = rearrange(x, 'b c d h w -> (b w) c d h')
        x_yz = self.relu(self.conv_1(x_yz))
        x_yz = self.relu(self.conv_2(x_yz))
        x_yz = rearrange(x_yz, '(b w) c d h -> b c d h w', b=b)
        
        # combine and mix with 3d conv
        out = x_xy + x_xz + x_yz
        out = self.relu(self.conv_3d(out))
        return self.conv_out(out)

