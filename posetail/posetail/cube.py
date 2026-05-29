import itertools

import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange, einsum, repeat


def project_volumes(volumes): 
    ''' 
    project volume to get the xy, xz, and yz planes 
    ''' 
    xy_planes = torch.sum(volumes, dim = -1)
    xz_planes = torch.sum(volumes, dim = -2)
    yz_planes = torch.sum(volumes, dim = -3)

    return xy_planes, xz_planes, yz_planes


def to_homogeneous(p):
    one_size = p.shape[:-1] + (1,)
    ones = torch.ones(size=one_size, dtype=p.dtype, device=p.device)
    return torch.cat([p, ones], dim=-1)


def from_homogeneous(p, eps=1e-10):
    denom = p[..., -1, None]
    denom = torch.where(denom >= 0, 
                        torch.clamp(denom, min=eps), 
                        torch.clamp(denom, max=-eps))
    return p[..., :-1] / denom    
    # return p[..., :-1] / (p[..., -1, None] + eps) 


# @torch.compile
def project_cam(cam, p3d_t, downsample_factor = 1, max_normalized = 3.0):
    if cam['type'] == 'orthographic':
        P = cam['proj_mat']
        p2d = to_homogeneous(p3d_t) @ P[:2].T
        if 'offset' in cam:
            p2d = p2d - cam['offset'][None, :]
        return p2d / downsample_factor

    # p3d_t = torch.as_tensor(p3d)
    # ext_t = torch.as_tensor(cam.get_extrinsics_mat(), dtype=p3d_t.dtype, device=p3d_t.device)
    # mat_t = torch.as_tensor(cam.get_camera_matrix(), dtype=p3d_t.dtype, device=p3d_t.device)
    ext_t = cam['ext']
    mat_t = cam['mat']
    dist = cam['dist']
    cam_type = cam['type'] # pinhole, fisheye # TODO: implement functionality for different camera types

    p2d_proj_raw = torch.matmul(to_homogeneous(p3d_t), ext_t.T)
    p2d_proj_raw = from_homogeneous(p2d_proj_raw[..., :3])

    # handle points way outside of the frame
    p2d_proj_raw = torch.clamp(p2d_proj_raw, -max_normalized, max_normalized)
    
    k1, k2, p1, p2, k3 = dist[:5]
    k4 = k5 = k6 = 0
    r2 = torch.sum(torch.square(p2d_proj_raw), dim=-1)
    r4 = r2 * r2
    r6 = r4 * r2
    kscale = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)

    x = p2d_proj_raw[..., 0]
    y = p2d_proj_raw[..., 1]
    dx = 2*p1*x*y + p2 * (r2 + 2*x*x)
    dy = p1*(r2 + 2*y*y) + 2*p2*x*y
    p1_p2_add = torch.stack([dx, dy], dim=-1)
    
    p2d_dist = kscale[..., None] * p2d_proj_raw + p1_p2_add

    # p2d_dist = p2d_proj_raw

    p2d = torch.matmul(p2d_dist, mat_t[:2,:2].T) + mat_t[:2,2]
    
    # p2d_raw = torch.matmul(to_homogeneous(p2d_dist), mat_t.T)
    # p2d = from_homogeneous(p2d_raw)

    # handle camera offset
    if 'offset' in cam: 
        offset = cam['offset']
        p2d = p2d - offset[None, :]

    # account for downsampling
    p2d = p2d / downsample_factor
    
    return p2d

# @torch.compile
def project_points_torch(camera_group, coords_3d, downsample_factor = 1):

    coords_proj = torch.stack([project_cam(cam, coords_3d, downsample_factor)
                               for cam in camera_group])

    return coords_proj


def triangulate_simple(points, camera_mats, weights):
    '''
    Inputs:
        points: [C, 2] 2d points to triangulate
        camera_mats: [C, 4, 4] camera extrinsics
        weights: [C] weight for each camera
    Outputs:
        p3d: [3] triangulated 3d point
    '''
    num_cams = len(camera_mats)
    A = torch.zeros((num_cams * 2, 4), dtype=points.dtype, device=points.device)
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        w = weights[i]
        A[(i * 2):(i * 2 + 1)] = w * (x * mat[2] - mat[0])
        A[(i * 2 + 1):(i * 2 + 2)] = w * (y * mat[2] - mat[1])
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d

def triangulate_simple_batch(points, camera_mats, weights):
    '''
    Inputs:
        points: [C, N, 2] 2d points to triangulate
        camera_mats: [C, 4, 4] camera extrinsics
        weights: [C, N] weight for each camera
    Outputs:
        p3d: [N, 3] triangulated 3d point
    '''
    C, N, _ = points.shape

    points = rearrange(points, 'c n r -> n c r')
    
    # Expand camera_mats to [N, C, 4, 4]
    cam_mats = repeat(camera_mats, 'c i j -> n c i j', n=N)
    
    # Extract x, y coordinates and reshape weights
    x = points[:, :, 0:1, None]  # [N, C, 1]
    y = points[:, :, 1:2, None]  # [N, C, 1]
    w = rearrange(weights, 'c n -> n c 1 1')  # [N, C, 1]
    
    # Build equations for each camera
    # x * mat[2] - mat[0] and y * mat[2] - mat[1]
    eq_x = w * (x * cam_mats[:, :, 2:3, :] - cam_mats[:, :, 0:1, :])  # [N, C, 1, 4]
    eq_y = w * (y * cam_mats[:, :, 2:3, :] - cam_mats[:, :, 1:2, :])  # [N, C, 1, 4]
    
    # Stack and reshape to [N, C*2, 4]
    A = rearrange([eq_x, eq_y], 'two n c 1 j -> n (c two) j')
    
    # SVD decomposition
    u, s, vh = torch.linalg.svd(A, full_matrices=True)  # vh: [N, 4, 4]
    
    # Take last row of vh for each point
    p3d_homogeneous = vh[:, -1, :]  # [N, 4]
    
    # Convert from homogeneous to 3D coordinates
    p3d = p3d_homogeneous[:, :3] / p3d_homogeneous[:, 3:4]  # [N, 3]
    
    return p3d


def triangulate_simple_batch_reg(points, camera_mats, weights):
    '''
    Inputs:
        points: [C, N, 2] 2d points to triangulate
        camera_mats: [C, 4, 4] camera extrinsics
        weights: [C, N] weight for each camera
    Outputs:
        p3d: [N, 3] triangulated 3d point
    '''
    C, N, _ = points.shape

    points = rearrange(points, 'c n r -> n c r')
    cam_mats = repeat(camera_mats, 'c i j -> n c i j', n=N)
    
    x = points[:, :, 0:1, None]
    y = points[:, :, 1:2, None]
    w = rearrange(weights, 'c n -> n c 1 1')
    
    eq_x = w * (x * cam_mats[:, :, 2:3, :] - cam_mats[:, :, 0:1, :])
    eq_y = w * (y * cam_mats[:, :, 2:3, :] - cam_mats[:, :, 1:2, :])
    
    A = rearrange([eq_x, eq_y], 'two n c 1 j -> n (c two) j')
    
    # Use eigendecomposition of A^T A instead of SVD
    # This is more numerically stable for gradients
    ATA = torch.bmm(A.transpose(-2, -1), A)
    
    # Add small regularization
    eps = 1e-6
    ATA = ATA + eps * torch.eye(4, device=A.device, dtype=A.dtype)
    
    # Smallest eigenvalue's eigenvector
    eigenvalues, eigenvectors = torch.linalg.eigh(ATA)
    p3d_homogeneous = eigenvectors[:, :, 0]  # eigenvector for smallest eigenvalue
    
    p3d = from_homogeneous(p3d_homogeneous)
    
    return p3d



def undistort_points(cam, points):
    if cam['type'] == 'orthographic':
        return points + cam['offset']
    matrix = cam['mat']
    dist = cam['dist']
    offset = cam['offset']

    shape = points.shape
    points = points.reshape(-1, 2)
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    x = (points[:, 0] + offset[0] - cx) / fx
    y = (points[:, 1] + offset[1] - cy) / fy
    x0, y0 = x.clone(), y.clone()
    for _ in range(5):
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
        if dist.shape[0] > 4:
            k3 = dist[4]
        else:
            k3 = torch.tensor(0.0, device=dist.device, dtype=dist.dtype)
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        dx = 2*p1*x*y + p2*(r2 + 2*x*x)
        dy = p1*(r2 + 2*y*y) + 2*p2*x*y
        x = (x0 - dx) / radial
        y = (y0 - dy) / radial
        
    return torch.stack([x, y], dim=1).reshape(shape)


def projection_sensitivity(cam, p):
    p = p.float()
    n_points = p.shape[0]
    if cam['type'] == 'orthographic':
        A = cam['proj_mat'][:2, :3].to(torch.float64)
        return A.unsqueeze(0).expand(n_points, -1, -1)
    fx = cam['mat'][0,0]
    fy = cam['mat'][1,1]
    ext_t = cam['ext'].float()
    
    p_cam = torch.matmul(to_homogeneous(p), ext_t.T)[:, :3]
    depth = p_cam[:,2]
    # print("depth min:", depth.min())
    # print("depth max:", depth.max())
    # print("near zero:", (depth.abs() < 1e-6).sum())
    # print("negative:", (depth < 0).sum())
    X = p_cam[:, 0]
    Y = p_cam[:, 1]
    Z = p_cam[:, 2]
    
    J_proj = torch.zeros((n_points, 2, 3), dtype=torch.float64, device=p_cam.device)
    J_proj[:, 0, 0] = fx / Z
    J_proj[:, 0, 2] = -fx * X / (Z**2)
    J_proj[:, 1, 1] = fy / Z
    J_proj[:, 1, 2] = -fy * Y / (Z**2)
    
    R = ext_t[:3,:3].to(torch.float64)
    J = einsum(J_proj, R, 'n i j, j k -> n i k')
    return J

def is_point_visible(cam, p3d, margin=0):
    """
    Check if 3D points project into camera view.
    margin: pixels from border (e.g., 10 to avoid edge effects)
    """
    p2d = project_cam(cam, p3d)
    w, h = cam['size']

    # check if in bounds
    in_bounds = (
        (p2d[:, 0] >= margin) &
        (p2d[:, 0] < w - margin) &
        (p2d[:, 1] >= margin) &
        (p2d[:, 1] < h - margin)
    )

    if cam['type'] == 'orthographic':
        return in_bounds

    # check if point is in front of camera
    p_cam = torch.matmul(to_homogeneous(p3d), cam['ext'].T)[:, :3]
    in_front = p_cam[:, 2] > 0

    return in_bounds & in_front

def fill_nan_with_batch_median(scale):
    """Fill NaN cells of a (cams, B) scale tensor with the per-batch median
    over finite cells along the cams axis. If all cameras for a batch element
    are NaN, the cell stays NaN and propagates downstream."""
    finite = torch.isfinite(scale)
    per_batch = torch.nanmedian(scale, dim=0).values  # (B,) — NaN where all-NaN
    return torch.where(finite, scale, per_batch[None, :].expand_as(scale))


def get_camera_scale(camera_group, p):
    """
    Args:
        camera_group: list of camera dicts
        p: (B, N, 3) 3D points
    Returns:
        scale: (n_cams, B) tensor; NaN cells filled with per-batch median
    """
    B, N, _ = p.shape
    n_cams = len(camera_group)
    sensitivity = p.new_full((n_cams, B), float('nan'))

    for ci, cam in enumerate(camera_group):
        for b in range(B):
            visible = is_point_visible(cam, p[b])
            if torch.sum(visible) > 0:
                with torch.autocast(device_type=p.device.type, enabled=False):
                    J = projection_sensitivity(cam, p[b][visible])
                    s = torch.linalg.svdvals(J.float())[:, 0]
                sensitivity[ci, b] = torch.median(s)

    scale = 1.0 / sensitivity
    return fill_nan_with_batch_median(scale)

class UnprojectViews:

    def __init__(self, 
                 camera_group, 
                 cube_center,
                 cube_extent = None,
                 cube_dim = 64, 
                 downsample_factor = 2, 
                 device = None):

        self.cgroup = camera_group
        self.cube_center = cube_center.cpu()
        self.cube_extent = cube_extent
        self.cube_dim = cube_dim 
        self.downsample_factor = downsample_factor
        self.device = device

        if cube_extent is not None: 
            self.cube_extent = cube_extent

        self.coords_proj = self.create_mesh_3d()


    def init_coords(self, dim = 0): 

        coords = np.linspace(
            self.cube_center[dim] - self.cube_extent, 
            self.cube_center[dim] + self.cube_extent, 
            num = self.cube_dim
        )

        return coords
    

    def create_mesh_3d(self):

        xs = self.init_coords(dim = 0)
        ys = self.init_coords(dim = 1)
        zs = self.init_coords(dim = 2)

        # create a mesh of all coords in the volume
        coords = np.array(np.meshgrid(zs, xs, ys))
        coords_flat = rearrange(coords, 'r cd ch cw -> (cd ch cw) r')
        coords_flat = torch.from_numpy(coords_flat).to(self.device).float()

        # project coordinates for each camera 
        coords_proj = project_points_torch(
            camera_group = self.cgroup, 
            coords_3d = coords_flat,
            downsample_factor = self.downsample_factor, 
        )

        return coords_proj


    def unproject_to_volume(self, feature_maps): 

        sampled_points = []

        for i, (features, coord_grid) in enumerate(zip(feature_maps, self.coords_proj)):

            B, S, D, H, W = features.shape
            features = rearrange(features, 'b s d h w -> (b s) d h w')

            coord_grid = rearrange(coord_grid, "dhw r -> 1 1 dhw r") 
            coord_grid = (torch.as_tensor(coord_grid)
                               .repeat((B * S, 1, 1, 1))
                               .to(features.device))

            scale = torch.tensor([H, W], device = features.device) 
            coord_grid = 2 * coord_grid / scale - 1

            # sample projected volumetric coordinates from the feature maps 
            sampled = F.grid_sample(
                features.float(), 
                coord_grid.float(),
                mode = 'bilinear', 
                padding_mode = 'zeros',
                align_corners = True
            )

            # switch from d h w -> h w d
            sampled = rearrange(
                sampled, 
                '(b s) d 1 (cd ch cw) -> 1 b s d ch cw cd', 
                b = B, s = S,
                ch = self.cube_dim, 
                cw = self.cube_dim, 
                cd = self.cube_dim
            )

            sampled_points.append(sampled)

        # (n_cams, B, S, D, cube_dim, cube_dim, cube_dim)
        volumes = torch.vstack(sampled_points) 

        return volumes

def apply_proj(feats, matrix):
    D = matrix.shape[-1]
    x = rearrange(feats, 'b heads cams (k d) -> b heads cams k d', d=D)
    out = einsum(matrix, x, 'b cams i j, b heads cams k j -> b heads cams k i')
    return rearrange(out, 'b heads cams k d -> b heads cams (k d)')

def prope_projmat_only_attention(
    q: torch.Tensor,  # (batch, heads, cams, head_dim)
    k: torch.Tensor,  # (batch, heads, cams, head_dim)
    v: torch.Tensor,  # (batch, heads, cams, head_dim)
    viewmats: torch.Tensor,  # (batch, cams, 4, 4)
    **kwargs,
) -> torch.Tensor:
    batch, heads, cams, head_dim = q.shape
    D = 4
    assert head_dim % D == 0

    P_T = viewmats.transpose(-1, -2)
    P_inv = _invert_SE3(viewmats)
    P = viewmats

    q_rot = apply_proj(q, P_T)
    k_rot = apply_proj(k, P_inv)
    v_rot = apply_proj(v, P_inv)

    out = F.scaled_dot_product_attention(q_rot, k_rot, v_rot, **kwargs)

    out = apply_proj(out, P)
    return out


class CameraSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 4 == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(
        self,
        vectors: torch.Tensor,   # (batch, cams, embed_dim)
        viewmats: torch.Tensor,  # (batch, cams, 4, 4)
    ) -> torch.Tensor:
        batch, cams, embed_dim = vectors.shape
        assert embed_dim == self.embed_dim

        q = self.q_proj(vectors)
        k = self.k_proj(vectors)
        v = self.v_proj(vectors)

        q = rearrange(q, 'b cams (heads d) -> b heads cams d', heads=self.num_heads)
        k = rearrange(k, 'b cams (heads d) -> b heads cams d', heads=self.num_heads)
        v = rearrange(v, 'b cams (heads d) -> b heads cams d', heads=self.num_heads)

        out = prope_projmat_only_attention(q, k, v, viewmats)

        out = rearrange(out, 'b heads cams d -> b cams (heads d)')
        out = self.out_proj(out)
        return out

def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out



def _points_to_rays_orthographic(cam, p2d, cube_scale=1, normalize_t=True):
    """Orthographic counterpart to points_to_rays.

    Ray direction is constant (proj_dir); origin is the lifted pixel in world.
    """
    B = p2d.shape[0]
    device = p2d.device
    dtype = p2d.dtype

    P = cam['proj_mat']
    A_pinv = cam['A_pinv']
    proj_dir = cam['proj_dir']

    p_full = undistort_points(cam, p2d)                     # [B, 2] full-image pixels
    t_proj = P[:2, 3]                                       # [2]
    origin = einsum(A_pinv, p_full - t_proj, 'i j, b j -> b i')  # [B, 3]

    if normalize_t:
        origin = origin / (cube_scale * 200.0)

    z_ray = repeat(proj_dir.to(dtype), 'c -> b c', b=B)
    cam_y = repeat(F.normalize(P[1, :3].to(dtype), dim=-1, eps=1e-8),
                   'c -> b c', b=B)
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-8)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-8)

    R_w2r = torch.stack([x_ray, y_ray, z_ray], dim=1)        # [B, 3, 3]
    t_w2r = -einsum(R_w2r, origin, 'b i j, b j -> b i')      # [B, 3]

    ray_matrices = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    ray_matrices[:, :3, :3] = R_w2r
    ray_matrices[:, :3, 3] = t_w2r
    ray_matrices[:, 3, 3] = 1.0
    return ray_matrices


def points_to_rays(cam, p2d, cube_scale=1, normalize_t=True):
    """Inputs:
    cam: camera dict
    p2d: [B, 2]

    Outputs:
    ray_matrices: [B, 4, 4]
    """
    if cam['type'] == 'orthographic':
        return _points_to_rays_orthographic(
            cam, p2d, cube_scale=cube_scale, normalize_t=normalize_t)
    B = p2d.shape[0]
    device = p2d.device
    dtype = p2d.dtype

    # Undistort and lift to normalized camera coords
    p2d_und = undistort_points(cam, p2d)          # [B, 2]
    d_cam = to_homogeneous(p2d_und)               # [B, 3]
    d_cam = F.normalize(d_cam, dim=-1, eps=1e-8)

    # Camera-to-world rotation and translation from ext
    ext = cam['ext'].clone()                              # [4, 4] world-to-camera
    R_c2w = ext[:3, :3].T # [3, 3], transpose = invert rotation
    if normalize_t:
        t_c2w = -einsum(R_c2w, ext[:3, 3] / cube_scale / 200.0, 'i j, j -> i')
    else:
        t_c2w = -einsum(R_c2w, ext[:3, 3], 'i j, j -> i')
        

    # Ray directions in world space
    d_world = einsum(R_c2w, d_cam, 'i j, b j -> b i')
    d_world = F.normalize(d_world, dim=-1, eps=1e-8)  # [B, 3]

    # Camera origin broadcast to all rays
    origin = repeat(t_c2w, 'c -> b c', b=B)      # [B, 3]

    # Camera y-axis (column 1 of R_c2w) broadcast to all rays
    cam_y = repeat(R_c2w[:, 1], 'c -> b c', b=B) # [B, 3]

    # Orthonormal ray-local frame: z=ray, x=cam_y×z, y=z×x
    z_ray = d_world
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-8)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-8)

    # world-to-ray rotation: stack axes as rows
    R_w2r = torch.stack([x_ray, y_ray, z_ray], dim=1)  # [B, 3, 3]

    # world-to-ray translation
    t_w2r = -einsum(R_w2r, origin, 'b i j, b j -> b i')  # [B, 3]

    # Assemble 4x4 matrices
    ray_matrices = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    ray_matrices[:, :3, :3] = R_w2r
    ray_matrices[:, :3, 3] = t_w2r
    ray_matrices[:, 3, 3] = 1.0

    return ray_matrices
