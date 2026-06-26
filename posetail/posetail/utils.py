import torch

from einops import rearrange


def apply_rope_1d(x, positions, base=10000):
    """
    Apply 1-D rotary position embeddings to x.
    x: (..., T, head_dim)  — head_dim must be even
    positions: (T,) integer positions
    Rotates consecutive pairs of head_dim dims by position-dependent angles.
    Applied to Q and K (not V) before scaled_dot_product_attention.
    """
    head_dim = x.shape[-1]
    assert head_dim % 2 == 0
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(half, device=x.device, dtype=x.dtype) * 2 / head_dim))
    angles = positions.to(x.dtype).unsqueeze(-1) * freqs.unsqueeze(0)  # (T, half)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

def _sincos_1d(dim, pos, base=10000.0):
    """Standard 1-D sin/cos positional encoding. dim must be even.
    pos: (M,) positions -> (M, dim) with [sin | cos] halves."""
    assert dim % 2 == 0, f'sincos dim must be even, got {dim}'
    half = dim // 2
    omega = torch.arange(half, device=pos.device, dtype=torch.float32) / half
    omega = 1.0 / (base ** omega)                       # (half,)
    out = pos.float()[:, None] * omega[None, :]         # (M, half)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (M, dim)


def get_3d_sincos_pos_embed(embed_dim, gT, gH, gW, device, dtype, base=10000.0):
    """Fixed (non-learned) factorized 3-D sin/cos positional embedding over a (T,H,W) token
    grid. Time gets embed_dim/2, height and width embed_dim/4 each (mirrors VJEPA's
    non-uniform split). Returns (1, gT*gH*gW, embed_dim) in t-major then h then w row-major
    order -- matching the (1, T', H', W', D) reshape SceneRepresentation uses for its learned
    table. Deterministic and length-agnostic: defined at any grid size with no interpolation,
    so it is the drop-in 'sincos' alternative to the learned pos_embed."""
    assert embed_dim % 4 == 0, f'sincos embed_dim must be divisible by 4, got {embed_dim}'
    dt, dh = embed_dim // 2, embed_dim // 4
    dw = embed_dim - dt - dh                            # == embed_dim // 4
    t = torch.arange(gT, device=device)
    h = torch.arange(gH, device=device)
    w = torch.arange(gW, device=device)
    grid_t = t.view(gT, 1, 1).expand(gT, gH, gW).reshape(-1)
    grid_h = h.view(1, gH, 1).expand(gT, gH, gW).reshape(-1)
    grid_w = w.view(1, 1, gW).expand(gT, gH, gW).reshape(-1)
    emb = torch.cat([_sincos_1d(dt, grid_t, base),
                     _sincos_1d(dh, grid_h, base),
                     _sincos_1d(dw, grid_w, base)], dim=1)  # (N, embed_dim)
    return emb[None].to(dtype)                          # (1, N, embed_dim)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def init_kwargs(kwargs_dict):
     
    d = {}
    if kwargs_dict is not None:
        d = kwargs_dict

    return d

# @torch.compile
def get_pos_encoding(pos, dim, dtype = torch.double): 
    ''' 
    pos: positions to encode
    dim: the embedding dimension (should match the transformer input dimension)
    '''
    assert (dim % 2 == 0)
    
    d = dim // 2
    dim_ixs = torch.arange(d, device = pos.device, dtype = dtype)

    v = 1 / (10000 ** (2 * dim_ixs / dim))
    m = torch.einsum('i,j->ij', pos, v)

    sin_encoding = torch.sin(m)
    cos_encoding = torch.cos(m)

    encoding = torch.cat((sin_encoding, cos_encoding), dim = 1)
    encoding = rearrange(encoding, 'p d -> 1 p 1 d').to(dtype)

    return encoding


# @torch.compile
def get_fourier_encoding(coords, min_freq = 0, max_freq = 1):

    freqs = 2 ** torch.arange(min_freq, max_freq, device = coords.device, dtype = coords.dtype)
    freq_coords = torch.einsum('bsnr,f->bsnfr', coords, freqs)
    freq_coords = rearrange(freq_coords, 'b s n f r -> b s n (f r)')

    fourier_enc = torch.cat([torch.sin(freq_coords), torch.cos(freq_coords)], dim = -1)

    return fourier_enc


def encode_dim(enc, coords, v): 

    coords = torch.unsqueeze(coords, dim = -1)
    enc[:, :, 0::2] = torch.sin(coords * v)
    enc[:, :, 1::2] = torch.cos(coords * v)

    return enc


class PadToMultiple:
    def __init__(self, multiple=32):
        self.multiple = multiple
    
    def __call__(self, img):
        # Works for any shape - assumes last 2 dims are H, W
        original_shape = img.shape
        *batch_dims, c, h, w = original_shape
        pad_h = (self.multiple - h % self.multiple) % self.multiple
        pad_w = (self.multiple - w % self.multiple) % self.multiple
        
        if pad_h == 0 and pad_w == 0:
            return img
        
        # Reshape to 4D for padding (B, C, H, W)
        img_4d = img.reshape(-1, c, h, w)  # Flatten all batch dims, keep C, H, W
        
        # Pad
        padded = torch.nn.functional.pad(img_4d, (0, pad_w, 0, pad_h), mode='constant', value=0)
        # padded = torch.nn.functional.pad(img_4d, (0, pad_w, 0, pad_h), mode='replicate')
        
        # Reshape back to original batch structure
        new_shape = batch_dims + [c, h + pad_h, w + pad_w]
        return padded.reshape(new_shape)


class PadToSize:
    def __init__(self, size=256):
        self.size = size
    
    def __call__(self, img):
        # Works for any shape - assumes last 2 dims are H, W
        original_shape = img.shape
        *batch_dims, c, h, w = original_shape
        pad_h = max(self.size - h, 0)
        pad_w = max(self.size - w, 0)
        
        if pad_h == 0 and pad_w == 0:
            return img
        
        # Reshape to 4D for padding (B, C, H, W)
        img_4d = img.reshape(-1, c, h, w)  # Flatten all batch dims, keep C, H, W
        
        # Pad
        padded = torch.nn.functional.pad(img_4d, (0, pad_w, 0, pad_h), mode='constant', value=0)
        # padded = torch.nn.functional.pad(img_4d, (0, pad_w, 0, pad_h), mode='replicate')
        
        # Reshape back to original batch structure
        new_shape = batch_dims + [c, h + pad_h, w + pad_w]
        return padded.reshape(new_shape)
