import torch

from einops import rearrange

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
