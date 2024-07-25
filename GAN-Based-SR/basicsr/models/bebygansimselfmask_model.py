import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import math
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import typing
import functools

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.losses.loss_util import similarity_map

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBBebyGANNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBBebyGANNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

def get_flat_mask(img, kernel_size=11, std_thresh=0.025, scale=1):
    if scale > 1:
        img = F.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
    B, _, H, W = img.size()
    r, g, b = torch.unbind(img, dim=1)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(dim=1)
    l_img_pad = F.pad(l_img, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
    unf_img = F.unfold(l_img_pad, kernel_size=kernel_size, padding=0, stride=1)
    std_map = torch.std(unf_img, dim=1, keepdim=True).view(B, 1, H, W)
    mask = torch.lt(std_map, std_thresh).float()

    return mask

K = typing.TypeVar('K', str, torch.Tensor)

def cubic_contribution(x: torch.Tensor, a: float=-0.5) -> torch.Tensor:
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = (ax <= 1)
    range_12 = (ax > 1) * (ax <= 2)

    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01.to(dtype=x.dtype)

    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12.to(dtype=x.dtype)

    cont = cont_01 + cont_12
    cont = cont / cont.sum()
    return cont

def gaussian_contribution(x: torch.Tensor, sigma: float=2.0) -> torch.Tensor:
    range_3sigma = (x.abs() <= 3 * sigma + 1)
    # Normalization will be done after
    cont = torch.exp(-x.pow(2) / (2 * sigma**2))
    cont = cont * range_3sigma.to(dtype=x.dtype)
    return cont

def discrete_kernel(
        kernel: str, scale: float, antialiasing: bool=True) -> torch.Tensor:

    '''
    For downsampling with integer scale only.
    '''
    downsampling_factor = int(1 / scale)
    if kernel == 'cubic':
        kernel_size_orig = 4
    else:
        raise ValueError('Pass!')

    if antialiasing:
        kernel_size = kernel_size_orig * downsampling_factor
    else:
        kernel_size = kernel_size_orig

    if downsampling_factor % 2 == 0:
        a = kernel_size_orig * (0.5 - 1 / (2 * kernel_size))
    else:
        kernel_size -= 1
        a = kernel_size_orig * (0.5 - 1 / (kernel_size + 1))

    with torch.no_grad():
        r = torch.linspace(-a, a, steps=kernel_size)
        k = cubic_contribution(r).view(-1, 1)
        k = torch.matmul(k, k.t())
        k /= k.sum()

    return k

def reflect_padding(
        x: torch.Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int) -> torch.Tensor:

    '''
    Apply reflect padding to the given Tensor.
    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.
    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    '''
    b, c, h, w = x.size()
    if dim == 2 or dim == -2:
        padding_buffer = x.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre:(h + pad_pre), :].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(x[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(x[..., -(p + 1), :])
    else:
        padding_buffer = x.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre:(w + pad_pre)].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(x[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(x[..., -(p + 1)])

    return padding_buffer

def padding(
        x: torch.Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int,
        padding_type: str='reflect') -> torch.Tensor:

    if padding_type == 'reflect':
        x_pad = reflect_padding(x, dim, pad_pre, pad_post)
    else:
        raise ValueError('{} padding is not supported!'.format(padding_type))

    return x_pad

def get_padding(
        base: torch.Tensor,
        kernel_size: int,
        x_size: int) -> typing.Tuple[int, int, torch.Tensor]:

    base = base.long()
    r_min = base.min()
    r_max = base.max() + kernel_size - 1

    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        base += pad_pre
    else:
        pad_pre = 0

    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0

    return pad_pre, pad_post, base

def get_weight(
        dist: torch.Tensor,
        kernel_size: int,
        kernel: str='cubic',
        sigma: float=2.0,
        antialiasing_factor: float=1) -> torch.Tensor:

    buffer_pos = dist.new_zeros(kernel_size, len(dist))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(dist - idx)

    # Expand (downsampling) / Shrink (upsampling) the receptive field.
    buffer_pos *= antialiasing_factor
    if kernel == 'cubic':
        weight = cubic_contribution(buffer_pos)
    elif kernel == 'gaussian':
        weight = gaussian_contribution(buffer_pos, sigma=sigma)
    else:
        raise ValueError('{} kernel is not supported!'.format(kernel))

    weight /= weight.sum(dim=0, keepdim=True)
    return weight

def reshape_tensor(x: torch.Tensor, dim: int, kernel_size: int) -> torch.Tensor:
    # Resize height
    if dim == 2 or dim == -2:
        k = (kernel_size, 1)
        h_out = x.size(-2) - kernel_size + 1
        w_out = x.size(-1)
    # Resize width
    else:
        k = (1, kernel_size)
        h_out = x.size(-2)
        w_out = x.size(-1) - kernel_size + 1

    unfold = F.unfold(x, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold

def resize_1d(
        x: torch.Tensor,
        dim: int,
        side: int=None,
        kernel: str='cubic',
        sigma: float=2.0,
        padding_type: str='reflect',
        antialiasing: bool=True) -> torch.Tensor:

    '''
    Args:
        x (torch.Tensor): A torch.Tensor of dimension (B x C, 1, H, W).
        dim (int):
        scale (float):
        side (int):
    Return:
    '''
    scale = side / x.size(dim)
    # Identity case
    if scale == 1:
        return x

    # Default bicubic kernel with antialiasing (only when downsampling)
    if kernel == 'cubic':
        kernel_size = 4
    else:
        kernel_size = math.floor(6 * sigma)

    if antialiasing and (scale < 1):
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1

    # We allow margin to both sides
    kernel_size += 2

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    with torch.no_grad():
        d = 1 / (2 * side)
        pos = torch.linspace(
            start=d,
            end=(1 - d),
            steps=side,
            dtype=x.dtype,
            device=x.device,
        )
        pos = x.size(dim) * pos - 0.5
        base = pos.floor() - (kernel_size // 2) + 1
        dist = pos - base
        weight = get_weight(
            dist,
            kernel_size,
            kernel=kernel,
            sigma=sigma,
            antialiasing_factor=antialiasing_factor,
        )
        pad_pre, pad_post, base = get_padding(base, kernel_size, x.size(dim))

    # To backpropagate through x
    x_pad = padding(x, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = reshape_tensor(x_pad, dim, kernel_size)
    # Subsampling first
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))

    # Apply the kernel
    down = sample * weight
    down = down.sum(dim=1, keepdim=True)
    return down

def downsampling_2d(
        x: torch.Tensor,
        k: torch.Tensor,
        scale: int,
        padding_type: str='reflect') -> torch.Tensor:

    c = x.size(1)
    k_h = k.size(-2)
    k_w = k.size(-1)

    k = k.to(dtype=x.dtype, device=x.device)
    k = k.view(1, 1, k_h, k_w)
    k = k.repeat(c, c, 1, 1)
    e = torch.eye(c, dtype=k.dtype, device=k.device, requires_grad=False)
    e = e.view(c, c, 1, 1)
    k = k * e

    pad_h = (k_h - scale) // 2
    pad_w = (k_w - scale) // 2
    x = padding(x, -2, pad_h, pad_h, padding_type=padding_type)
    x = padding(x, -1, pad_w, pad_w, padding_type=padding_type)
    y = F.conv2d(x, k, padding=0, stride=scale)
    return y

def imresize(
        x: torch.Tensor,
        scale: float=None,
        sides: typing.Tuple[int, int]=None,
        kernel: K='cubic',
        sigma: float=2,
        rotation_degree: float=0,
        padding_type: str='reflect',
        antialiasing: bool=True) -> torch.Tensor:

    '''
    Args:
        x (torch.Tensor):
        scale (float):
        sides (tuple(int, int)):
        kernel (str, default='cubic'):
        sigma (float, default=2):
        rotation_degree (float, default=0):
        padding_type (str, default='reflect'):
        antialiasing (bool, default=True):
    Return:
        torch.Tensor:
    '''

    if scale is None and sides is None:
        raise ValueError('One of scale or sides must be specified!')
    if scale is not None and sides is not None:
        raise ValueError('Please specify scale or sides to avoid conflict!')

    if x.dim() == 4:
        b, c, h, w = x.size()
    elif x.dim() == 3:
        c, h, w = x.size()
        b = None
    elif x.dim() == 2:
        h, w = x.size()
        b = c = None
    else:
        raise ValueError('{}-dim Tensor is not supported!'.format(x.dim()))

    x = x.view(-1, 1, h, w)

    if sides is None:
        # Determine output size
        sides = (math.ceil(h * scale), math.ceil(w * scale))
        scale_inv = 1 / scale
        if isinstance(kernel, str) and scale_inv.is_integer():
            kernel = discrete_kernel(kernel, scale, antialiasing=antialiasing)
        elif isinstance(kernel, torch.Tensor) and not scale_inv.is_integer():
            raise ValueError(
                'An integer downsampling factor '
                'should be used with a predefined kernel!'
            )

    if x.dtype != torch.float32 or x.dtype != torch.float64:
        dtype = x.dtype
        x = x.float()
    else:
        dtype = None

    if isinstance(kernel, str):
        # Shared keyword arguments across dimensions
        kwargs = {
            'kernel': kernel,
            'sigma': sigma,
            'padding_type': padding_type,
            'antialiasing': antialiasing,
        }
        # Core resizing module
        x = resize_1d(x, -2, side=sides[0], **kwargs)
        x = resize_1d(x, -1, side=sides[1], **kwargs)
    elif isinstance(kernel, torch.Tensor):
        x = downsampling_2d(x, kernel, scale=int(1 / scale))

    rh = x.size(-2)
    rw = x.size(-1)
    # Back to the original dimension
    if b is not None:
        x = x.view(b, c, rh, rw)        # 4-dim
    else:
        if c is not None:
            x = x.view(c, rh, rw)       # 3-dim
        else:
            x = x.view(rh, rw)          # 2-dim

    if dtype is not None:
        if not dtype.is_floating_point:
            x = x.round()
        # To prevent over/underflow when converting types
        if dtype is torch.uint8:
            x = x.clamp(0, 255)

        x = x.to(dtype=dtype)

    return x

class BBL():
    def __init__(self, alpha=1.0, beta=1.0, ksize=3, pad=0, stride=3, dist_norm='l2'):
        super(BBL, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ksize = ksize
        self.pad = pad
        self.stride = stride
        self.dist_norm = dist_norm

    def pairwise_distance(self, x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a BxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        if y is None:
            dist = dist - torch.diag(dist.diag())

        return torch.clamp(dist, 0.0, np.inf)

    def batch_pairwise_distance(self, x, y=None):
        '''
        Input: x is a BxNxd matrix
               y is an optional BxMxd matirx
        Output: dist is a BxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
                if y is not given then use 'y=x'.
        i.e. dist[b,i,j] = ||x[b,i,:]-y[b,j,:]||^2
        '''
        B, N, d = x.size()
        if self.dist_norm == 'l1':
            x_norm = x.view(B, N, 1, d)
            if y is not None:
                y_norm = y.view(B, 1, -1, d)
            else:
                y_norm = x.view(B, 1, -1, d)
            dist = torch.abs(x_norm - y_norm).sum(dim=3)
        elif self.dist_norm == 'l2':
            x_norm = (x ** 2).sum(dim=2).view(B, N, 1)
            if y is not None:
                M = y.size(1)
                y_t = torch.transpose(y, 1, 2)
                y_norm = (y ** 2).sum(dim=2).view(B, 1, M)
            else:
                y_t = torch.transpose(x, 1, 2)
                y_norm = x_norm.view(B, 1, N)

            dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
            # Ensure diagonal is zero if x=y
            if y is None:
                dist = dist - torch.diag_embed(torch.diagonal(dist, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
            dist = torch.clamp(dist, 0.0, np.inf)
            # dist = torch.sqrt(torch.clamp(dist, 0.0, np.inf) / d)
        else:
            raise NotImplementedError('%s norm has not been supported.' % self.dist_norm)

        return dist

    def forward(self, x, gt):
        p1 = F.unfold(x, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        B, C, H = p1.size()
        p1 = p1.permute(0, 2, 1).contiguous() # [B, H, C]

        p2 = F.unfold(gt, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2 = p2.permute(0, 2, 1).contiguous() # [B, H, C]

        gt_2 = F.interpolate(gt, scale_factor=0.5, mode='bicubic', align_corners = False)
        p2_2 = F.unfold(gt_2, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2_2 = p2_2.permute(0, 2, 1).contiguous() # [B, H, C]

        gt_4 = F.interpolate(gt, scale_factor=0.25, mode='bicubic',align_corners = False)
        p2_4 = F.unfold(gt_4, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2_4 = p2_4.permute(0, 2, 1).contiguous() # [B, H, C]
        p2_cat = torch.cat([p2, p2_2, p2_4], 1)

        score1 = self.alpha * self.batch_pairwise_distance(p1, p2_cat)
        score = score1 + self.beta * self.batch_pairwise_distance(p2, p2_cat) # [B, H, H]

        weight, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, C]) # [B, H, C]
        sel_p2 = torch.gather(p2_cat, dim=1, index=index) # [B, H, C]

        return p1, sel_p2

@MODEL_REGISTRY.register()
class BebyGANSimSelfMaskModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(BebyGANSimSelfMaskModel, self).__init__(opt)

        self.pre_pad = self.opt['pre_pad']
        self.tile_size = self.opt['tile_size']
        self.tile_pad = self.opt['tile_pad']

        self.use_network_d = False

        # define network
        if self.opt['load_mode_g'] == 'my_pretrain':
            self.net_g = build_network(opt['network_g'])
            self.net_g = self.model_to_device(self.net_g)
            self.print_network(self.net_g)

        self.logger = get_root_logger()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            if self.opt['load_mode_g'] == 'original':
                net_g = torch.load(load_path)
                self.net_g = RRDBBebyGANNet().to(self.device)
                self.net_g.load_state_dict(net_g, strict=True)
            elif self.opt['load_mode_g'] == 'my_pretrain':
                param_key = self.opt['path'].get('param_key_g', 'params')
                self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.opt.get('network_d', None) is not None:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)

            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_d', 'params')
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

            self.use_network_d = True

        if self.is_train:
            if self.opt['train'].get('mask_stride', 0) > 1:
                mask_size = int(self.opt['datasets']['train']['gt_size'])
                mask_stride = torch.eye(self.opt['train'].get('mask_stride', 0), self.opt['train'].get('mask_stride', 0), dtype=torch.float32)
                mask_stride = mask_stride.repeat(math.ceil(mask_size / self.opt['train'].get('mask_stride', 0)), math.ceil(mask_size / self.opt['train'].get('mask_stride', 0)))
                mask_stride = mask_stride[:mask_size, :mask_size]
                mask_stride = mask_stride.unsqueeze(0).unsqueeze(0)
                self.mask_stride = nn.Parameter(data=mask_stride, requires_grad=False).cuda()
                print(f"mask stride is {self.opt['train'].get('mask_stride', 0)}")

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()

        self.best_buddy = BBL()

        train_opt = self.opt['train']

        if self.use_network_d:
            self.net_d.train()
            self.net_d_iters = train_opt.get('net_d_iters', 1)
            self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                if self.opt['load_mode_g'] == 'original':
                    net_g_ema = torch.load(load_path)
                    self.net_g_ema = RRDBBebyGANNet().to(self.device)
                    self.net_g_ema.load_state_dict(net_g_ema, strict=True)
                elif self.opt['load_mode_g'] == 'my_pretrain':
                    self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_bb_opt'):
            self.cri_pix_bb = build_loss(train_opt['pixel_bb_opt']).to(self.device)
        else:
            self.cri_pix_bb = None

        if train_opt.get('pixel_bp_opt'):
            self.cri_pix_bp = build_loss(train_opt['pixel_bp_opt']).to(self.device)
        else:
            self.cri_pix_bp = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if train_opt.get('selfsim_opt'):
            self.cri_selfsim = build_loss(train_opt['selfsim_opt']).to(self.device)
        else:
            self.cri_selfsim = None

        if train_opt.get('selfsim1_opt'):
            self.cri_selfsim1 = build_loss(train_opt['selfsim1_opt']).to(self.device)
        else:
            self.cri_selfsim1 = None


        if self.cri_pix_bb is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        if self.use_network_d:
            # optimizer d
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'gt_mask' in data:
            self.gt_mask = data['gt_mask'].to(self.device)
            # print(f"self gt mask shape is {self.gt_mask.shape}")

    def train_net_g(self):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        # self.output_ema = self.net_g_ema(self.lq)
        # pixel loss
        if self.cri_pix_bb:
            p1, sel_p2 = self.best_buddy.forward(x=self.output, gt=self.gt)
            l_pix_bb = self.cri_pix_bb(p1, sel_p2)
            self.l_g_total += l_pix_bb
            self.loss_dict['l_pix_bb'] = l_pix_bb
        if self.cri_pix_bp:
            bp_lr_img = imresize(self.output, scale=1 / self.opt['scale'])
            l_pix_bp = self.cri_pix_bp(bp_lr_img, self.lq)
            self.l_g_total += l_pix_bp
            self.loss_dict['l_pix_bp'] = l_pix_bp
        # self similarity loss
        if self.cri_selfsim or self.cri_selfsim1:
            b, _, _, _ = self.gt.shape
            b_gt_list = []
            b_sr_list = []
            for i in range(b):
                b_mask_gt = self.gt_mask[i, :].unsqueeze(0)  # 1,1,256, 256
                if self.opt['train'].get('mask_stride', 0) > 1:
                    b_mask_gt = self.mask_stride * b_mask_gt
                if b_mask_gt.sum() == 0:
                    pass
                else:
                    b_gt = self.gt[i, :].unsqueeze(0)  # 1,3,256, 256
                    b_sr = self.output[i, :].unsqueeze(0)  # 1,3,256,256
                    output_self_matrix = similarity_map(img=b_sr.clone(), mask=b_mask_gt.clone(),
                                                        simself_strategy=self.opt['train']['simself_strategy'],
                                                        dh=self.opt['train'].get('simself_dh', 16),
                                                        dw=self.opt['train'].get('simself_dw', 16),
                                                        kernel_size=self.opt['train']['kernel_size'],
                                                        scaling_factor=self.opt['train']['scaling_factor'],
                                                        softmax=self.opt['train'].get('softmax_sr', False),
                                                        temperature=self.opt['train'].get('temperature', 0),
                                                        crossentropy=self.opt['train'].get('crossentropy', False),
                                                        rearrange_back=self.opt['train'].get('rearrange_back', True),
                                                        stride=1, pix_num=1, index=None,
                                                        kernel_size_center=self.opt['train'].get('kernel_size_center',9),
                                                        mean=self.opt['train'].get('mean', False),
                                                        var=self.opt['train'].get('var', False),
                                                        gene_type=self.opt['train'].get('gene_type', "sum"),
                                                        largest_k=self.opt['train'].get('largest_k', 0))
                    output_self_matrix = output_self_matrix.getitem()

                    gt_self_matrix = similarity_map(img=b_gt.clone(), mask=b_mask_gt.clone(),
                                                    simself_strategy=self.opt['train']['simself_strategy'],
                                                    dh=self.opt['train'].get('simself_dh', 16),
                                                    dw=self.opt['train'].get('simself_dw', 16),
                                                    kernel_size=self.opt['train']['kernel_size'],
                                                    scaling_factor=self.opt['train']['scaling_factor'],
                                                    softmax=self.opt['train'].get('softmax_gt', False),
                                                    temperature=self.opt['train'].get('temperature', 0),
                                                    crossentropy=self.opt['train'].get('crossentropy', False),
                                                    rearrange_back=self.opt['train'].get('rearrange_back', True),
                                                    stride=1, pix_num=1, index=None,
                                                    kernel_size_center=self.opt['train'].get('kernel_size_center', 9),
                                                    mean=self.opt['train'].get('mean', False),
                                                    var=self.opt['train'].get('var', False),
                                                    gene_type=self.opt['train'].get('gene_type', "sum"),
                                                    largest_k=self.opt['train'].get('largest_k', 0))
                    gt_self_matrix = gt_self_matrix.getitem()

                    b_sr_list.append(output_self_matrix)
                    b_gt_list.append(gt_self_matrix)
                    del output_self_matrix
                    del gt_self_matrix
            b_sr_list = torch.cat(b_sr_list, dim = 1)
            b_gt_list = torch.cat(b_gt_list, dim = 1)

        if self.cri_selfsim:
            l_selfsim = self.cri_selfsim(b_sr_list, b_gt_list)
            self.l_g_total += l_selfsim
            self.loss_dict['l_selfsim'] = l_selfsim

        if self.cri_selfsim1:
            l_selfsim_kl = self.cri_selfsim1(b_sr_list, b_gt_list)
            self.l_g_total += l_selfsim_kl
            self.loss_dict['l_selfsim_kl'] = l_selfsim_kl

        if self.cri_selfsim or self.cri_selfsim1:
            del b_sr_list
            del b_gt_list


        if self.cri_perceptual is None and self.cri_gan is None:
            self.l_g_total.backward()
            self.optimizer_g.step()

    def optimize_parameters(self, current_iter):
        self.l_g_total = 0  # total loss, include L1Loss, perceptual loss, GAN loss and so on
        self.loss_dict = OrderedDict()

        if not self.use_network_d:
            self.train_net_g()
        else:
            for p in self.net_d.parameters():
                p.requires_grad = False
            if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
                self.train_net_g()

                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                    if l_g_percep is not None:
                        self.l_g_total += l_g_percep
                        self.loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        self.l_g_total += l_g_style
                        self.loss_dict['l_g_style'] = l_g_style

                # gan loss (relativistic gan)
                if self.cri_gan:
                    flat_mask = get_flat_mask(img=self.gt, scale=1)
                    self.output_det = self.output * (1-flat_mask)
                    self.gt_det = self.gt*(1-flat_mask)

                    real_d_pred = self.net_d(self.gt_det).detach()
                    fake_g_pred = self.net_d(self.output_det)
                    l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
                    l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
                    l_g_gan = (l_g_real + l_g_fake) / 2
                    self.l_g_total += l_g_gan
                    self.loss_dict['l_g_gan'] = l_g_gan

                self.l_g_total.backward()
                self.optimizer_g.step()

            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # gan loss (relativistic gan)

            # In order to avoid the error in distributed training:
            # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
            # the variables needed for gradient computation has been modified by
            # an inplace operation",
            # we separate the backwards for real and fake, and also detach the
            # tensor for calculating mean.

            # real
            fake_d_pred = self.net_d(self.output_det).detach()
            real_d_pred = self.net_d(self.gt_det)
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output_det.detach())
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
            l_d_fake.backward()
            self.optimizer_d.step()

            self.loss_dict['l_d_real'] = l_d_real
            self.loss_dict['l_d_fake'] = l_d_fake
            self.loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            self.loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(self.loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def tile_process(self, lq):
        """Modified from: https://github.com/ata4/esrgan-launcher
        """
        self.scale = self.opt['scale']
        self.img = lq
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        if hasattr(self, 'net_g_ema'):
                            output_tile = self.net_g_ema(input_tile)
                        else:
                            output_tile = self.net_g(input_tile)
                except Exception as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return self.output

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.opt['tile_process']:
                    self.output = self.tile_process(self.lq)
                else:
                    self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.opt['tile_process']:
                    self.output = self.tile_process(self.lq)
                else:
                    self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.opt['tile_process']:
                    out_list = [self.tile_process(aug) for aug in lq_list]
                else:
                    out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.opt['tile_process']:
                    out_list = [self.tile_process(aug) for aug in lq_list]
                else:
                    out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        self.logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        if self.use_network_d:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)