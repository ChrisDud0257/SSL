import functools
import torch
from torch.nn import functional as F
from einops import rearrange
import torch.nn as nn
import math
import random

from basicsr.losses.similarity.similaritywrapper import compute_similarity

# from basicsr.losses.similarity_p.similaritywrapper import compute_similarity as compute_similarity_p

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


def get_local_weights(residual, ksize):
    """Get local weights for generating the artifact map of LDL.

    It is only called by the `get_refined_artifact_map` function.

    Args:
        residual (Tensor): Residual between predicted and ground truth images.
        ksize (Int): size of the local window.

    Returns:
        Tensor: weight for each pixel to be discriminated as an artifact pixel
    """

    pad = (ksize - 1) // 2
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')

    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight

def get_artifact_map(img_gt, img_output, ksize):
    residual_sr = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

    patch_level_weight = torch.var(residual_sr.clone(), dim=(-1, -2, -3), keepdim=True) ** (1 / 5)
    pixel_level_weight = get_local_weights(residual_sr.clone(), ksize)
    overall_weight = patch_level_weight * pixel_level_weight

    return overall_weight


def get_refined_artifact_map(img_gt, img_output, img_ema, ksize):
    """Calculate the artifact map of LDL
    (Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution. In CVPR 2022)

    Args:
        img_gt (Tensor): ground truth images.
        img_output (Tensor): output images given by the optimizing model.
        img_ema (Tensor): output images given by the ema model.
        ksize (Int): size of the local window.

    Returns:
        overall_weight: weight for each pixel to be discriminated as an artifact pixel
        (calculated based on both local and global observations).
    """

    residual_ema = torch.sum(torch.abs(img_gt - img_ema), 1, keepdim=True)
    residual_sr = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

    patch_level_weight = torch.var(residual_sr.clone(), dim=(-1, -2, -3), keepdim=True) ** (1 / 5)
    pixel_level_weight = get_local_weights(residual_sr.clone(), ksize)
    overall_weight = patch_level_weight * pixel_level_weight

    overall_weight[residual_sr < residual_ema] = 0

    return overall_weight


def efdm_grad(x, y, grad=True, efdm_type='L1Loss'):
    assert x.shape == y.shape, f"The shape of x {x.shape} is not the same as the shape of y {y.shape}."
    b, c, h, w = x.shape
    _, index_x = torch.sort(x.view(b, c, -1))
    value_y, _ = torch.sort(y.view(b, c, -1))
    inverse_index_x = index_x.argsort(-1)
    if grad:
        transferred_x = x.view(b, c, -1) + value_y.gather(-1, inverse_index_x) - x.view(b, c, -1).detach()
        # print(f"Gradient")
    else:
        transferred_x = value_y.gather(-1, inverse_index_x)
        # print(f"No gradient")
    if efdm_type == 'KLDistanceLoss':
        # print(f"KL")
        return transferred_x.view(b, c, -1)
    else:
        return transferred_x.view(b, c, h, w)


def self_similarity(tensor, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32):
    b, c, h, w = tensor.shape
    x = tensor
    if is_shift:
        x = torch.roll(x, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
    q = rearrange(x, 'b c (h dh) (w dw) -> b h w (dh dw) c', dh=dh, dw=dw)
    s = (q @ q.transpose(-2, -1))
    s = s.softmax(dim=-1)
    s = rearrange(s, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)
    if is_shift:
        s = torch.roll(s, shifts=(shift_h, shift_w), dims=(1, 2))
    return s


def get_gradient(x):
    kernel_v = [[0, -1, 0],
                [0, 0, 0],
                [0, 1, 0]]
    kernel_h = [[0, 0, 0],
                [-1, 0, 1],
                [0, 0, 0]]
    tensor_kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
    tensor_kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
    weight_h = nn.Parameter(data=tensor_kernel_h, requires_grad=False).cuda()
    weight_v = nn.Parameter(data=tensor_kernel_v, requires_grad=False).cuda()
    x_list = []
    b, c, h, w = x.shape
    for i in range(c):
        x_c = x[:, i, :, :]
        x_grad_v = F.conv2d(x_c.unsqueeze(1), weight_v, padding=1)
        x_grad_h = F.conv2d(x_c.unsqueeze(1), weight_h, padding=1)
        x_grad = torch.sqrt(torch.pow(x_grad_v, 2) + torch.pow(x_grad_h, 2) + 1e-6)
        x_list.append(x_grad)
    grad = torch.cat(x_list, dim=1)
    return grad


def gradient_img_similarity(img, is_shift=False, shift_h=16, shift_w=16, dh=32, dw=32, gray=False, threshold=1e-3):
    if gray:
        img = (img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]) / 3
        img = img.unsqueeze(1)
    grad = get_gradient(img)
    b, c, h, w = grad.shape
    threshold_map = (torch.ones(b, c, h, w) * threshold).cuda()
    grad_map = grad.clone()
    grad_map[grad <= threshold_map] = 0
    if is_shift:
        grad_map = torch.roll(grad_map, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        img = torch.roll(img, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
    grad_map = rearrange(grad_map, 'b c (h dh) (w dw) -> b h w (dh dw) c', dh=dh, dw=dw)
    img = rearrange(img, 'b c (h dh) (w dw) -> b h w (dh dw) c', dh=dh, dw=dw)
    similarity_map = grad_map @ (img.transpose(-2, -1))
    similarity_map = similarity_map.softmax(dim=-1)
    similarity_map = rearrange(similarity_map, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)
    if is_shift:
        similarity_map = torch.roll(similarity_map, shifts=(shift_h, shift_w), dims=(1, 2))
    return similarity_map


class similarity_map():
    def __init__(self, img, mask=None, img_sr=None, simself_strategy='imgimg', is_shift=False, shift_h=16, shift_w=16,
                 dh=32, dw=32,
                 gray=False, threshold=2e-3, kernel_size=5, scaling_factor=4, softmax=True, rearrange_back=True,
                 crossentropy=False, temperature=0, stride=1, pix_num=1, index=None, kernel_size_center=9, mean=False, var = False,
                 largest_k = 0, gene_type = "sum"):
        super(similarity_map, self).__init__()

        if simself_strategy == 'imgimg':
            self.simself_imgimg(img=img, is_shift=is_shift, shift_h=shift_h, shift_w=shift_w, dh=dh, dw=dw)

        elif simself_strategy == 'gradimg':
            self.simself_gradimg(img=img, is_shift=is_shift, shift_h=shift_h, shift_w=shift_w, dh=dh, dw=dw, gray=gray,
                                 threshold=threshold)

        elif simself_strategy == 'areaarea':
            self.simself_areaarea(img=img, is_shift=is_shift, shift_h=shift_h, shift_w=shift_w, dh=dh, dw=dw,
                                  kernel_size=kernel_size, softmax=softmax, rearrange_back=rearrange_back,
                                  crossentropy=crossentropy, temperature=temperature,
                                  mean=mean)

        elif simself_strategy == 'areaarea_ori':
            self.simself_areaarea_ori(img=img, is_shift=is_shift, shift_h=shift_h, shift_w=shift_w, dh=dh, dw=dw,
                                      kernel_size=kernel_size, mean=mean)

        elif simself_strategy == 'gradgrad':
            self.simself_gradgrad(img=img, is_shift=is_shift, shift_h=shift_h, shift_w=shift_w, dh=dh, dw=dw, gray=gray,
                                  threshold=threshold)

        elif simself_strategy == 'areaarea_nonlocal':
            self.simself_areaarea_nonlocal(img=img, is_shift=is_shift, shift_h=shift_h, shift_w=shift_w, dh=dh, dw=dw,
                                           kernel_size=kernel_size, scaling_factor=scaling_factor)
        elif simself_strategy == 'areaarea_nonlocal_slow':
            self.simself_areaarea_nonlocal_slow(img=img, is_shift=is_shift, shift_h=shift_h, shift_w=shift_w, dh=dh,
                                                dw=dw,
                                                kernel_size=kernel_size, scaling_factor=scaling_factor)

        elif simself_strategy == 'areaarea_cos':
            self.simself_areaarea_cos(img=img, is_shift=is_shift, shift_h=shift_h, shift_w=shift_w, dh=dh, dw=dw,
                                      kernel_size=kernel_size, softmax=softmax, rearrange_back=rearrange_back,
                                      crossentropy=crossentropy, temperature=temperature)

        elif simself_strategy == 'areaarea_stride':
            self.simself_areaarea_stride(img=img, is_shift=is_shift, shift_h=shift_h, shift_w=shift_w, dh=dh, dw=dw,
                                         kernel_size=kernel_size, softmax=softmax, rearrange_back=rearrange_back,
                                         crossentropy=crossentropy, temperature=temperature, stride=stride)

        elif simself_strategy == 'areaarea_pad_roll':
            self.simself_areaarea_pad_roll(img=img, is_shift=is_shift, shift_h=shift_h, shift_w=shift_w, dh=dh, dw=dw,
                                           kernel_size=kernel_size, softmax=softmax, rearrange_back=rearrange_back,
                                           crossentropy=crossentropy, temperature=temperature)

        elif simself_strategy == 'areaarea_gradfilter':
            self.simself_gradfilter(img=img, is_shift=False, shift_h=4, shift_w=4, dh=dh, dw=dw,
                                    kernel_size=kernel_size, softmax=softmax,
                                    rearrange_back=rearrange_back, crossentropy=crossentropy, temperature=temperature,
                                    pix_num=pix_num, gray=gray, index=index)
        elif simself_strategy == 'areaarea_mask_nonlocal':
            self.simself_mask_nonlocal(img=img, mask=mask, kernel_size_search=kernel_size,
                                       kernel_size_center=kernel_size_center, sigma=scaling_factor,
                                       softmax=softmax)

        elif simself_strategy == 'areaarea_mask_trans':
            self.simself_mask_trans(img=img, mask=mask, kernel_size_search=kernel_size,
                                    kernel_size_center=kernel_size_center, mean=mean, softmax=softmax, var=var)

        elif simself_strategy == 'areaarea_mask_nonlocal_slow':
            self.simself_mask_nonlocal_slow(img=img, mask=mask, kernel_size_search=kernel_size,
                                            kernel_size_center=kernel_size_center, sigma=scaling_factor,
                                            softmax=softmax)

        elif simself_strategy == 'areaarea_mask_nonlocal_patch':
            self.simself_mask_nonlocal_patch(img=img, mask=mask, kernel_size_search=kernel_size,
                                             kernel_size_center=kernel_size_center, sigma=scaling_factor,
                                             softmax=softmax, dh=dh, dw=dw)

        elif simself_strategy == 'areaarea_mask_trans_patch':
            self.simself_mask_trans_patch(img=img, mask=mask, kernel_size_search=kernel_size,
                                          kernel_size_center=kernel_size_center, softmax=softmax, dh=dh, dw=dw,
                                          mean=mean)

        elif simself_strategy == 'areaarea_mask_nonlocal_patch_mutual':
            self.simmutual_mask_nonlocal_patch(img_gt=img, img_sr = img_sr, mask=mask, kernel_size_search=kernel_size,
                                             kernel_size_center=kernel_size_center, sigma=scaling_factor,
                                             softmax=softmax, dh=dh, dw=dw)

        elif simself_strategy == 'areaarea_mask_nonlocal_cuda_v1':
            self.simself_mask_nonlocal_cuda_v1(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax)

        elif simself_strategy == 'areaarea_mask_nonlocal_cuda_v1_patch':
            self.simself_mask_nonlocal_cuda_v1_patch(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax,
                                                     dh=dh, dw=dw)

        elif simself_strategy == 'areaarea_mask_nonlocal_cuda_v2':
            self.simself_mask_nonlocal_cuda_v2(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax)

        elif simself_strategy == 'areaarea_mask_nonlocalavg_cuda_v1': #v21
            self.simself_mask_nonlocalavg_cuda_v1(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax)

        elif simself_strategy == 'areaarea_mask_nonlocalavg_cuda_v2':
            self.simself_mask_nonlocalavg_cuda_v2(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax)

        elif simself_strategy == 'areaarea_mask_eulardistanceavg_cuda_v1':
            self.simself_mask_eulardistanceavg_cuda_v1(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax)

        elif simself_strategy == 'areaarea_mask_nonlocalavg_cuda_v3':
            self.simself_mask_nonlocalavg_cuda_v3(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax)

        elif simself_strategy == 'areaarea_mask_nonlocalavg_cuda_v4':
            self.simself_mask_nonlocalavg_cuda_v4(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax)

        elif simself_strategy == 'areaarea_mask_nonlocalavg_cuda_v1RGB':
            self.simself_mask_nonlocalavg_cuda_v1RGB(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax)

        elif simself_strategy == 'areaarea_mask_nonlocalavg_cuda_v5':
            self.simself_mask_nonlocalavg_cuda_v5(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax, gene_type=gene_type, largest_k=largest_k)

        elif simself_strategy == "areaarea_mask_nonlocalavg_cuda_maxh_v1": ###v28
            self.simself_mask_nonlocalavg_cuda_maxh_v1(img_gt = img, img_sr = img_sr, mask = mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax)

        elif simself_strategy == 'areaarea_mask_nonlocalavg_cuda_v1_p': #v29
            self.simself_mask_nonlocalavg_cuda_v1_p(img=img, mask=mask, kernel_size_search=kernel_size, kernel_size_center=kernel_size_center, sigma=scaling_factor, softmax=softmax)

    def simself_imgimg(self, img, is_shift=False, shift_h=16, shift_w=16, dh=32, dw=32, softmax=True):
        b, c, h, w = img.shape
        x = img
        if is_shift:
            x = torch.roll(x, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        q = rearrange(x, 'b c (h dh) (w dw) -> b h w (dh dw) c', dh=dh, dw=dw)
        s = q @ (q.transpose(-2, -1))
        if softmax:
            s = s.softmax(dim=-1)
        s = rearrange(s, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)
        if is_shift:
            s = torch.roll(s, shifts=(shift_h, shift_w), dims=(1, 2))
        self.s = s

    def simself_gradimg(self, img, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32, gray=False, threshold=2e-3,
                        softmax=True):
        if gray:
            img_ = (img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]) / 3
            img_ = img_.unsqueeze(1)
        else:
            img_ = img
        grad = self.get_gradient(img_)
        b, c, h, w = grad.shape
        threshold_map = (torch.ones(b, c, h, w) * threshold).cuda()
        grad_map = grad.clone()
        grad_map[grad <= threshold_map] = 0
        if is_shift:
            grad_map = torch.roll(grad_map, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
            img_ = torch.roll(img_, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        grad_map_patch = rearrange(grad_map, 'b c (h dh) (w dw) -> b h w (dh dw) c', dh=dh, dw=dw)
        img_patch = rearrange(img_, 'b c (h dh) (w dw) -> b h w (dh dw) c', dh=dh, dw=dw)
        similarity_map = grad_map_patch @ (img_patch.transpose(-2, -1))
        if softmax:
            similarity_map = similarity_map.softmax(dim=-1)
        similarity_map = rearrange(similarity_map, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)
        if is_shift:
            similarity_map = torch.roll(similarity_map, shifts=(shift_h, shift_w), dims=(1, 2))
        self.s = similarity_map

    def simself_gradgrad(self, img, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32, gray=False, threshold=2e-3):
        if gray:
            img_ = (img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]) / 3
            img_ = img_.unsqueeze(1)
        else:
            img_ = img
        grad = self.get_gradient(img_)
        b, c, h, w = grad.shape
        threshold_map = (torch.ones(b, c, h, w) * threshold).cuda()
        grad_map = grad.clone()
        grad_map[grad <= threshold_map] = 0
        if is_shift:
            grad_map = torch.roll(grad_map, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        grad_map_patch = rearrange(grad_map, 'b c (h dh) (w dw) -> b h w (dh dw) c', dh=dh, dw=dw)
        s = grad_map_patch @ (grad_map_patch.transpose(-2, -1))
        s = s.softmax(dim=-1)
        s = rearrange(s, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)
        if is_shift:
            s = torch.roll(s, shifts=(shift_h, shift_w), dims=(1, 2))
        self.s = s

    def simself_areaarea(self, img, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32, kernel_size=5, softmax=True,
                         rearrange_back=True, crossentropy=False, temperature=1, mean=False):
        b, c, h, w = img.shape
        x = img
        if is_shift:
            x = torch.roll(x, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        q = rearrange(x, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        q = q.permute(0, 1, 2, 4, 3, 5)  # b,c,h,w,dh,dw
        b, c, h, w, dh, dw = q.shape
        q = q.reshape(b, c * h * w, dh, dw)
        q = F.unfold(input=q, kernel_size=kernel_size, padding=kernel_size // 2,
                     stride=1)  # b,c*h*w*kernel_size*kernel_size, dh*dw
        q = q.reshape(b, c, h * w, kernel_size * kernel_size, dh * dw)
        q = q.permute(0, 2, 4, 1, 3)  # b,h*w, dh*dw, c, kernel_size*kernel_size
        if mean:
            q = q - torch.mean(input=q, dim=-1, keepdim=True)
        q = q.reshape(b, h, w, dh * dw, c * kernel_size * kernel_size)
        s = q @ (q.transpose(-2, -1))  # b, h, w, dh*dw, dh*dw
        if temperature != 0:
            s = s / temperature
            # if softmax:
            #     print(f"GT-Softmax is {s.softmax(dim = -1)[0][0][0][0][:20]}")
            # else:
            #     print(f"SR-Softmax is {s.softmax(dim=-1)[0][0][0][0][:20]}")
        if softmax:
            s = s.softmax(dim=-1)
        if crossentropy:
            s = s.reshape(b * h * w * dh * dw, dh * dw)
        else:
            if rearrange_back:
                s = rearrange(s, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)  # b,dh*dw, h*dh, w*dw
                if is_shift:
                    s = torch.roll(s, shifts=(shift_h, shift_w), dims=(1, 2))
        self.s = s

    def simself_areaarea_ori(self, img, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32, kernel_size=5, mean=False):
        b, c, h, w = img.shape
        x = img
        if is_shift:
            x = torch.roll(x, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        q = rearrange(x, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        q = q.permute(0, 1, 2, 4, 3, 5)
        b, c, h, w, dh, dw = q.shape
        q = q.reshape(b, c * h * w, dh, dw)
        q = F.unfold(input=q, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        q = q.reshape(b, c, h * w, kernel_size * kernel_size, dh * dw)
        q = q.permute(0, 2, 4, 1, 3)  # b,h*w,dh*dw,c,kernel_size*kernel_size
        if mean:
            q = q - torch.mean(input=q, dim=-1, keepdim=True)
        q = q.reshape(b, h, w, dh * dw, c * kernel_size * kernel_size)
        s = q @ (q.transpose(-2, -1))
        s = s.softmax(dim=-1)
        s = rearrange(s, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)
        if is_shift:
            s = torch.roll(s, shifts=(shift_h, shift_w), dims=(2, 3))
        self.s = s

    def simself_areaarea_nonlocal(self, img, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32, kernel_size=3,
                                  scaling_factor=1):
        # compute the attention map in non-local means
        # b, c, h, w = img.shape
        x = img
        if is_shift:
            x = torch.roll(x, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        q = rearrange(x, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        q = q.permute(0, 1, 2, 4, 3, 5)  # b,c,h,w,dh,dw
        b, c, h, w, dh, dw = q.shape
        q = q.reshape(b, c * h * w, dh, dw)
        q = F.unfold(input=q, kernel_size=kernel_size, padding=kernel_size // 2,
                     stride=1)  # b,c*h*w*kernel_size*kernel_size, dh*dw
        q = q.unsqueeze(-1) - q.unsqueeze(-2)  # b, c*h*w*kernel_size*kernel_size, dh*dw, dh*dw
        q = q.reshape(b, c, h * w, kernel_size * kernel_size, dh * dw, dh * dw).permute(0, 2, 1, 3, 4,
                                                                                        5)  # b, h*w, c, kernel_size * kernel_size, dh*dw, dh*dw
        q = q.reshape(b, h * w, c * kernel_size * kernel_size, dh * dw, dh * dw)
        q = q.abs().pow(2).sum(2) / (c * math.pow(kernel_size, 2))  # b, h*w, dh*dw, dh*dw
        q = torch.exp(input=(-1 * q / scaling_factor))  # b, h*w, dh*dw, dh*dw
        q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q  # b, h*w, dh*dw, dh*dw
        q = q.reshape(b, h, w, dh * dw, dh * dw)
        q = rearrange(q, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)
        if is_shift:
            q = torch.roll(q, shifts=(shift_h, shift_w), dims=(1, 2))
        self.s = q

    def simself_areaarea_nonlocal_slow(self, img, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32, kernel_size=3,
                                       scaling_factor=1):
        # compute the attention map in non-local means
        # b, c, h, w = img.shape
        x = img
        if is_shift:
            x = torch.roll(x, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        q = rearrange(x, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        q = q.permute(0, 1, 2, 4, 3, 5)  # b,c,h,w,dh,dw
        b, c, h, w, dh, dw = q.shape
        q = q.reshape(b, c * h * w, dh, dw)
        q = F.unfold(input=q, kernel_size=kernel_size, padding=kernel_size // 2,
                     stride=1)  # b,c*h*w*kernel_size*kernel_size, dh*dw
        q = q.reshape(b, c, h * w, kernel_size * kernel_size, dh * dw)
        q = q.permute(0, 2, 1, 3, 4)  # b,h*w, c, kernel_size * kernel_size, dh * dw
        q = q.reshape(b, h * w, c * kernel_size * kernel_size, dh * dw)

        list2 = []
        for i in range(dh * dw):
            list1 = []
            for j in range(dh * dw):
                list1.append((q[:, :, :, i] - q[:, :, :, j]).pow(2).sum(2))
            s = torch.stack(list1, dim=2)  # b, h*w, dh*dw
            list2.append(s)
        ss = torch.stack(list2, dim=3)  # b, h*w ,dh*dw, dh*dw
        ss = ss / (c * math.pow(kernel_size, 2))
        ss = torch.exp(input=(-1 * ss / scaling_factor))
        ss = 1 / (ss.max()) * ss
        ss = ss.reshape(b, h, w, dh * dw, dh * dw)
        ss = rearrange(ss, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)

        if is_shift:
            ss = torch.roll(ss, shifts=(shift_h, shift_w), dims=(1, 2))
        self.s = ss

    def simself_areaarea_cos(self, img, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32, kernel_size=5, softmax=True,
                             rearrange_back=True, crossentropy=False, temperature=1):
        b, c, h, w = img.shape
        x = img
        if is_shift:
            x = torch.roll(x, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        q = rearrange(x, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        q = q.permute(0, 1, 2, 4, 3, 5)  # b,c,h,w,dh,dw
        b, c, h, w, dh, dw = q.shape
        q = q.reshape(b, c * h * w, dh, dw)
        q = F.unfold(input=q, kernel_size=kernel_size, padding=kernel_size // 2,
                     stride=1)  # b,c*h*w*kernel_size*kernel_size, dh*dw
        q = q.reshape(b, c, h * w, kernel_size * kernel_size, dh * dw)
        q = q.permute(0, 2, 4, 1, 3)  # b,h*w, dh*dw, c, kernel_size*kernel_size
        q = q.reshape(b, h, w, dh * dw, c * kernel_size * kernel_size)
        q = q / (torch.norm(input=q, p=2, dim=-1).unsqueeze(-1) + 1e-6)
        s = q @ (q.transpose(-2, -1))  # b, h, w, dh*dw, dh*dw
        if temperature != 0:
            s = s / temperature
            # if softmax:
            #     print(f"GT-Softmax is {s.softmax(dim = -1)[0][0][0][0][:20]}")
            # else:
            #     print(f"SR-Softmax is {s.softmax(dim=-1)[0][0][0][0][:20]}")
        if softmax:
            s = s.softmax(dim=-1)
        if crossentropy:
            s = s.reshape(b * h * w * dh * dw, dh * dw)
        else:
            if rearrange_back:
                s = rearrange(s, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)  # b,dh*dw, h*dh, w*dw
                if is_shift:
                    s = torch.roll(s, shifts=(shift_h, shift_w), dims=(1, 2))
        self.s = s

    def simself_areaarea_stride(self, img, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32, kernel_size=5,
                                softmax=True,
                                rearrange_back=True, crossentropy=False, temperature=1, stride=1):
        b, c, h, w = img.shape
        x = img
        if is_shift:
            x = torch.roll(x, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        q = rearrange(x, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        q = q.permute(0, 1, 2, 4, 3, 5)  # b,c,h,w,dh,dw
        b, c, h, w, dh, dw = q.shape
        q = q.reshape(b, c * h * w, dh, dw)
        q = F.unfold(input=q, kernel_size=kernel_size, padding=math.ceil((kernel_size - stride) / 2),
                     stride=stride)  # b,c*h*w*kernel_size*kernel_size, dh*dw
        dh1 = dh // stride
        dw1 = dw // stride
        # print(f"shape is {q.shape}")
        q = q.reshape(b, c, h * w, kernel_size * kernel_size, dh1 * dw1)
        q = q.permute(0, 2, 4, 1, 3)  # b,h*w, dh*dw, c, kernel_size*kernel_size
        q = q.reshape(b, h, w, dh1 * dw1, c * kernel_size * kernel_size)
        # q = q / (torch.norm(input=q, p=2, dim=-1).unsqueeze(-1) + 1e-6)
        s = q @ (q.transpose(-2, -1))  # b, h, w, dh*dw, dh*dw
        if temperature != 0:
            s = s / temperature
            # if softmax:
            #     print(f"GT-Softmax is {s.softmax(dim = -1)[0][0][0][0][:20]}")
            # else:
            #     print(f"SR-Softmax is {s.softmax(dim=-1)[0][0][0][0][:20]}")
        if softmax:
            s = s.softmax(dim=-1)
        if crossentropy:
            s = s.reshape(b * h * w * dh1 * dw1, dh1 * dw1)
        else:
            if rearrange_back:
                s = rearrange(s, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh1, dw=dw1)  # b,dh*dw, h*dh, w*dw
                if is_shift:
                    s = torch.roll(s, shifts=(shift_h, shift_w), dims=(1, 2))
        self.s = s

    def simself_areaarea_pad_roll(self, img, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32, kernel_size=5,
                                  softmax=True,
                                  rearrange_back=True, crossentropy=False, temperature=1):
        b, c, h, w = img.shape
        x = img

        q = rearrange(x, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        q = q.permute(0, 1, 2, 4, 3, 5)  # b,c,h,w,dh,dw
        b, c, h, w, dh, dw = q.shape
        q = q.reshape(b, c * h * w, dh, dw)
        q = F.pad(input=q, pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode="reflect")
        q = F.unfold(input=q, kernel_size=kernel_size, padding=0, stride=1)  # b,c*h*w*kernel_size*kernel_size, dh*dw
        # print(f"q shape is {q.shape}")
        q = q.reshape(b, c, h * w, kernel_size * kernel_size, dh * dw)
        q = q.permute(0, 2, 4, 1, 3)  # b,h*w, dh*dw, c, kernel_size*kernel_size
        q = q.reshape(b, h, w, dh * dw, c * kernel_size * kernel_size)

        x = torch.roll(x, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        # print(f"shift_h is {shift_h}")

        q1 = rearrange(x, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        q1 = q1.permute(0, 1, 2, 4, 3, 5)  # b,c,h,w,dh,dw
        b, c, h, w, dh, dw = q1.shape
        q1 = q1.reshape(b, c * h * w, dh, dw)
        q1 = F.pad(input=q1, pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                   mode="reflect")
        q1 = F.unfold(input=q1, kernel_size=kernel_size, padding=0, stride=1)  # b,c*h*w*kernel_size*kernel_size, dh*dw
        q1 = q1.reshape(b, c, h * w, kernel_size * kernel_size, dh * dw)
        q1 = q1.permute(0, 2, 4, 1, 3)  # b,h*w, dh*dw, c, kernel_size*kernel_size
        q1 = q1.reshape(b, h, w, dh * dw, c * kernel_size * kernel_size)

        s = q @ (q1.transpose(-2, -1))  # b, h, w, dh*dw, dh*dw
        if temperature != 0:
            s = s / temperature
            # if softmax:
            #     print(f"GT-Softmax is {s.softmax(dim = -1)[0][0][0][0][:20]}")
            # else:
            #     print(f"SR-Softmax is {s.softmax(dim=-1)[0][0][0][0][:20]}")
        if softmax:
            s = s.softmax(dim=-1)
        if crossentropy:
            s = s.reshape(b * h * w * dh * dw, dh * dw)
        else:
            if rearrange_back:
                s = rearrange(s, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw)  # b,dh*dw, h*dh, w*dw
                # if is_shift:
                #     s = torch.roll(s, shifts=(shift_h, shift_w), dims=(2, 3))
        self.s = s

    def simself_gradfilter(self, img, is_shift=False, shift_h=4, shift_w=4, dh=32, dw=32, kernel_size=5, softmax=True,
                           rearrange_back=True, crossentropy=False, temperature=1, pix_num=0.75, gray=False,
                           index=None):
        if is_shift:
            img = torch.roll(img, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        if index is None:
            if gray:
                img = (img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]) / 3
                img = img.unsqueeze(1)
            grad = self.get_gradient(img)
            # b,c,h,w = grad.shape
            q_grad = rearrange(grad, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
            q_grad = q_grad.permute(0, 1, 2, 4, 3, 5)
            b, c, h, w, dh, dw = q_grad.shape
            q_grad = q_grad.reshape(b, c * h * w, dh, dw)
            # q_grad = F.unfold(input=q_grad, kernel_size=kernel_size, padding=kernel_size // 2, stride=1) #b,c*h*w*kernel_size*kernel_size, dh*dw

            q_grad = F.pad(input=q_grad, pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                           mode="reflect")
            q_grad = F.unfold(input=q_grad, kernel_size=kernel_size, padding=0,
                              stride=1)  # b,c*h*w*kernel_size*kernel_size, dh*dw

            q_grad = q_grad.reshape(b, c, h * w, kernel_size * kernel_size, dh * dw)
            q_grad = q_grad.permute(0, 2, 4, 1, 3)  # b,h*w, dh*dw, c, kernel_size*kernel_size
            q_grad = q_grad.reshape(b, h, w, dh * dw, c * kernel_size * kernel_size)
            q_grad = torch.norm(input=q_grad, p=2, dim=-1).unsqueeze(-1)  # b,h,w, dh*dw, 1
            _, sorted_indices = torch.sort(q_grad, descending=True, dim=-2)
            select_indices = sorted_indices[:, :, :, :int(dh * dw * pix_num), :]
        else:
            select_indices = index

        q = rearrange(img, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        q = q.permute(0, 1, 2, 4, 3, 5)
        b, c, h, w, dh, dw = q.shape
        q = q.reshape(b, c * h * w, dh, dw)
        # q = F.unfold(input=q, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)  # b,c*h*w*kernel_size,kernel_size

        q = F.pad(input=q, pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode="reflect")
        q = F.unfold(input=q, kernel_size=kernel_size, padding=0, stride=1)  # b,c*h*w*kernel_size*kernel_size, dh*dw

        q = q.reshape(b, c, h * w, kernel_size * kernel_size, dh * dw)
        q = q.permute(0, 2, 4, 1, 3)  # b,h*w, dh*dw, c, kernel_size*kernel_size
        q = q.reshape(b, h, w, dh * dw, c * kernel_size * kernel_size)
        q = torch.gather(q, dim=-2, index=select_indices)  # b,h,w,dh1*dw1,c*kernel_size*kernel_size
        q = q - torch.mean(q, dim=-1, keepdim=True)

        _, _, _, k, _ = q.shape
        s = q @ (q.transpose(-2, -1))  # b, h, w, dh1*dw1, dh1*dw1
        if temperature != 0:
            s = s / temperature
            # if softmax:
            #     print(f"GT-Softmax is {s.softmax(dim = -1)[0][0][0][0][:20]}")
            # else:
            #     print(f"SR-Softmax is {s.softmax(dim=-1)[0][0][0][0][:20]}")
        if softmax:
            s = s.softmax(dim=-1)
        if crossentropy:
            s = s.reshape(b * h * w * k, k)
        self.s = s
        self.index = select_indices

    def simself_mask_nonlocal(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False):
        # img, 1*3*h*w
        # mask, 1*1*h*w
        b, c, h, w = img.shape
        # print(f"mask shape is {mask.shape}")
        _, c1, _, _ = mask.shape

        img_search_area = F.pad(input=img, pad=(
        kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2),
                                mode="reflect")
        img_search_area = F.unfold(input=img_search_area, padding=0, kernel_size=kernel_size_search,
                                   stride=1)  # 1,3*k_search*k_search, h*w

        mask = F.unfold(input=mask, padding=0, kernel_size=1, stride=1)  # 1,1*1*1, h*w
        index = torch.where(mask == 1)

        img_search_area = img_search_area[:, :, index[
                                                    -1]]  # 1, 3*k_search*k_search, num         num is the total amount of the pixels which is 1 in the mask
        del mask
        del index
        _, _, num = img_search_area.shape
        img_search_area = img_search_area.reshape(b, c, kernel_size_search * kernel_size_search, num)
        img_search_area = img_search_area.permute(0, 1, 3, 2)  # 1, 3, num, k_search*k_search
        img_search_area = img_search_area.reshape(b, c * num, kernel_size_search,
                                                  kernel_size_search)  # 1,3*num, k_search, k_search

        img_search_area = F.unfold(input=img_search_area, kernel_size=kernel_size_center,
                                   padding=kernel_size_center // 2, stride=1)  # 1, 3*num*k_c*k_c, k_s*k_s
        img_search_area = img_search_area.reshape(b, c, num, kernel_size_center * kernel_size_center,
                                                  kernel_size_search * kernel_size_search)
        img_search_area = img_search_area.permute(0, 2, 1, 3, 4)  # 1, num, 3, k_c*k_c, k_s*k_s
        img_search_area = img_search_area.reshape(b, num, c * kernel_size_center * kernel_size_center,
                                                  kernel_size_search * kernel_size_search)  # 1, num, c*k_c*k_c, k_s*k_s

        # img_center_neighbor = F.pad(input=img, pad=(kernel_size_center // 2, kernel_size_center // 2, kernel_size_center // 2, kernel_size_center // 2), mode="reflect")
        # img_center_neighbor = F.unfold(input=img_center_neighbor, padding=0, kernel_size=kernel_size_center, stride=1)  # 1,c*k_c*k_c, h*w
        # img_center_neighbor = img_center_neighbor[:, :, index[-1]]  # 1, c*k_c*k_c, num      num is the total amount of the pixels which is 1 in the mask
        # img_center_neighbor = img_center_neighbor.unsqueeze(-1)  # 1, c*k_c*k_c, num, 1
        # img_center_neighbor = img_center_neighbor.permute(0, 2, 1, 3)  # 1, num, c*k_c*k_c, 1

        img_center_neighbor = img_search_area[:, :, :, (kernel_size_search * kernel_size_search) // 2].unsqueeze(
            -1)  # 1, num, c*k_c*k_c, 1

        q = img_search_area - img_center_neighbor  # 1, num, c*k_c*k_c, k_s*k_s
        # print(f"q shape is {q.shape}")
        del img_search_area
        del img_center_neighbor
        q = q.pow(2).sum(2)  # 1, num, k_s*k_s
        q = torch.exp(-1 * q / sigma)
        if softmax:
            q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q
        self.s = q
        del q

    def simself_mask_nonlocal_slow(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0,
                                   softmax=False):
        # img, 1*3*h*w
        # mask, 1*1*h*w
        b, c, h, w = img.shape
        # print(f"mask shape is {mask.shape}")
        _, c1, _, _ = mask.shape

        img_search_area = F.pad(input=img, pad=(
        kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2),
                                mode="reflect")
        img_search_area = F.unfold(input=img_search_area, padding=0, kernel_size=kernel_size_search,
                                   stride=1)  # 1,3*k_search*k_search, h*w

        mask = F.unfold(input=mask, padding=0, kernel_size=1, stride=1)  # 1,1*1*1, h*w
        index = torch.where(mask == 1)

        img_search_area = img_search_area[:, :, index[
                                                    -1]]  # 1, 3*k_search*k_search, num         num is the total amount of the pixels which is 1 in the mask
        _, _, num = img_search_area.shape
        img_search_area = img_search_area.reshape(b, c, kernel_size_search, kernel_size_search, num)
        img_search_area = img_search_area.permute(0, 4, 1, 2, 3)  # 1, num, 3, k_search, k_search
        img_search_area = img_search_area.reshape(b, num * c, kernel_size_search,
                                                  kernel_size_search)  # 1, num*3, k_s, k_s

        img_search_area = F.pad(input=img_search_area, pad=(
        kernel_size_center // 2, kernel_size_center // 2, kernel_size_center // 2, kernel_size_center // 2),
                                mode="reflect")
        img_search_area = img_search_area.reshape(b, num, c, kernel_size_search + kernel_size_center // 2 * 2,
                                                  kernel_size_search + kernel_size_center // 2 * 2)

        img_center_neighbor = F.pad(input=img, pad=(
        kernel_size_center // 2, kernel_size_center // 2, kernel_size_center // 2, kernel_size_center // 2),
                                    mode="reflect")
        img_center_neighbor = F.unfold(input=img_center_neighbor, padding=0, kernel_size=kernel_size_center,
                                       stride=1)  # 1,c*k_c*k_c, h*w
        img_center_neighbor = img_center_neighbor[:, :, index[
                                                            -1]]  # 1, c*k_c*k_c, num      num is the total amount of the pixels which is 1 in the mask
        img_center_neighbor = img_center_neighbor.reshape(b, c, kernel_size_center, kernel_size_center, num)
        img_center_neighbor = img_center_neighbor.permute(0, 4, 1, 2, 3)  # 1,num,3, k_c, k_c

        q_list = []
        for i in range(kernel_size_center // 2, kernel_size_center // 2 + kernel_size_search):
            for j in range(kernel_size_center // 2, kernel_size_center // 2 + kernel_size_search):
                img_neighbor = img_search_area[:, :, :, i - kernel_size_center // 2:i + kernel_size_center // 2 + 1,
                               j - kernel_size_center // 2:j + kernel_size_center // 2 + 1]  # 1, num, 3, k_c, k_c
                q = img_center_neighbor - img_neighbor  # 1, num, 3, k_c, k_c
                q = q.pow(2).sum(-1).sum(-1).sum(-1)  # 1, num
                q_list.append(q)
        q = torch.stack(q_list, dim=2)

        # q_list_1 = []
        # for k in range(0, num):
        #     q_list = []
        #     for i in range(kernel_size_center // 2, kernel_size_center // 2 + kernel_size_search):
        #         for j in range(kernel_size_center // 2, kernel_size_center // 2 + kernel_size_search):
        #             img_neighbor = img_search_area[:, k, :, i - kernel_size_center // 2:i + kernel_size_center // 2 + 1,
        #                            j - kernel_size_center // 2:j + kernel_size_center // 2 + 1] #1, 3, k_c, k_c
        #             q = img_center_neighbor[:, k, :, :, :] - img_neighbor #1, 3, k_c, k_c
        #             q = q.pow(2).sum().unsqueeze(-1) #1,1
        #             q_list.append(q)
        #     q_list = torch.stack(q_list, dim = 1) #1, k_s*k_s
        #     q_list_1.append(q_list)
        # q = torch.stack(q_list_1, dim = 1) #1, num, k_s*k_s

        # print(f"q shape is {q.shape}")
        q = torch.exp(-1 * q / sigma)
        if softmax:
            q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q
        self.s = q

    def simself_mask_trans(self, img, mask, kernel_size_search=25, kernel_size_center=9, mean=False, softmax=True, var = False):
        # img, 1*3*h*w
        # mask, 1*1*h*w
        b, c, h, w = img.shape
        # print(f"mask shape is {mask.shape}")
        _, c1, _, _ = mask.shape

        img_search_area = F.pad(input=img, pad=(
        kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2),
                                mode="reflect")
        img_search_area = F.unfold(input=img_search_area, padding=0, kernel_size=kernel_size_search,
                                   stride=1)  # 1,3*k_search*k_search, h*w

        mask = F.unfold(input=mask, padding=0, kernel_size=1, stride=1)  # 1,1*1*1, h*w
        index = torch.where(mask == 1)

        img_search_area = img_search_area[:, :, index[
                                                    -1]]  # 1, 3*k_search*k_search, num         num is the total amount of the pixels which is 1 in the mask
        _, _, num = img_search_area.shape
        img_search_area = img_search_area.reshape(b, c, kernel_size_search * kernel_size_search, num)
        img_search_area = img_search_area.permute(0, 1, 3, 2)  # 1, 3, num, k_search*k_search
        img_search_area = img_search_area.reshape(b, c * num, kernel_size_search,
                                                  kernel_size_search)  # 1,3*num, k_search, k_search

        img_search_area = F.unfold(input=img_search_area, kernel_size=kernel_size_center,
                                   padding=kernel_size_center // 2, stride=1)  # 1, 3*num*k_c*k_c, k_s*k_s
        img_search_area = img_search_area.reshape(b, c, num, kernel_size_center * kernel_size_center,
                                                  kernel_size_search * kernel_size_search)
        img_search_area = img_search_area.permute(0, 2, 1, 3, 4)  # 1, num, 3, k_c*k_c, k_s*k_s
        if mean:
            img_search_area = img_search_area - torch.mean(input=img_search_area, dim=-2, keepdim=True)
        if var:
            img_search_area = img_search_area / (torch.var(input=img_search_area, dim = -2, keepdim=True) + 1e-8)
        img_search_area = img_search_area.reshape(b, num, c * kernel_size_center * kernel_size_center,
                                                  kernel_size_search * kernel_size_search)  # 1, num, 3*k_c*k_c, k_s*k_s
        # print(f"search shape is {img_search_area.shape}")

        # img_center_neighbor = F.pad(input=img, pad=(
        # kernel_size_center // 2, kernel_size_center // 2, kernel_size_center // 2, kernel_size_center // 2),
        #                             mode="reflect")
        # img_center_neighbor = F.unfold(input=img_center_neighbor, padding=0, kernel_size=kernel_size_center,
        #                                stride=1)  # 1,c*k_c*k_c, h*w
        # img_center_neighbor = img_center_neighbor[:, :, index[
        #                                                     -1]]  # 1, c*k_c*k_c, num      num is the total amount of the pixels which is 1 in the mask
        # img_center_neighbor = img_center_neighbor.permute(0, 2, 1)  # 1, num, c*k_c*k_c
        # if mean:
        #     img_center_neighbor = img_center_neighbor.reshape(b, num, c,
        #                                                       kernel_size_center * kernel_size_center)  # 1,num,c,k_c*k_c
        #     img_center_neighbor = img_center_neighbor - torch.mean(input=img_center_neighbor, dim=-1, keepdim=True)
        #     img_center_neighbor = img_center_neighbor.reshape(b, num, c * kernel_size_center * kernel_size_center)
        # print(f"center shape is {img_center_neighbor.shape}")

        img_center_neighbor = img_search_area[:, :, :, (kernel_size_search * kernel_size_search) // 2] #1, num, 3*k_c*k_c

        q = torch.einsum('bnij,bni->bnj', img_search_area, img_center_neighbor)  # 1,num, k_s*k_s
        if softmax:
            q = q.softmax(dim=-1)
        self.s = q

    def simself_mask_nonlocal_patch(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0,
                                    softmax=False, dh=64, dw=64):
        # img, 1*3*h*w
        # mask, 1*1*h*w
        b, c, h, w = img.shape
        # print(f"mask shape is {mask.shape}")
        _, c1, _, _ = mask.shape

        img_search_area = rearrange(img, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        img_search_area = img_search_area.permute(0, 2, 4, 1, 3, 5)  # b,h,w,c,dh,dw
        img_search_area = img_search_area.reshape(b, h // dh * w // dw, c, dh, dw)
        mask = rearrange(mask, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        mask = mask.permute(0, 2, 4, 1, 3, 5)  # b,h,w,c1,dh,dw
        mask = mask.reshape(b, h // dh * w // dw, c1, dh, dw)

        q_list = []
        for i in range(0, h // dh * w // dw):
            img_search_area_patch = img_search_area[:, i, :, :, :]  # 1,3,dh,dw
            mask_patch = mask[:, i, :, :, :]  # 1,1,dh,dw
            if mask_patch.sum() != 0:
                img_search_area_patch = F.pad(input=img_search_area_patch,
                                              pad=(
                                              kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2,
                                              kernel_size_search // 2), mode="reflect")
                img_search_area_patch = F.unfold(input=img_search_area_patch, padding=0, kernel_size=kernel_size_search,
                                                 stride=1)  # 1,3*k_s*k_s, dh*dw
                mask_patch = F.unfold(input=mask_patch, padding=0, kernel_size=1, stride=1)  # 1,1*1*1, dh*dw
                index = torch.where(mask_patch == 1)

                img_search_area_patch = img_search_area_patch[:, :, index[
                                                                        -1]]  # 1, 3*k_s*k_s, num         num is the total amount of the pixels which is 1 in the mask
                del index
                del mask_patch
                _, _, num = img_search_area_patch.shape
                img_search_area_patch = img_search_area_patch.reshape(b, c, kernel_size_search * kernel_size_search,
                                                                      num)
                img_search_area_patch = img_search_area_patch.permute(0, 1, 3, 2)  # 1, 3, num, k_s*k_s
                img_search_area_patch = img_search_area_patch.reshape(b, c * num, kernel_size_search,
                                                                      kernel_size_search)  # 1,3*num, k_s, k_s

                img_search_area_patch = F.unfold(input=img_search_area_patch, kernel_size=kernel_size_center,
                                                 padding=kernel_size_center // 2, stride=1)  # 1, 3*num*k_c*k_c, k_s*k_s
                img_search_area_patch = img_search_area_patch.reshape(b, c, num,
                                                                      kernel_size_center * kernel_size_center,
                                                                      kernel_size_search * kernel_size_search)
                img_search_area_patch = img_search_area_patch.permute(0, 2, 1, 3, 4)  # 1, num, 3, k_c*k_c, k_s*k_s
                img_search_area_patch = img_search_area_patch.reshape(b, num,
                                                                      c * kernel_size_center * kernel_size_center,
                                                                      kernel_size_search * kernel_size_search)  # 1, num, c*k_c*k_c, k_s*k_s

                img_center_neighbor_patch = img_search_area_patch[:, :, :,
                                            (kernel_size_search * kernel_size_search) // 2].unsqueeze(
                    -1)  # 1, num, c*k_c*k_c, 1
                q = img_search_area_patch - img_center_neighbor_patch  # 1, num, c*k_c*k_c, k_s*k_s
                # print(f"q shape is {q.shape}")
                del img_search_area_patch
                del img_center_neighbor_patch
                q = q.pow(2).sum(2)  # 1, num, k_s*k_s
                q = torch.exp(-1 * q / sigma)
                if softmax:
                    q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q
                q_list.append(q)
                del q
        del img_search_area
        del mask
        self.s = torch.cat(q_list, dim=1)
        del q_list

    def simself_mask_trans_patch(self, img, mask, kernel_size_search=25, kernel_size_center=9, softmax=True, dh=64,
                                 dw=64, mean=True):
        # img, 1*3*h*w
        # mask, 1*1*h*w
        b, c, h, w = img.shape
        # print(f"mask shape is {mask.shape}")
        _, c1, _, _ = mask.shape

        img_search_area = rearrange(img, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        img_search_area = img_search_area.permute(0, 2, 4, 1, 3, 5)  # b,h,w,c,dh,dw
        img_search_area = img_search_area.reshape(b, h // dh * w // dw, c, dh, dw)
        mask = rearrange(mask, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        mask = mask.permute(0, 2, 4, 1, 3, 5)  # b,h,w,c1,dh,dw
        mask = mask.reshape(b, h // dh * w // dw, c1, dh, dw)

        q_list = []
        for i in range(0, h // dh * w // dw):
            img_search_area_patch = img_search_area[:, i, :, :, :]  # 1,3,dh,dw
            mask_patch = mask[:, i, :, :, :]  # 1,1,dh,dw
            if mask_patch.sum() != 0:
                img_search_area_patch = F.pad(input=img_search_area_patch,
                                              pad=(
                                              kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2,
                                              kernel_size_search // 2), mode="reflect")
                img_search_area_patch = F.unfold(input=img_search_area_patch, padding=0, kernel_size=kernel_size_search,
                                                 stride=1)  # 1,3*k_s*k_s, dh*dw
                mask_patch = F.unfold(input=mask_patch, padding=0, kernel_size=1, stride=1)  # 1,1*1*1, dh*dw
                index = torch.where(mask_patch == 1)

                img_search_area_patch = img_search_area_patch[:, :, index[
                                                                        -1]]  # 1, 3*k_s*k_s, num         num is the total amount of the pixels which is 1 in the mask
                del index
                del mask_patch
                _, _, num = img_search_area_patch.shape
                img_search_area_patch = img_search_area_patch.reshape(b, c, kernel_size_search * kernel_size_search,
                                                                      num)
                img_search_area_patch = img_search_area_patch.permute(0, 1, 3, 2)  # 1, 3, num, k_s*k_s
                img_search_area_patch = img_search_area_patch.reshape(b, c * num, kernel_size_search,
                                                                      kernel_size_search)  # 1,3*num, k_s, k_s

                img_search_area_patch = F.unfold(input=img_search_area_patch, kernel_size=kernel_size_center,
                                                 padding=kernel_size_center // 2, stride=1)  # 1, 3*num*k_c*k_c, k_s*k_s
                img_search_area_patch = img_search_area_patch.reshape(b, c, num,
                                                                      kernel_size_center * kernel_size_center,
                                                                      kernel_size_search * kernel_size_search)
                img_search_area_patch = img_search_area_patch.permute(0, 2, 1, 3, 4)  # 1, num, c, k_c*k_c, k_s*k_s

                if mean:
                    img_search_area_patch = img_search_area_patch - torch.mean(input=img_search_area_patch, dim=-2,
                                                                               keepdim=True)  # 1, num, c, k_c*k_c, k_s*k_s
                    img_search_area_patch = img_search_area_patch.reshape(b, num,
                                                                          c * kernel_size_center * kernel_size_center,
                                                                          kernel_size_search * kernel_size_search)
                else:
                    img_search_area_patch = img_search_area_patch.reshape(b, num,
                                                                          c * kernel_size_center * kernel_size_center,
                                                                          kernel_size_search * kernel_size_search)

                img_center_neighbor_patch = img_search_area_patch[:, :, :,
                                            (kernel_size_search * kernel_size_search) // 2]  # 1, num, c*k_c*k_c

                q = torch.einsum('bnij,bni->bnj', img_search_area_patch, img_center_neighbor_patch)  # 1,num, k_s*k_s

                q_1 = q[:, :, : kernel_size_search * kernel_size_search // 2]
                q_2 = q[:, :, kernel_size_search * kernel_size_search // 2 + 1:]

                q = torch.cat([q_1, q_2], dim=-1)
                del q_1
                del q_2

                del img_search_area_patch
                del img_center_neighbor_patch
                if softmax:
                    q = q.softmax(dim=-1)
                # print(f"q shape is {q.shape}")
                q_list.append(q)
                del q
        del img_search_area
        del mask
        self.s = torch.cat(q_list, dim=1)
        del q_list

    def simmutual_mask_nonlocal_patch(self, img_gt, img_sr, mask, kernel_size_search=25, kernel_size_center=9,
                                      sigma=1.0, softmax=False, dh=64, dw=64):
        # img, 1*3*h*w
        # mask, 1*1*h*w
        b, c, h, w = img_gt.shape
        # print(f"mask shape is {mask.shape}")
        _, c1, _, _ = mask.shape

        img_search_area_gt = rearrange(img_gt, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        img_search_area_gt = img_search_area_gt.permute(0, 2, 4, 1, 3, 5)  # b,h,w,c,dh,dw
        img_search_area_gt = img_search_area_gt.reshape(b, h // dh * w // dw, c, dh, dw)

        img_search_area_sr = rearrange(img_sr, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        img_search_area_sr = img_search_area_sr.permute(0, 2, 4, 1, 3, 5)  # b,h,w,c,dh,dw
        img_search_area_sr = img_search_area_sr.reshape(b, h // dh * w // dw, c, dh, dw)

        mask = rearrange(mask, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        mask = mask.permute(0, 2, 4, 1, 3, 5)  # b,h,w,c1,dh,dw
        mask = mask.reshape(b, h // dh * w // dw, c1, dh, dw)

        q_list = []
        q1_list = []
        for i in range(0, h // dh * w // dw):
            img_search_area_gt_patch = img_search_area_gt[:, i, :, :, :]  # 1,3,dh,dw
            img_search_area_sr_patch = img_search_area_sr[:, i, :, :, :]  # 1,3,dh,dw
            mask_patch = mask[:, i, :, :, :]  # 1,1,dh,dw
            if mask_patch.sum() != 0:
                img_search_area_gt_patch = F.pad(input=img_search_area_gt_patch, pad=(
                kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2),
                                                 mode="reflect")
                img_search_area_gt_patch = F.unfold(input=img_search_area_gt_patch, padding=0,
                                                    kernel_size=kernel_size_search, stride=1)  # 1,3*k_s*k_s, dh*dw
                mask_patch = F.unfold(input=mask_patch, padding=0, kernel_size=1, stride=1)  # 1,1*1*1, dh*dw
                index = torch.where(mask_patch == 1)
                del mask_patch

                img_search_area_gt_patch = img_search_area_gt_patch[:, :, index[
                                                                              -1]]  # 1, 3*k_s*k_s, num         num is the total amount of the pixels which is 1 in the mask

                _, _, num = img_search_area_gt_patch.shape
                img_search_area_gt_patch = img_search_area_gt_patch.reshape(b, c,
                                                                            kernel_size_search * kernel_size_search,
                                                                            num)
                img_search_area_gt_patch = img_search_area_gt_patch.permute(0, 1, 3, 2)  # 1, 3, num, k_s*k_s
                img_search_area_gt_patch = img_search_area_gt_patch.reshape(b, c * num, kernel_size_search,
                                                                            kernel_size_search)  # 1,3*num, k_s, k_s

                img_search_area_gt_patch = F.unfold(input=img_search_area_gt_patch, kernel_size=kernel_size_center,
                                                    padding=kernel_size_center // 2,
                                                    stride=1)  # 1, 3*num*k_c*k_c, k_s*k_s
                img_search_area_gt_patch = img_search_area_gt_patch.reshape(b, c, num,
                                                                            kernel_size_center * kernel_size_center,
                                                                            kernel_size_search * kernel_size_search)
                img_search_area_gt_patch = img_search_area_gt_patch.permute(0, 2, 1, 3,
                                                                            4)  # 1, num, 3, k_c*k_c, k_s*k_s
                img_search_area_gt_patch = img_search_area_gt_patch.reshape(b, num,
                                                                            c * kernel_size_center * kernel_size_center,
                                                                            kernel_size_search * kernel_size_search)  # 1, num, c*k_c*k_c, k_s*k_s

                img_center_neighbor_patch = img_search_area_gt_patch[:, :, :,
                                            (kernel_size_search * kernel_size_search) // 2].unsqueeze(
                    -1)  # 1, num, c*k_c*k_c, 1
                q = img_search_area_gt_patch - img_center_neighbor_patch  # 1, num, c*k_c*k_c, k_s*k_s
                # print(f"q shape is {q.shape}")
                del img_search_area_gt_patch

                q = q.pow(2).sum(2)  # 1, num, k_s*k_s
                q = torch.exp(-1 * q / sigma)
                if softmax:
                    q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q
                q_list.append(q)
                del q

                img_search_area_sr_patch = F.pad(input=img_search_area_sr_patch, pad=(
                kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2, kernel_size_search // 2),
                                                 mode="reflect")
                img_search_area_sr_patch = F.unfold(input=img_search_area_sr_patch, padding=0,
                                                    kernel_size=kernel_size_search, stride=1)  # 1,3*k_s*k_s, dh*dw

                img_search_area_sr_patch = img_search_area_sr_patch[:, :, index[
                                                                              -1]]  # 1, 3*k_s*k_s, num         num is the total amount of the pixels which is 1 in the mask
                del index

                img_search_area_sr_patch = img_search_area_sr_patch.reshape(b, c,
                                                                            kernel_size_search * kernel_size_search,
                                                                            num)
                img_search_area_sr_patch = img_search_area_sr_patch.permute(0, 1, 3, 2)  # 1, 3, num, k_s*k_s
                img_search_area_sr_patch = img_search_area_sr_patch.reshape(b, c * num, kernel_size_search,
                                                                            kernel_size_search)  # 1,3*num, k_s, k_s

                img_search_area_sr_patch = F.unfold(input=img_search_area_sr_patch, kernel_size=kernel_size_center,
                                                    padding=kernel_size_center // 2,
                                                    stride=1)  # 1, 3*num*k_c*k_c, k_s*k_s
                img_search_area_sr_patch = img_search_area_sr_patch.reshape(b, c, num,
                                                                            kernel_size_center * kernel_size_center,
                                                                            kernel_size_search * kernel_size_search)
                img_search_area_sr_patch = img_search_area_sr_patch.permute(0, 2, 1, 3,
                                                                            4)  # 1, num, 3, k_c*k_c, k_s*k_s
                img_search_area_sr_patch = img_search_area_sr_patch.reshape(b, num,
                                                                            c * kernel_size_center * kernel_size_center,
                                                                            kernel_size_search * kernel_size_search)  # 1, num, c*k_c*k_c, k_s*k_s

                q = img_search_area_sr_patch - img_center_neighbor_patch  # 1, num, c*k_c*k_c, k_s*k_s
                del img_search_area_sr_patch
                del img_center_neighbor_patch

                q = q.pow(2).sum(2)  # 1, num, k_s*k_s
                q = torch.exp(-1 * q / sigma)
                if softmax:
                    q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q
                q1_list.append(q)
                del q

        del img_search_area_gt
        del img_search_area_sr
        del mask
        self.s = torch.cat(q_list, dim=1)
        del q_list
        self.s1 = torch.cat(q1_list, dim=1)
        del q1_list

    def simself_mask_nonlocal_cuda_v1(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False):
        q = compute_similarity(image=img[0], mask=mask[0,0], psize=kernel_size_search, ksize=kernel_size_center)
        q = q.unsqueeze(0)
        b, num, _, _ = q.shape
        q = q.reshape(b, num, kernel_size_search * kernel_size_search)

        q = torch.exp(-1 * q / sigma)

        if softmax:
            q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q
        self.s = q
        del q

    def simself_mask_nonlocal_cuda_v1_patch(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False,
                                            dh = 64, dw = 64):
        # img, 1*3*h*w
        # mask, 1*1*h*w
        b, c, h, w = img.shape
        # print(f"mask shape is {mask.shape}")
        _, c1, _, _ = mask.shape

        img_search_area = rearrange(img, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        img_search_area = img_search_area.permute(0, 2, 4, 1, 3, 5)  # b,h,w,c,dh,dw
        img_search_area = img_search_area.reshape(b, h // dh * w // dw, c, dh, dw)
        mask = rearrange(mask, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
        mask = mask.permute(0, 2, 4, 1, 3, 5)  # b,h,w,c1,dh,dw
        mask = mask.reshape(b, h // dh * w // dw, c1, dh, dw)

        q_list = []
        for i in range(0, h // dh * w // dw):
            img_search_area_patch = img_search_area[:, i, :, :, :]  # 1,3,dh,dw
            mask_patch = mask[:, i, :, :, :]  # 1,1,dh,dw
            if mask_patch.sum() != 0:
                q = compute_similarity(image=img_search_area_patch[0], mask=mask_patch[0, 0], psize=kernel_size_search,
                                       ksize=kernel_size_center)
                q = q.unsqueeze(0)
                b, num, _, _ = q.shape
                q = q.reshape(b, num, kernel_size_search * kernel_size_search)
                q = torch.exp(-1 * q / sigma)
                if softmax:
                    q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q
                q_list.append(q)
                del q

        self.s = torch.cat(q_list, dim=1)
        del q_list

    def simself_mask_nonlocal_cuda_v2(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False):
        q = compute_similarity(image=img[0], mask=mask[0,0], psize=kernel_size_search, ksize=kernel_size_center)
        q = q.unsqueeze(0)
        b, num, _, _ = q.shape
        q = q.reshape(b, num, kernel_size_search * kernel_size_search)
        q = torch.sqrt(q + 1e-8)
        q = torch.exp(-1 * q / sigma)
        if softmax:
            q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q
        self.s = q
        del q

    def simself_mask_nonlocalavg_cuda_v1(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False):
        b,c,h,w = img.shape
        q = compute_similarity(image=img[0], mask=mask[0,0], psize=kernel_size_search, ksize=kernel_size_center)
        q = q / (c * math.pow(kernel_size_center, 2))
        q = q.unsqueeze(0)
        b, num, _, _ = q.shape
        q = q.reshape(b, num, kernel_size_search * kernel_size_search)

        q = torch.exp(-1 * q / sigma)

        if softmax:
            q = 1 / (torch.sum(q, dim=-1) + 1e-20).unsqueeze(-1) * q
        self.s = q
        del q

    def simself_mask_nonlocalavg_cuda_v2(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False):
        q = compute_similarity(image=img[0], mask=mask[0,0], psize=kernel_size_search, ksize=kernel_size_center)
        q = q / (3 * math.pow(kernel_size_center, 2))
        q = q.unsqueeze(0)
        b, num, _, _ = q.shape
        q = q.reshape(b, num, kernel_size_search * kernel_size_search)

        q = torch.exp(-1 * q / sigma)

        q_1 = q[:, :, : kernel_size_search * kernel_size_search // 2]
        q_2 = q[:, :, kernel_size_search * kernel_size_search // 2 + 1: ]
        q = torch.cat([q_1,  q_2], dim=-1)

        if softmax:
            q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q
        self.s = q
        del q

    def simself_mask_eulardistanceavg_cuda_v1(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False):
        q = compute_similarity(image=img[0], mask=mask[0, 0], psize=kernel_size_search, ksize=kernel_size_center)
        q = q / (3 * math.pow(kernel_size_center, 2))
        q = q.unsqueeze(0)
        b, num, _, _ = q.shape
        q = q.reshape(b, num, kernel_size_search * kernel_size_search)
        q = q / sigma

        q_1 = q[:, :, : kernel_size_search * kernel_size_search // 2]
        q_2 = q[:, :, kernel_size_search * kernel_size_search // 2 + 1: ]
        q = torch.cat([q_1,  q_2], dim=-1)
        del q_1
        del q_2

        if softmax:
            q = -1 * q
            q = q.softmax(dim=-1)
        self.s = q
        del q

    def simself_mask_nonlocalavg_cuda_v3(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False):
        q = compute_similarity(image=img[0], mask=mask[0,0], psize=kernel_size_search, ksize=kernel_size_center)
        q = q / (3 * math.pow(kernel_size_center, 2))
        q = q.unsqueeze(0)
        b, num, _, _ = q.shape
        q = q.reshape(b, num, kernel_size_search * kernel_size_search)

        q = torch.exp(-1 * q / sigma)

        weight = q.sum(dim=-1, keepdim=True) / (math.pow(kernel_size_search, 2)) #b, num, 1
        q = weight * q

        del weight

        if softmax:
            q = 1 / (torch.sum(q, dim=-1) + 1e-6).unsqueeze(-1) * q
        self.s = q
        del q

    def simself_mask_nonlocalavg_cuda_v4(self, img, mask, kernel_size_search=25, kernel_size_center=[5, 9, 13], sigma=1.0, softmax=False):
        q_list = []
        for k in kernel_size_center:
            q = compute_similarity(image=img[0], mask=mask[0,0], psize=kernel_size_search, ksize=k)
            q = q / (3 * math.pow(k, 2))
            q = q.unsqueeze(0)
            b, num, _, _ = q.shape
            q = q.reshape(b, num, kernel_size_search * kernel_size_search)

            q = torch.exp(-1 * q / sigma)

            if softmax:
                q = 1 / (torch.sum(q, dim=-1) + 1e-10).unsqueeze(-1) * q
            q_list.append(q)
            del q

        q = torch.stack(q_list, dim = 3)
        del q_list
        q = torch.max(q, dim = 3)[0]
        self.s = q
        del q

    def simself_mask_nonlocalavg_cuda_v1RGB(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False):
        b,c,h,w = img.shape
        q_list = []
        for i in range(c):
            q = compute_similarity(image=img[0, i, :, :].unsqueeze(0), mask=mask[0, i, :, :], psize=kernel_size_search, ksize=kernel_size_center)
            q = q / math.pow(kernel_size_center, 2)
            q = q.unsqueeze(0)
            b, num, _, _ = q.shape
            q = q.reshape(b, num, kernel_size_search * kernel_size_search)

            q = torch.exp(-1 * q / sigma)

            if softmax:
                q = 1 / (torch.sum(q, dim=-1) + 1e-10).unsqueeze(-1) * q
            q_list.append(q)
            del q
        self.s = torch.cat(q_list, dim=1)
        del q_list

    def simself_mask_nonlocalavg_cuda_v5(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False, gene_type = "sum", largest_k = 0): # v27
        b,c,h,w = img.shape
        q = compute_similarity(image=img[0], mask=mask[0,0], psize=kernel_size_search, ksize=kernel_size_center)
        q = q / (c * math.pow(kernel_size_center, 2))
        q = q.unsqueeze(0)
        b, num, _, _ = q.shape
        q = q.reshape(b, num, kernel_size_search * kernel_size_search)

        q = torch.exp(-1 * q / sigma)

        if softmax:
            if gene_type == 'sum':
                q = 1 / (torch.sum(q, dim=-1) + 1e-10).unsqueeze(-1) * q
            elif gene_type == "softmax":
                q = q.softmax(dim = -1)

        if largest_k > 0:
            q, _ = torch.sort(q, dim = -1, descending=True)
            q = q[ : , : , : largest_k]

        self.s = q
        del q

    def simself_mask_nonlocalavg_cuda_maxh_v1(self, img_gt, img_sr, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False):
        b,c,h,w = img_gt.shape
        q_gt = compute_similarity(image=img_gt[0], mask=mask[0,0], psize=kernel_size_search, ksize=kernel_size_center)
        q_gt = q_gt / (c * math.pow(kernel_size_center, 2))
        q_gt = q_gt.unsqueeze(0)
        b, num, _, _ = q_gt.shape
        q_gt = q_gt.reshape(b, num, kernel_size_search * kernel_size_search)

        q_sr = compute_similarity(image=img_sr[0], mask=mask[0,0], psize=kernel_size_search, ksize=kernel_size_center)
        q_sr = q_sr / (c * math.pow(kernel_size_center, 2))
        q_sr = q_sr.unsqueeze(0)
        b, num, _, _ = q_sr.shape
        q_sr = q_sr.reshape(b, num, kernel_size_search * kernel_size_search)

        max_h = (q_gt-q_sr + 1e-20) / (((q_gt.pow(2) + 1e-20)/(q_sr.pow(2) + 1e-20)).log() + 1e-20)

        q_gt = torch.exp(-1 * q_gt / max_h)

        q_sr = torch.exp(-1 * q_sr / max_h)

        if softmax:
            q_gt = 1 / (torch.sum(q_gt, dim=-1) + 1e-20).unsqueeze(-1) * q_gt
            q_sr = 1 / (torch.sum(q_sr, dim=-1) + 1e-20).unsqueeze(-1) * q_sr
        self.s = q_gt
        self.s1 = q_sr

    # def simself_mask_nonlocalavg_cuda_v1_p(self, img, mask, kernel_size_search=25, kernel_size_center=9, sigma=1.0, softmax=False):
    #     b,c,h,w = img.shape
    #     q = compute_similarity_p(image=img[0], mask=mask[0,0], psize=kernel_size_search, ksize=kernel_size_center)
    #     q = q / (c * math.pow(kernel_size_center, 2))
    #     q = q.unsqueeze(0)
    #     b, num, _, _ = q.shape
    #     q = q.reshape(b, num, kernel_size_search * kernel_size_search)
    #
    #     q = torch.exp(-1 * q / sigma)
    #
    #     if softmax:
    #         q = 1 / torch.sum(q, dim=-1).unsqueeze(-1) * q
    #     self.s = q
    #     del q


    def get_gradient(self, x):
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        tensor_kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        tensor_kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        weight_h = nn.Parameter(data=tensor_kernel_h, requires_grad=False).cuda()
        weight_v = nn.Parameter(data=tensor_kernel_v, requires_grad=False).cuda()
        x_list = []
        b, c, h, w = x.shape
        for i in range(c):
            x_c = x[:, i, :, :]
            x_grad_v = F.conv2d(x_c.unsqueeze(1), weight_v, padding=1)
            x_grad_h = F.conv2d(x_c.unsqueeze(1), weight_h, padding=1)
            x_grad = torch.sqrt(torch.pow(x_grad_v, 2) + torch.pow(x_grad_h, 2) + 1e-6)
            x_list.append(x_grad)
        grad = torch.cat(x_list, dim=1)
        return grad

    def getitem(self):
        return self.s

    def getitem_gradfilter(self):
        return self.s, self.index

    def getitem_simmutual(self):
        return self.s, self.s1
    
class trainable_similarity_map(nn.Module):
    def __init__(self, scaling_factor=4):
        super(trainable_similarity_map, self).__init__()

        self.sigma = nn.Parameter(torch.tensor([float(scaling_factor)]), requires_grad=True)

    def forward(self, img, img_sr, mask, kernel_size_search=25, kernel_size_center=9, softmax=False):
        b, c, h, w = img.shape
        q = compute_similarity(image=img[0], mask=mask[0, 0], psize=kernel_size_search, ksize=kernel_size_center)
        q = q / (c * math.pow(kernel_size_center, 2))
        q = q.unsqueeze(0)
        b, num, _, _ = q.shape
        q = q.reshape(b, num, kernel_size_search * kernel_size_search)
        q = torch.exp(-1 * q / torch.relu(self.sigma) + 1e-20)
        if softmax:
            q = 1 / (torch.sum(q, dim=-1) + 1e-20).unsqueeze(-1) * q
        s = q
        del q

        q = compute_similarity(image=img_sr[0], mask=mask[0, 0], psize=kernel_size_search, ksize=kernel_size_center)
        q = q / (c * math.pow(kernel_size_center, 2))
        q = q.unsqueeze(0)
        b, num, _, _ = q.shape
        q = q.reshape(b, num, kernel_size_search * kernel_size_search)
        q = torch.exp(-1 * q / torch.relu(self.sigma) + 1e-20)
        if softmax:
            q = 1 / (torch.sum(q, dim=-1) + 1e-20).unsqueeze(-1) * q
        s1 = q
        del q

        return s, s1

    def getitem_h(self):
        return self.sigma.item()

def judge_abnormal_pixel(sr, gt, kernel_size = 3):
    # sr 1,3,h,w
    sr_original = sr
    gt_original = gt
    b,c,h,w = sr.shape
    sr = F.pad(input=sr, pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode="reflect")
    sr = F.unfold(input=sr, padding=0, kernel_size=kernel_size, stride=1) #1, c*k*k, h*w

    sr = sr.reshape(b, c, kernel_size * kernel_size, h*w)

    gt = F.pad(input=gt, pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode="reflect")
    gt = F.unfold(input=gt, padding=0, kernel_size=kernel_size, stride=1) #1, c*k*k, h*w

    gt = gt.reshape(b, c, kernel_size * kernel_size, h*w)

    sr_center = sr[:, :, kernel_size * kernel_size // 2, :] #b,c,1,h*w
    gt_center = gt[:, :, kernel_size * kernel_size // 2, :] #b,c,1,h*w

    sr_neighbour = torch.cat([sr[:, :, : kernel_size * kernel_size // 2, :], sr[:, :, kernel_size * kernel_size // 2 + 1 : , :]], dim = 2)
    gt_neighbour = torch.cat([gt[:, :, : kernel_size * kernel_size // 2, :], gt[:, :, kernel_size * kernel_size // 2 + 1 : , :]], dim = 2)

    diff_center = (sr_center - gt_center).abs().sum(dim=1, keepdim=True) #b, 1, 1, h*w
    diff_neighbour = (sr_neighbour - gt_neighbour).abs().sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) / (kernel_size * kernel_size - 1) #b, 1, 1, h*w

    diff_center = torch.cat([diff_center, diff_center, diff_center], dim = 1) #b, c, 1, h*w
    diff_neighbour = torch.cat([diff_neighbour, diff_neighbour, diff_neighbour], dim=1)

    diff_center = diff_center.squeeze(2).reshape(b, c, h, w)
    diff_neighbour = diff_neighbour.squeeze(2).reshape(b, c, h, w)

    index_abnormal = torch.where(diff_center > 3 * diff_neighbour)
    index_normal = torch.where(diff_center <= 3 * diff_neighbour)

    sr_abnormal = sr_original[index_abnormal]
    gt_abnormal = gt_original[index_abnormal]
    # print(f"index_abnormal length is {len(index_abnormal[0])}")
    # print(f"sr_abnormal shape is {sr_abnormal}")
    return sr_abnormal, gt_abnormal, index_normal, index_abnormal