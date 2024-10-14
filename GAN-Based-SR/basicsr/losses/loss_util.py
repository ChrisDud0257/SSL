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



class similarity_map():
    def __init__(self, img, mask=None, ssl_mode='cuda', kernel_size_search=5, generalization=True, kernel_size_window=9, sigma = 0.004):
        super(similarity_map, self).__init__()

        if ssl_mode == 'pytorch':
            self.ssl_pytorch(img=img, mask=mask, kernel_size_search=kernel_size_search,
                             kernel_size_window=kernel_size_window, sigma=sigma,
                             generalization=generalization)

        elif ssl_mode == 'cuda':
            self.ssl_cuda(img=img, mask=mask, kernel_size_search=kernel_size_search,
                                                  kernel_size_window=kernel_size_window,
                                                  sigma=sigma, generalization=generalization)
        else:
            raise ValueError(f"The ssl_mode should either be cuda or pytorch.")


    def ssl_pytorch(self, img, mask, kernel_size_search=25, kernel_size_window=9, sigma=1.0, generalization=False):
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

        img_search_area = F.unfold(input=img_search_area, kernel_size=kernel_size_window,
                                   padding=kernel_size_window // 2, stride=1)  # 1, 3*num*k_c*k_c, k_s*k_s
        img_search_area = img_search_area.reshape(b, c, num, kernel_size_window * kernel_size_window,
                                                  kernel_size_search * kernel_size_search)
        img_search_area = img_search_area.permute(0, 2, 1, 3, 4)  # 1, num, 3, k_c*k_c, k_s*k_s
        img_search_area = img_search_area.reshape(b, num, c * kernel_size_window * kernel_size_window,
                                                  kernel_size_search * kernel_size_search)  # 1, num, c*k_c*k_c, k_s*k_s

        img_center_neighbor = img_search_area[:, :, :, (kernel_size_search * kernel_size_search) // 2].unsqueeze(
            -1)  # 1, num, c*k_c*k_c, 1

        q = img_search_area - img_center_neighbor  # 1, num, c*k_c*k_c, k_s*k_s
        # print(f"q shape is {q.shape}")
        del img_search_area
        del img_center_neighbor
        q = q.pow(2).sum(2)  # 1, num, k_s*k_s
        q = q / (c * math.pow(kernel_size_window, 2))
        q = torch.exp(-1 * q / sigma)
        if generalization:
            q = 1 / (torch.sum(q, dim=-1) + 1e-10).unsqueeze(-1) * q
        self.s = q
        del q

    def ssl_cuda(self, img, mask, kernel_size_search=25, kernel_size_window=9, sigma=1.0, generalization=False):
        b,c,h,w = img.shape
        q = compute_similarity(image=img[0], mask=mask[0,0], psize=kernel_size_search, ksize=kernel_size_window)
        q = q / (c * math.pow(kernel_size_window, 2))
        q = q.unsqueeze(0)
        b, num, _, _ = q.shape
        q = q.reshape(b, num, kernel_size_search * kernel_size_search)

        q = torch.exp(-1 * q / sigma)

        if generalization:
            q = 1 / (torch.sum(q, dim=-1) + 1e-10).unsqueeze(-1) * q
        self.s = q
        del q


    def getitem(self):
        return self.s

