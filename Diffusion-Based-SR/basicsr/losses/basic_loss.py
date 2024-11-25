import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
import clip


_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
    def forward(self, pred, target, **kwargs):
        return self.loss_weight * self.loss(pred, target)

@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

@LOSS_REGISTRY.register()
class PerceptualSimLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion_perceptual_style='l1',
                 simself_weight = 0,
                 simself_layer_weights = (0, 0, 1, 1, 1),
                 criterion_simself = 'l1',
                 feat_simself_dh_list=(0, 0, 16, 16, 0),
                 feat_simself_dw_list=(0, 0, 16, 16, 0),
                 feat_kernel_size_list=(0, 0, 0, 0, 0),
                 cos_distance = False,
                 temperature = 0,
                 softmax_sr = True,
                 softmax_gt = True,
                 rearrange_back = True,
                 crossentropy = False,
                 simself_channel_weight = 0.,
                 simself_channel_layer_wights = (0,0,1,1,1),
                 criterion_simself_channel = 'l1',
                 feat_simself_dc_list = (0,0,16,16,16),
                 feat_channel_kernel_size_list = (0,0,0,0,0)):
        super(PerceptualSimLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.simself_weight = simself_weight
        self.simself_layer_weights = simself_layer_weights
        self.feat_simself_dh_list = feat_simself_dh_list
        self.feat_simself_dw_list = feat_simself_dw_list
        self.feat_kernel_size_list = feat_kernel_size_list
        self.cos_distance = cos_distance
        self.temperature = temperature
        self.softmax_sr = softmax_sr
        self.softmax_gt = softmax_gt

        self.rearrange_back = rearrange_back
        self.crossentropy = crossentropy

        self.simself_channel_weight = simself_channel_weight
        self.simself_channel_layer_wights = simself_channel_layer_wights
        self.feat_simself_dc_list = feat_simself_dc_list
        self.feat_channel_kernel_size_list = feat_channel_kernel_size_list

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_perceptual_style_type = criterion_perceptual_style
        if self.criterion_perceptual_style_type == 'l1':
            self.criterion_perceptual_style = torch.nn.L1Loss()
        elif self.criterion_perceptual_style_type == 'l2':
            self.criterion_perceptual_style = torch.nn.L2loss()
        elif self.criterion_perceptual_style_type == 'fro':
            self.criterion_perceptual_style = None
        else:
            raise NotImplementedError(f'{criterion_perceptual_style} criterion has not been supported.')

        self.criterion_simself_type = criterion_simself
        if self.criterion_simself_type == 'l1':
            self.criterion_simself = torch.nn.L1Loss(reduction='mean')
        elif self.criterion_simself_type == 'crossentropy':
            self.criterion_simself = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            raise NotImplementedError(f'{criterion_simself} criterion has not been supported.')

        self.criterion_simself_channel_type = criterion_simself_channel
        if self.criterion_simself_channel_type == 'l1':
            self.criterion_simself_channel = torch.nn.L1Loss(reduction='mean')
        elif self.criterion_simself_channel_type == 'crossentropy':
            self.criterion_simself_channel = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            raise NotImplementedError(f'{criterion_simself_channel} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                # print(f"perceptual size for {k} is {x_features[k].shape}")
                if self.criterion_perceptual_style_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion_perceptual_style(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion_perceptual_style(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        # calculate simself loss
        if self.simself_weight > 0:
            simself_loss = 0
            for idx, k in enumerate(x_features.keys()):
                # print(f"{x_features[k].shape}")
                # print(f"idx is {idx}")
                if self.simself_layer_weights[idx] > 0:
                    # print(f"idx-{idx+1} layer will be computed")
                    # if self.split_size != 0:
                    #     x_in = self.split_sum(x_features[k], split_size=self.split_size)
                    #     gt_in = self.split_sum(gt_features[k], split_size=self.split_size)
                    # else:
                    #     x_in = x_features[k]
                    #     gt_in = gt_features[k]
                    #     print(f"shape for x_features{idx+1} is {x_features[k].shape}")
                    b,c,h,w = x_features[k].shape
                    x_self_matrix = self.simself_areaarea(img = x_features[k], is_shift = False, shift_h = 4, shift_w = 4,
                                                        dh = self.feat_simself_dh_list[idx], dw = self.feat_simself_dw_list[idx],
                                                        kernel_size =self.feat_kernel_size_list[idx], softmax = self.softmax_sr,
                                                        rearrange_back = self.rearrange_back, crossentropy = self.crossentropy, temperature = self.temperature, cos_distance=self.cos_distance)
                    gt_self_matrix = self.simself_areaarea(img = gt_features[k], is_shift = False, shift_h = 4, shift_w = 4,
                                                         dh = self.feat_simself_dh_list[idx], dw = self.feat_simself_dw_list[idx],
                                                         kernel_size = self.feat_kernel_size_list[idx], softmax = self.softmax_gt,
                                                         rearrange_back = self.rearrange_back, crossentropy = self.crossentropy, temperature = self.temperature, cos_distance=self.cos_distance)
                    loss = self.criterion_simself(x_self_matrix, gt_self_matrix) * self.simself_layer_weights[idx]
                    # print(f"loss is {loss}")
                    simself_loss += loss
                    # print(f"x shape is {x_self_matrix.shape}, gt shape is {gt_self_matrix.shape}")
                    # x_self_matrix = 0
                    # gt_self_matrix = 0
            simself_loss *= self.simself_weight
        else:
            simself_loss = None

        if self.simself_channel_weight > 0:
            simself_channel_loss = 0
            for idx, k in enumerate(x_features.keys()):
                # print(f"{x_features[k].shape}")
                # print(f"idx is {idx}")
                if self.simself_channel_layer_wights[idx] > 0:
                    # print(f"idx-{idx+1} layer will be computed")
                    # if self.split_size != 0:
                    #     x_in = self.split_sum(x_features[k], split_size=self.split_size)
                    #     gt_in = self.split_sum(gt_features[k], split_size=self.split_size)
                    # else:
                    #     x_in = x_features[k]
                    #     gt_in = gt_features[k]
                    #     print(f"shape for x_features{idx+1} is {x_features[k].shape}")
                    x_self_matrix = self.simself_channelchannel(img = x_features[k], is_shift = False, shift_c = 4, dc = self.feat_simself_dc_list[idx],
                                                                kernel_size = self.feat_channel_kernel_size_list[idx],
                                                                softmax = self.softmax_sr, crossentropy = self.crossentropy,
                                                                temperature = self.temperature, cos_distance = self.cos_distance)
                    gt_self_matrix = self.simself_channelchannel(img = gt_features[k], is_shift = False, shift_c = 4, dc = self.feat_simself_dc_list[idx],
                                                                kernel_size = self.feat_channel_kernel_size_list[idx],
                                                                softmax = self.softmax_gt, crossentropy = self.crossentropy,
                                                                temperature = self.temperature, cos_distance = self.cos_distance)
                    loss = self.criterion_simself_channel(x_self_matrix, gt_self_matrix) * self.simself_channel_layer_wights[idx]
                    # print(f"loss is {loss}")
                    simself_channel_loss += loss
            simself_channel_loss *= self.simself_channel_weight
        else:
            simself_channel_loss = None


        return percep_loss, style_loss, simself_loss, simself_channel_loss

    def split_sum(self, x, split_size = 4):
        x = torch.split(x, split_size_or_sections=split_size, dim=1)
        x = [torch.sum(i, dim=1).unsqueeze(1) for i in x]
        x = torch.cat(x, dim=1)

        return x

    def simself_areaarea(self, img, is_shift = False, shift_h = 4, shift_w = 4, dh = 32, dw = 32, kernel_size = 5, softmax = True,
                         rearrange_back = True, crossentropy = False, temperature = 0, cos_distance = False):
        b, c, h, w = img.shape
        x = img
        if is_shift:
            x = torch.roll(x, shifts=(-1 * shift_h, -1 * shift_w), dims=(2, 3))
        if dh == 0 or dw == 0:
            if kernel_size > 0:
                q = F.unfold(input=x, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)  # b,c*kernel_size*kernel_size, h*w
            else:
                q = x.reshape(b,c,h*w)
            q = q.transpose(-2, -1)  # b, h*w, c*kernel_size*kernel_size / b,h*w,c
            if cos_distance:
                q = q / (torch.norm(input = q, p=2, dim=-1).unsqueeze(-1) + 1e-6)
            s = q @ (q.transpose(-2, -1))  # b, h*w, h*w
            if temperature != 0:
                s = s / temperature
                # if softmax:
                #     print(f"GT-Softmax is {s.softmax(dim = -1)[0][0][0][0][:20]}")
                # else:
                #     print(f"SR-Softmax is {s.softmax(dim=-1)[0][0][0][0][:20]}")
            if softmax:
                s = s.softmax(dim=-1)
            if crossentropy:
                s = s.reshape(b * h * w, h * w)
            else:
                if rearrange_back:
                    s = s.reshape(b, h * w, h, w)
                    if is_shift:
                        s = torch.roll(s, shifts=(shift_h, shift_w), dims=(2, 3))
        else:
            q = rearrange(x, 'b c (h dh) (w dw) -> b c h dh w dw', dh=dh, dw=dw)
            q = q.permute(0, 1, 2, 4, 3, 5)  # b,c,h,w,dh,dw
            b, c, h, w, dh, dw = q.shape
            if kernel_size > 0:
                q = q.reshape(b, c * h * w, dh, dw)
                q = F.unfold(input=q, kernel_size=kernel_size, padding=kernel_size//2, stride=1) #b,c*h*w*kernel_size*kernel_size, dh*dw
                q = q.reshape(b, c, h*w, kernel_size*kernel_size, dh * dw)
                q = q.permute(0,2,4,1,3) #b,h*w, dh*dw, c, kernel_size*kernel_size
                q = q.reshape(b, h, w, dh*dw, c*kernel_size*kernel_size)
            else:
                q = q.reshape(b,c,h*w,dh*dw)
                q = q.permute(0,2,3,1) #b,h*w,dh*dw,c
            if cos_distance:
                q = q / (torch.norm(input = q, p=2, dim=-1).unsqueeze(-1) + 1e-6)
            s = q @ (q.transpose(-2,-1))  #b, h, w, dh*dw, dh*dw
            if temperature != 0:
                s = s / temperature
                # if softmax:
                #     print(f"GT-Softmax is {s.softmax(dim = -1)[0][0][0][0][:20]}")
                # else:
                #     print(f"SR-Softmax is {s.softmax(dim=-1)[0][0][0][0][:20]}")
            if softmax:
                s = s.softmax(dim=-1)
            if crossentropy:
                s = s.reshape(b*h*w*dh*dw, dh*dw)
            else:
                if rearrange_back:
                    s = rearrange(s, 'b h w (dh dw) C -> b C (h dh) (w dw)', dh=dh, dw=dw) #b,dh*dw, h*dh, w*dw
                    if is_shift:
                        s = torch.roll(s, shifts=(shift_h, shift_w), dims=(2, 3))
        return s

    def simself_channelchannel(self, img, is_shift = False, shift_c = 4, dc = 32, kernel_size = 5, softmax = True, crossentropy = False,
                               temperature = 0, cos_distance = False):
        b, c, h, w = img.shape
        x = img.clone()
        if is_shift:
            x = torch.roll(x, shifts=(-1 * shift_c), dims=(1))
        if dc == 0:
            if kernel_size > 0:
                q = x.permute(0,2,3,1) #b,h,w,c
                q = F.pad(input=q, pad=(kernel_size//2, kernel_size//2), mode='reflect')
                q = q.unfold(dimension=-1, step=1, size=kernel_size)  # b,h,w,c,kernel_size
                q = q.permute(0,3,1,2,4) #b,c,h,w,kernel_size
                q = q.reshape(b,c,h*w*kernel_size)
            else:
                q = x.reshape(b, c, h*w)
            if cos_distance:
                q = q / (torch.norm(input = q, p=2, dim=-1).unsqueeze(-1) + 1e-6)
            s = q @ q.transpose(-2, -1) #b,c,c
            if temperature != 0:
                s = s / temperature
            if softmax:
                s = s.softmax(dim=-1)
            if crossentropy:
                s = s.reshape(b*c, c)
        else:
            q = rearrange(x, 'b (c dc) h w -> b c dc h w', dc= dc) # b,c,dc,h,w
            b, c, dc, h, w = q.shape
            q = q.reshape(b, c, dc, h*w)
            q = q.permute(0,1,3,2) #b,c,h*w,dc
            if kernel_size > 0:
                q = F.pad(input=q, pad=(kernel_size//2, kernel_size//2), mode='reflect')
                q = q.unfold(dimension=-1, step=1, size=kernel_size) #b,c,h*w,dc,kernel_size
                q = q.permute(0,1,3,2,4) #b,c,dc,h*w,kernel_size
                q = q.reshape(b,c,dc, h*w*kernel_size)
            else:
                q = q.transpose(-2,-1) #b,c,dc,h*w
            if cos_distance:
                q = q / (torch.norm(input = q, p=2, dim=-1).unsqueeze(-1) + 1e-6)
            s = q @ q.transpose(-2,-1) #b,c,dc,dc
            if temperature != 0:
                s = s / temperature
            if softmax:
                s = s.softmax(dim=-1)
            if crossentropy:
                s = s.reshape(b*c*dc, dc)
        return s


    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

@LOSS_REGISTRY.register()
class CosineDistanceLoss(nn.Module):
    def __init__(self, loss_weight = 0.1):
        super(CosineDistanceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x , y):
        cosdistance = (1-F.cosine_similarity(x, y, dim=-1)).mean() * self.loss_weight
        return cosdistance

@LOSS_REGISTRY.register()
class BCELoss(nn.Module):
    def __init__(self, loss_weight = 0.1, reduction = 'mean'):
        super(BCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, x, y):
        return self.loss_weight * self.bce(x,y)

@LOSS_REGISTRY.register()
class KLDistanceLoss(nn.Module):
    def __init__(self, loss_weight = 0.1, reduction = 'mean', softmax = False):
        super(KLDistanceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.softmax = softmax

    def forward(self, x, y):
        if self.softmax:
            x=x.softmax(dim= -1)
            y=y.softmax(dim= -1)
        kldistance = self.loss_weight * F.kl_div((torch.clamp(input=x, min=1e-10)).log(), torch.clamp(input=y, min=1e-10), reduction = self.reduction)
        return kldistance

@LOSS_REGISTRY.register()
class KLDistanceLoss1(nn.Module):
    def __init__(self, loss_weight = 0.1, reduction = 'mean', softmax = False):
        super(KLDistanceLoss1, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.softmax = softmax

    def forward(self, x, y):
        if self.softmax:
            x=x.softmax(dim= -1)
            y=y.softmax(dim= -1)
        kldistance = self.loss_weight * F.kl_div((torch.clamp(input=x, min=1e-25)).log(), (torch.clamp(input=y, min=1e-25)).log(), reduction = self.reduction, log_target=True)
        return kldistance

@LOSS_REGISTRY.register()
class MaxDistanceLoss(nn.Module):
    def __init__(self, loss_weight = 0.1, reduction = 'mean'):
        super(MaxDistanceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, x, y):
        if self.reduction == 'mean':
            b,c,h,w = x.shape
            maxdistance = self.loss_weight * torch.max(torch.abs(x - y)) / (b*c*h*w)
        else:
            maxdistance = self.loss_weight * torch.max(torch.abs(x - y))
        return maxdistance

@LOSS_REGISTRY.register()
class SmoothL2Loss(nn.Module):
    def __init__(self, delta = 0.1, loss_weight = 1.0, reduction = 'mean'):
        super(SmoothL2Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.delta = delta

    def forward(self, x, y):
        smoothl2loss = torch.where(torch.abs(x-y) > self.delta, 0.5*((x-y)**2), torch.abs(self.delta * torch.abs(x-y) - 0.5*(self.delta**2)))
        b,c,h,w = x.shape
        if self.reduction == 'mean':
            loss = smoothl2loss.sum()/(b*c*h*w)
        elif self.reduction == 'sum':
            loss = smoothl2loss.sum()
        return loss

@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, loss_weight = 1.0):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        loss = 1 - ssim(x, y, data_range=1, size_average=True)
        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class ClipLoss(nn.Module):
    def __init__(self, pretrain_clipmodel_path,
                 perceptual_weight = 0.1, style_weight = 0.,
                 criterion = 'l1',
                 input_norm = True,
                 mean = (0.48145466, 0.4578275, 0.40821073),
                 std = (0.26862954, 0.26130258, 0.27577711),
                 layer_weights = (0.1, 0.1, 1, 1, 1),
                 feature_choose = (0,2,5,8,11),
                 self_similarity = 'no'):
        super(ClipLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.criterion_type = criterion
        self.input_norm = input_norm
        self.layer_weights = layer_weights
        self.feature_choose = feature_choose
        self.self_similarity = self_similarity

        self.model, _ = clip.load(str(pretrain_clipmodel_path), device='cuda')

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

        if self.input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor(mean).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor(std).view(1, 3, 1, 1))

    def forward(self, x, gt):
        # b,c,h,w = x.shape
        b, c, h, w = x.shape
        if h != 224 or w != 224:
            x = F.interpolate(x, size=(224,224), mode='bicubic')
            gt = F.interpolate(gt, size=(224,224), mode='bicubic')
        if self.input_norm:
            x = (x - self.mean) / self.std
            gt = (gt - self.mean) / self.std
        with torch.no_grad():
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x_final = self.model.encode_image(x)
            # print(f"x11 shape is {x11.shape}")
            gt0, gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8, gt9, gt10 ,gt11, gt_final = self.model.encode_image(gt.detach())
            x_features = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x_final]
            gt_features = [gt0, gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8, gt9, gt10 ,gt11, gt_final]

            x_choosed_features = [x_features[i] for i in self.feature_choose]
            gt_choosed_features = [gt_features[i] for i in self.feature_choose]

            if self.perceptual_weight > 0:
                percep_loss = 0
                for i in range(len(x_choosed_features)):
                    if self.criterion_type == "fro":
                        percep_loss += torch.norm(x_choosed_features[i] - gt_choosed_features[i], p='fro') * self.layer_weights[i]
                    else:
                        if self.self_similarity == 'no':
                            percep_loss += self.criterion(x_choosed_features[i], gt_choosed_features[i]) * self.layer_weights[i]
                        elif self.self_similarity == 'similarity_featfeat_nopatch':
                            percep_loss += self.criterion(self.similarity_featfeat_nopatch(x_choosed_features[i]),
                                                          self.similarity_featfeat_nopatch(gt_choosed_features[i])) * self.layer_weights[i]
                        elif self.self_similarity == 'similarity_featfeat_nopatch_final':
                            if x_choosed_features[i].ndim !=2 :
                                percep_loss += self.criterion(self.similarity_featfeat_nopatch(x_choosed_features[i]),
                                                              self.similarity_featfeat_nopatch(gt_choosed_features[i])) * self.layer_weights[i]
                            else:
                                percep_loss += self.criterion(self.similarity_final_feat(x_choosed_features[i]),
                                                              self.similarity_final_feat(gt_choosed_features[i])) * self.layer_weights[i]
                percep_loss *= self.perceptual_weight
            else:
                percep_loss = None

            # calculate style loss
            if self.style_weight > 0:
                style_loss = 0
                assert x.ndim == 4, f"The dimension of x is {x.ndim}, but not 4."
                for i in range(len(x_choosed_features)):
                    if self.criterion_type == 'fro':
                        style_loss += torch.norm(self._gram_mat(x_choosed_features[i]) - self._gram_mat(gt_choosed_features[i]), p='fro') * self.layer_weights[i]
                    else:
                        style_loss += self.criterion(self._gram_mat(x_choosed_features[i]), self._gram_mat(gt_choosed_features[i])) * self.layer_weights[i]
                style_loss *= self.style_weight
            else:
                style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def similarity_final_feat(self, feature):
        b, k = feature.shape
        x = feature
        x = x.unsqueeze(1) #b,1,k
        s = x @ (x.transpose(-2,-1)) #b,1
        return s


    def similarity_featfeat_nopatch(self, feature):
        k, b, c = feature.shape
        x = feature
        x = x.permute(1, 0, 2)  # b,k,c
        s = x @ (x.transpose(-2, -1))
        s = s.reshape(b, k*k)
        s = s.softmax(dim=-1)
        s = s.reshape(b, k, k)

        return s