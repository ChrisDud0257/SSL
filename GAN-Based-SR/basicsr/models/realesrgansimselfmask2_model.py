import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import math
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.losses.loss_util import similarity_map
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, paired_random_crop_img_mask

@MODEL_REGISTRY.register()
class RealESRGANSimSelfMask2Model(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(RealESRGANSimSelfMask2Model, self).__init__(opt)

        self.pre_pad = self.opt['pre_pad']
        self.tile_size = self.opt['tile_size']
        self.tile_pad = self.opt['tile_pad']

        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 160)

        self.use_network_d = False

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.logger = get_root_logger()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
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
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

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

        if self.cri_pix is None and self.cri_perceptual is None:
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

    @torch.no_grad()
    def feed_data(self, data):
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'gt_mask' in data:
            self.gt_mask = data['gt_mask'].to(self.device)
            # print(f"self gt mask shape is {self.gt_mask.shape}")
        if 'kernel1' in data:
            self.kernel1 = data['kernel1'].to(self.device)
        if 'kernel2' in data:
            self.kernel2 = data['kernel2'].to(self.device)
        if 'sinc_kernel' in data:
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

        if self.opt.get('Use_sharpen', None) is not None:
            self.gt_usm = self.usm_sharpener(self.gt)

        ori_h, ori_w = self.gt.size()[2:4]

        if self.opt['degradation_order'] == 'one':
            # ----------------------- The first degradation process ----------------------- #
            # blur
            if self.opt.get('Use_sharpen', None) is not None and self.opt['Sharpen_before_degra']:
                out = filter2D(self.gt_usm, self.kernel1)
            else:
                out = filter2D(self.gt, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        elif self.opt['degradation_order'] == 'two':
            # ----------------------- The first degradation process ----------------------- #
            # blur
            if self.opt.get('Use_sharpen', None) is not None and self.opt['Sharpen_before_degra']:
                out = filter2D(self.gt_usm, self.kernel1)
            else:
                out = filter2D(self.gt, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # random crop
        gt_size = self.opt['datasets']['train']['gt_size']
        if self.opt.get('Use_sharpen', None) is not None:
            (self.gt, self.gt_usm), self.lq, self.gt_mask = paired_random_crop_img_mask(img_gts = [self.gt, self.gt_usm], img_lqs = self.lq,
                                                                                        masks=self.gt_mask,
                                                                                        gt_patch_size = gt_size,
                                                                                        scale = self.opt['scale'])
        else:
            self.gt, self.lq, self.gt_mask = paired_random_crop_img_mask(img_gts = self.gt, img_lqs = self.lq,
                                                                                        masks=self.gt_mask,
                                                                                        gt_patch_size = gt_size,
                                                                                        scale = self.opt['scale'])
        # training pair pool
        self._dequeue_and_enqueue()
        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        if self.opt.get('Use_sharpen', None) is not None:
            self.gt_usm = self.usm_sharpener(self.gt)
        self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract


    @torch.no_grad()
    def feed_val_data(self, data):
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_gt_mask = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            self.queue_gt_mask = self.queue_gt_mask[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            gt_mask_dequeue = self.queue_gt_mask[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()
            self.queue_gt_mask[0:b, :, :, :] = self.gt_mask.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
            self.gt_mask = gt_mask_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_gt_mask[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt_mask.clone()
            self.queue_ptr = self.queue_ptr + b

    def train_net_g(self):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        # self.output_ema = self.net_g_ema(self.lq)
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.l1_gt)
            self.l_g_total += l_pix
            self.loss_dict['l_pix'] = l_pix
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

        if self.opt.get('Use_sharpen', None) is not None:
            self.l1_gt = self.gt_usm
        if self.opt.get('Use_sharpen', None) is not None:
            self.ssl_gt = self.gt_usm
        if self.opt.get('Use_sharpen', None) is not None:
            self.percep_gt = self.gt_usm
        if self.opt.get('Use_sharpen', None) is not None:
            self.gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            self.l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            self.percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            self.gan_gt = self.gt
        if self.opt['ssl_gt_usm'] is False:
            self.ssl_gt = self.gt

        if not self.use_network_d:
            self.train_net_g()
        else:
            for p in self.net_d.parameters():
                p.requires_grad = False
            if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
                self.train_net_g()

                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, self.percep_gt)
                    if l_g_percep is not None:
                        self.l_g_total += l_g_percep
                        self.loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        self.l_g_total += l_g_style
                        self.loss_dict['l_g_style'] = l_g_style

                # gan loss (relativistic gan)
                if self.cri_gan:
                    if self.opt['train']['gan_loss_compute'] == 'RaGAN':
                        real_d_pred = self.net_d(self.gan_gt).detach()
                        fake_g_pred = self.net_d(self.output)
                        l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
                        l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
                        l_g_gan = (l_g_real + l_g_fake) / 2
                    elif self.opt['train']['gan_loss_compute'] == 'GAN':
                        fake_g_pred = self.net_d(self.output)
                        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
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
            if self.opt['train']['gan_loss_compute'] == 'RaGAN':
                # real
                fake_d_pred = self.net_d(self.output).detach()
                real_d_pred = self.net_d(self.gan_gt)
                l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
                l_d_real.backward()
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
                l_d_fake.backward()
                self.optimizer_d.step()

            elif self.opt['train']['gan_loss_compute'] == 'GAN':
                real_d_pred = self.net_d(self.gan_gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                l_d_real.backward()
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                l_d_fake.backward()
                self.optimizer_d.step()

            self.loss_dict['l_d_real'] = l_d_real
            self.loss_dict['l_d_fake'] = l_d_fake
            self.loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            self.loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(self.loss_dict)

        del self.output
        del self.l1_gt
        del self.percep_gt
        del self.gan_gt

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
            self.feed_val_data(val_data)
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