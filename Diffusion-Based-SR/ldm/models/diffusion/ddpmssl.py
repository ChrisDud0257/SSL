import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian
import random
import torch.nn.functional as F
import math

from ldm.modules.diffusionmodules.util import make_ddim_timesteps
import copy
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ldm.models.diffusion.ddpm import LatentDiffusionSRTextWT
from basicsr.losses import build_loss
from basicsr.losses.loss_util import self_similarity, gradient_img_similarity, similarity_map

class LatentDiffusionSRTextWTSSL(LatentDiffusionSRTextWT):
    """main class"""
    def __init__(self, sslopt = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sslopt = sslopt

        print(f"sslopt is {self.sslopt}")

    def init_issl_settings(self):
        if self.sslopt.get('mask_stride', 0) > 1:
            mask_size = int(self.image_size)
            mask_stride = torch.eye(self.sslopt.get('mask_stride', 0), self.sslopt.get('mask_stride', 0),
                                    dtype=torch.float32)
            mask_stride = mask_stride.repeat(math.ceil(mask_size / self.sslopt.get('mask_stride', 0)),
                                             math.ceil(mask_size / self.sslopt.get('mask_stride', 0)))
            mask_stride = mask_stride[:mask_size, :mask_size]
            mask_stride = mask_stride.unsqueeze(0).unsqueeze(0)
            self.register_buffer('mask_stride', mask_stride)
            print(f"mask stride is {self.sslopt.get('mask_stride', 0)}")

        if self.configs.ISSL_loss.get('selfsim_opt'):
            self.cri_selfsim = build_loss(self.configs.ISSL_loss['selfsim_opt']).to(self.device)
        else:
            self.cri_selfsim = None

        if self.configs.ISSL_loss.get('selfsim1_opt'):
            self.cri_selfsim1 = build_loss(self.configs.ISSL_loss['selfsim1_opt']).to(self.device)
        else:
            self.cri_selfsim1 = None

    def instantiate_first_stage(self, config):
        self.first_stage_model = instantiate_from_config(config)

    def shared_step(self, batch, **kwargs):
        x_z, c, gt_z, mask, x, gt, sr = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True)
        loss = self(x_z, c, gt_z, mask, x, gt, sr)
        return loss

    @torch.no_grad()
    def get_input(self, batch, k=None, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, val=False, text_cond=[''], return_gt=False,
                  resize_lq=True):

        """Degradation pipeline, modified from Real-ESRGAN:
        https://github.com/xinntao/Real-ESRGAN
        """

        if not hasattr(self, 'jpeger'):
            jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        if not hasattr(self, 'usm_sharpener'):
            usm_sharpener = USMSharp().cuda()  # do usm sharpening

        # print(f'batch is {batch}')

        im_gt = batch['gt'].cuda()
        if self.use_usm:
            im_gt = usm_sharpener(im_gt)
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        kernel1 = batch['kernel1'].cuda()
        kernel2 = batch['kernel2'].cuda()
        sinc_kernel = batch['sinc_kernel'].cuda()
        gt_mask = batch['gt_mask'].cuda()

        ori_h, ori_w = im_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
            ['up', 'down', 'keep'],
            self.configs.degradation['resize_prob'],
        )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.configs.degradation['gray_noise_prob']
        if random.random() < self.configs.degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < self.configs.degradation['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
            ['up', 'down', 'keep'],
            self.configs.degradation['resize_prob2'],
        )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out,
            size=(int(ori_h / self.configs.sf * scale),
                  int(ori_w / self.configs.sf * scale)),
            mode=mode,
        )
        # add noise
        gray_noise_prob = self.configs.degradation['gray_noise_prob2']
        if random.random() < self.configs.degradation['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
            )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // self.configs.sf,
                      ori_w // self.configs.sf),
                mode=mode,
            )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // self.configs.sf,
                      ori_w // self.configs.sf),
                mode=mode,
            )
            out = filter2D(out, sinc_kernel)

        # clamp and round
        im_lq = torch.clamp(out, 0, 1.0)

        # random crop
        # gt_size = self.configs.degradation['gt_size']
        # im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, self.configs.sf)
        self.lq, self.gt = im_lq, im_gt
        self.gt_mask = gt_mask

        del im_gt, im_lq, gt_mask

        if resize_lq:
            self.lq = F.interpolate(
                self.lq,
                size=(self.gt.size(-2),
                      self.gt.size(-1)),
                mode='bicubic',
            )

        if random.random() < self.configs.degradation['no_degradation_prob'] or torch.isnan(self.lq).any():
            self.lq = self.gt

        # training pair pool
        if not val and not self.random_size:
            self._dequeue_and_enqueue()
        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        self.lq = self.lq * 2 - 1.0
        self.gt = self.gt * 2 - 1.0

        if self.random_size:
            self.lq, self.gt = self.randn_cropinput(self.lq, self.gt)

        self.lq = torch.clamp(self.lq, -1.0, 1.0)

        x = self.lq
        y = self.gt
        gt_m = self.gt_mask
        del self.lq, self.gt, self.gt_mask
        if bs is not None:
            x = x[:bs]
            y = y[:bs]
        x = x.to(self.device)
        y = y.to(self.device)
        gt_m = gt_m.to(self.device)
        with torch.no_grad():
            encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        encoder_posterior_y = self.encode_first_stage(y)
        z_gt = self.get_first_stage_encoding(encoder_posterior_y).detach()

        xc = None
        if self.use_positional_encodings:
            assert NotImplementedError
            pos_x, pos_y = self.compute_latent_shifts(batch)
            c = {'pos_x': pos_x, 'pos_y': pos_y}

        while len(text_cond) < z.size(0):
            text_cond.append(text_cond[-1])
        if len(text_cond) > z.size(0):
            text_cond = text_cond[:z.size(0)]
        assert len(text_cond) == z.size(0)

        out = [z, text_cond]
        out.append(z_gt)
        out.append(gt_m)

        if return_first_stage_outputs:
            xrec = self.differentiable_decode_first_stage(z)
            xrec = torch.clamp((xrec + 1.0) / 2.0, min=0.0, max=1.0)
            y = torch.clamp((y + 1.0) / 2.0, min=0.0, max=1.0)
            out.extend([x, y, xrec])
        if return_original_cond:
            out.append(xc)

        return out

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.data.params.train.params.get('queue_size', b * 16)
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

    def forward(self, x, c, gt, mask, lq_img, gt_img, sr_img, *args, **kwargs):
        index = np.random.randint(0, self.num_timesteps, size=x.size(0))
        t = torch.from_numpy(index)
        t = t.to(self.device).long()

        t_ori = torch.tensor([self.ori_timesteps[index_i] for index_i in index])
        t_ori = t_ori.long().to(x.device)

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            else:
                c = self.cond_stage_model(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                # print(s)
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        if self.test_gt:
            struc_c = self.structcond_stage_model(gt, t_ori)
        else:
            struc_c = self.structcond_stage_model(x, t_ori)
        return self.p_losses(gt, c, struc_c, t, t_ori, x, mask, lq_img, gt_img, sr_img, *args, **kwargs)

    def p_losses(self, x_start, cond, struct_cond, t, t_ori, lq, mask, lq_img, gt_img, sr_img, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.mix_ratio > 0:
            if random.random() < self.mix_ratio:
                noise_new = default(noise, lambda: torch.randn_like(x_start))
                noise = noise_new * 0.5 + noise * 0.5
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output = self.apply_model(x_noisy, t_ori, cond, struct_cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        model_output_ = model_output

        loss_simple = self.get_loss(model_output_, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        #P2 weighting
        if self.snr is not None:
            self.snr = self.snr.to(loss_simple.device)
            weight = extract_into_tensor(1 / (self.p2_k + self.snr)**self.p2_gamma, t, target.shape)
            loss_simple = weight * loss_simple

        logvar_t = self.logvar[t.cpu()].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output_, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        # === add loss in the pixel space ===
        # --- obtain the x0 ---
        x0 = self.predict_start_from_noise(x_t=x_noisy, t=t, noise=model_output_)

        # --- obtain the image ---
        image = self.differentiable_decode_first_stage(x0)


        # --- compute the loss ---
        l1_loss = torch.mean(torch.abs(image - gt_img))
        loss = loss + 0.1*l1_loss
        loss_dict.update({f'{prefix}/loss_pixel': l1_loss})

        loss_selfsim, loss_selfsim_kl = self.issl(sr=image, gt=gt_img, mask=mask, sslopt=self.sslopt)
        loss += loss_selfsim
        loss += loss_selfsim_kl

        loss_dict.update({f'{prefix}/loss_selfsim': loss_selfsim})
        loss_dict.update({f'{prefix}/loss_selfsim_kl': loss_selfsim_kl})
        # === ends ===

        return loss, loss_dict

    def issl(self, sr, gt, mask, sslopt):
        if self.cri_selfsim or self.cri_selfsim1:
            b, _, _, _ = gt.shape
            b_gt_list = []
            b_sr_list = []
            for i in range(b):
                b_mask_gt = mask[i, :].unsqueeze(0)
                if sslopt.get('mask_stride', 0) > 1:
                    b_mask_gt = self.mask_stride * b_mask_gt
                if b_mask_gt.sum() == 0:
                    pass
                else:
                    b_gt = gt[i, :].unsqueeze(0)  # 1,3,256, 256
                    b_sr = sr[i, :].unsqueeze(0)  # 1,3,256,256
                    output_self_matrix = similarity_map(img=b_sr.clone(), mask=b_mask_gt.clone(),
                                                        simself_strategy=sslopt['simself_strategy'],
                                                        dh=sslopt.get('simself_dh', 16),
                                                        dw=sslopt.get('simself_dw', 16),
                                                        kernel_size=sslopt['kernel_size'],
                                                        scaling_factor=sslopt['scaling_factor'],
                                                        softmax=sslopt.get('softmax_sr', False),
                                                        temperature=sslopt.get('temperature', 0),
                                                        crossentropy=sslopt.get('crossentropy', False),
                                                        rearrange_back=sslopt.get('rearrange_back', True),
                                                        stride=1, pix_num=1, index=None,
                                                        kernel_size_center=sslopt.get('kernel_size_center', 9),
                                                        mean=sslopt.get('mean', False),
                                                        var=sslopt.get('var', False),
                                                        gene_type=sslopt.get('gene_type', "sum"),
                                                        largest_k=sslopt.get('largest_k', 0))
                    output_self_matrix = output_self_matrix.getitem()

                    gt_self_matrix = similarity_map(img=b_gt.clone(), mask=b_mask_gt.clone(),
                                                    simself_strategy=sslopt['simself_strategy'],
                                                    dh=sslopt.get('simself_dh', 16),
                                                    dw=sslopt.get('simself_dw', 16),
                                                    kernel_size=sslopt['kernel_size'],
                                                    scaling_factor=sslopt['scaling_factor'],
                                                    softmax=sslopt.get('softmax_gt', False),
                                                    temperature=sslopt.get('temperature', 0),
                                                    crossentropy=sslopt.get('crossentropy', False),
                                                    rearrange_back=sslopt.get('rearrange_back', True),
                                                    stride=1, pix_num=1, index=None,
                                                    kernel_size_center=sslopt.get('kernel_size_center', 9),
                                                    mean=sslopt.get('mean', False),
                                                    var=sslopt.get('var', False),
                                                    gene_type=sslopt.get('gene_type', "sum"),
                                                    largest_k=sslopt.get('largest_k', 0))
                    gt_self_matrix = gt_self_matrix.getitem()

                    b_sr_list.append(output_self_matrix)
                    b_gt_list.append(gt_self_matrix)
                    del output_self_matrix
                    del gt_self_matrix
            if len(b_sr_list) <=0 or len(b_gt_list) <= 0:
                return 0.0, 0.0
            b_sr_list = torch.cat(b_sr_list, dim=1)
            b_gt_list = torch.cat(b_gt_list, dim=1)

        if self.cri_selfsim:
            l_selfsim = self.cri_selfsim(b_sr_list, b_gt_list)
        else:
            l_selfsim = None


        if self.cri_selfsim1:
            l_selfsim_kl = self.cri_selfsim1(b_sr_list, b_gt_list)
        else:
            l_selfsim_kl = None


        if self.cri_selfsim or self.cri_selfsim1:
            del b_sr_list
            del b_gt_list

        return l_selfsim, l_selfsim_kl
