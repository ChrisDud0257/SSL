## Getting started for Diffusion-based Models

### 1.Installation
 - Python == 3.9
 - CUDA == 11.7
 - PyTorch == 1.10
 - ninja
 - loguru
 - prettytable

**Since we implement SSL with a hand-crafted CUDA operator, 
please make sure you have already installed a correct CUDA version. 
We have tested that the CUDA version from 11.3 to 11.7 is just OK.**

 - clone this repo.
> git clone https://github.com/ChrisDud0257/SSL && cd Diffusion-Based-SR

 - Since this example code is integrated into [StableSR](https://github.com/IceClear/StableSR), please follow their instrucations for installation. For the integration to other diffusion framework, please refer to the [Integration to other diffusion framework](#other_diffusion_framework).

### 2.Data preparation:
Please refer to [the instructions in GAN-Based-SR](../GAN-Based-SR/README.md) for details.

### 3.Training commands

You can use the following command to start the training:
> python3 main.py --train --base configs/SSL/base.yaml --gpus 0,1 --name stablesr_ssl --scale_lr False --logdir experiments

*Remember to modify the base.yml according to your settings, such as the data paths and pretrained model path of [VAE](https://github.com/IceClear/StableSR).*


## 4. Testing commands

You can use the following command to test the trained model:
> python test.py --config configs/SSL/base.yaml --ckpt xxx.ckpt --vqgan_ckpt vqgan.ckpt --init-img path_to_LR --outdir results --ddpm_steps 200 --dec_w 0.0 --colorfix_type adain --random-count 1

*Please to set the path of test folder and models.*

## 5. Trained models
You could download all of our well-trained models through [GoogleDrive]
or [BaiduDrive].

## 6. Intergration to other diffusion framework 
<!-- <a id="other_diffusion_framework" /> -->

The integration of SSL function to other diffusion framework can be divided into **two** parts. The first part is related to the initialization of SSL function while the second one computes the SSL function during each training iteration.

- **Initialization**
The initialization aims to pre-process the mask datas and pre-compile the cuda-version SSL function. It will only be called once at the beginning. You can directly use the same code as those used in this StableSR framework. The source code is the "init_issl_settings" in the L46-L66 of `ldm/models/diffusion/ddpmssl.py`:

```python
    def init_issl_settings(self):
        if self.sslopt.get('mask_stride', 0) > 1:
            mask_size = int(self.configs.model.params['image_size'])
            mask_stride = torch.eye(self.sslopt.get('mask_stride', 0), self.sslopt.get('mask_stride', 0),
                                    dtype=torch.float32)
            mask_stride = mask_stride.repeat(math.ceil(mask_size / self.sslopt.get('mask_stride', 0)),
                                             math.ceil(mask_size / self.sslopt.get('mask_stride', 0)))
            mask_stride = mask_stride[:mask_size, :mask_size]
            mask_stride = mask_stride.unsqueeze(0).unsqueeze(0)
            self.mask_stride = nn.Parameter(data=mask_stride, requires_grad=False).cuda()
            print(f"mask stride is {self.sslopt.get('mask_stride', 0)}")

        if self.configs.ISSL_loss.get('selfsim_opt'):
            self.cri_selfsim = build_loss(self.configs.ISSL_loss['selfsim_opt']).to(self.device)
        else:
            self.cri_selfsim = None

        if self.configs.ISSL_loss.get('selfsim1_opt'):
            self.cri_selfsim1 = build_loss(self.configs.ISSL_loss['selfsim1_opt']).to(self.device)
        else:
            self.cri_selfsim1 = None
```

- **Compute SSL during training**
In each training iteration, please follow the code of function "issl" in L423-L496 of `ldm/models/diffusion/ddpmssl.py`:

```python
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
```