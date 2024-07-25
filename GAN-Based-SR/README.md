## Getting started for GAN-based Models

### 1.Installation
 - Python == 3.9
 - CUDA == 11.7 
 - PyTorch == 1.13.1
 - Anaconda

**Since we implement SSL with a hand-crafted CUDA operator, 
please make sure you have already installed a correct CUDA version. 
We have tested that the CUDA version from 11.3 to 11.7 is just OK.**

 - clone this repo.
```bash
git clone https://github.com/ChrisDud0257/SSL
cd GAN-Based-SR
conda create --name ssl-gan python=3.9
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
 - Then you should make sure you have installed the CUDA correctly. Export your CUDA path into
the environment. For example, in my path
```bash
export CUDA_HOME=/data0/chendu/cuda-11.7
```
 - At last, compile the BasicSR framework.
```bash
BASICSR_EXT=True python setup.py develop
```

All of the models are trained under the excellent BasicSR framework. For any installation issues,
please refer to [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md).

### 2.Data preparation:
For training:
 - Please prepare the training dataset by following this [instruction](datasets/README.md).

For testing:
 - Please prepare the testing dataset by following this [instruction](datasets/README_TEST.MD).

### 3.Training commands
We write all of the training/testing commands into the ["demo.sh"](demo.sh) file. 

We provide all of the codes that have been reported in our paper:
 - ESRGAN-SSL
 - RankSRGAN-SSL (We train the SSL-guided counterpart with RankSRGAN-PI model)
 - SPSR-SSL
 - LDL-SSL
 - BebyGAN-SSL
 - ELANGAN-SSL
 - SwinIRGAN-SSL

3.1 Firstly, you need to modify the training configuration file, for example, in [train_ESRGANSSL_bicubic_x4.yml](options/train/ESRGANSSL/train_ESRGANSSL_bicubic_x4.yml),
you are supposed to modify:
```bash
(line 17) dataroot_gt: [path to the 512*512 patches]
(line 18) dataroot_lq: [path to the x4 downsampling patches]
(line 19) dataroot_gt_mask: [path to the edge mask, they are saved as ".mat" in the "threshold-20/mat" folder]

(line 42) dataroot_gt: [path to the testing GT]
(line 43) dataroot_lq: [path to the testing LR]

(line 64) pretrain_network_g: [path to the pretrained ESRGAN-PSNR model]
```
Note that, for each SSL-guided version, we finetune the SSL-guided counterparts from a well-trained 
PSNR-oriented version. You are supposed to download the pretrained models provided by the original method.
And then modify the "pretrain_network_g" path in line 64.

If your GPU memory is limited, please decrease the training batch size,
```bash
(line 35) batch_size_per_gpu: [please set to an appropriate value that will not raise CUDA memory error]
```


3.2 Start training
 - For example, when training ESRGAN-SSL, you could use this for single GPU training:
```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/ESRGANSSL/train_ESRGANSSL_bicubic_x4.yml --auto_resume
```
 - Or use this for DDP training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt ./options/train/ESRGANSSL/train_ESRGANSSL_bicubic_x4.yml --launcher pytorch --auto_resume
```

## 4. Testing commands

4.1 Firstly, you need to modify the testing configuration file, for example,
in [test_ESRGANSSL_bicubic_x4.yml](options/test/ESRGANSSL/test_ESRGANSSL_bicubic_x4.yml),
you are suuposed to modify the testing datasets' paths from line 13 to line 62.
And together with the well-trained ESRGAN-SSL model path:

```bash
(line 13)test_1:  # the 1st test dataset
           name: Set5
           type: PairedImageDataset
           dataroot_gt: [path to Set5 GT]
           dataroot_lq: [path to Set5 x4 downsampled LR]
           io_backend:
             type: disk
......
(line 75) pretrain_network_g: [path to the well-trained ESRGAN-SSL model]
```

4.2 The models provided by us. You could download all of our well-trained models through [GoogleDrive]
or [BaiduDrive].

For fair comparison, we download all of the original GAN-based SR models and re-test them, which means
we test our SSL-guided version and the original model with the same datasets, 
the same border cropping size, the same border shaving size, the IQA metric codes and so on.

4.3 Start testing
 - For example, when testing ESRGAN-SSL, you could use this command:
```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/ESRGANSSL/test_ESRGANSSL_bicubic_x4.yml
```
