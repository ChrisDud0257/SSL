# SSL
Official PyTorch code for our Paper "SSL" in ACM MM 2024.



> **SSL: A Self-similarity Loss for Improving Generative Image Super-resolution** <br>
> [Du CHEN\*](https://github.com/ChrisDud0257), [Zhengqiang ZHANG\*](https://github.com/xtudbxk), Jie LIANG and [Lei ZHANG](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>
> Accepted by ACM MM 2024.<br>


## Abstract
Generative adversarial networks (GAN) and generative diffusion models (DM) have been widely used 
in real-world image super-resolution (Real-ISR) to enhance the image perceptual quality. 
However, these generative models are prone to generating visual artifacts and false image structures, 
resulting in unnatural Real-ISR results. Based on the fact that natural images exhibit high 
self-similarities, i.e., a local patch can have many similar patches to it in the whole image, 
in this work we propose a simple yet effective self-similarity loss (SSL) to improve the performance 
of generative Real-ISR models, enhancing the hallucination of structural and textural details while 
reducing the unpleasant visual artifacts. Specifically, we compute a self-similarity graph (SSG) of 
the ground-truth image, and enforce the SSG of Real-ISR output to be close to it. To reduce the 
training cost and focus on edge areas, we generate an edge mask from the ground-truth image, and 
compute the SSG only on the masked pixels. The proposed SSL serves as a general plug-and-play 
penalty, which could be easily applied to the off-the-shelf Real-ISR models. Our experiments 
demonstrate that, by coupling with SSL, the performance of many state-of-the-art Real-ISR models, 
including those GAN and DM based ones, can be largely improved, reproducing more perceptually 
realistic image details and eliminating many false reconstructions and visual artifacts. 

## The implementation of Self-similarity Loss (SSL) when embedd into existing GAN-based or DM-based models:
![implementation](./figures/DMSSL.png)
 The GAN or DM network is employed to map the input LR image to an ISR output. We calculate 
the self-similarity graphs (SSG) of both ISR output and ground-truth (GT) image, 
and calculate the SSL between them to supervise the generation of image details and 
structures.


## The calculation progress of Self-similarity Graph (SSG)
![SSG](./figures/SSG.png)
We first generate a mask to indicate the image edge areas by applying the Laplacian 
Operator on the GT image. During the training period, for each edge pixel in the mask, 
we find the corresponding pixels in the GT image and ISR image, and set a search area 
centred at them. A local sliding window is utilized to calculate the similarity between 
each pixel in the search area and the central pixel so that an SSG can be respectively 
computed for the GT image and the ISR image, with which the SSL can be computed. 
The red pixel means the edge pixel, while the blue block means the sliding window.



## Getting started for GAN-based Models

### 1.Installation
 - Python == 3.9
 - CUDA == 11.7 
 - PyTorch == 1.13.1
 - Anaconda

Since we implement SSL with a hand-crafted CUDA operator, 
please make sure you have already installed a correct CUDA version. 
We have tested that at least with CUDA-11.3, or a higher version is just OK.

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
 - At last, compile the bascisr framework.
```bash
BASICSR_EXT=True python setup.py develop
```

All of the models are trained under the excellent BasicSR framework. For any installation issues,
please refer to [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md).

### 2.Data preparation (for training):

 - Please prepare the training dataset by following this [instruction](GAN-Based-SR/datasets/README.md).








