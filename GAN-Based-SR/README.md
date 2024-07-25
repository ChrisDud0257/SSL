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
