## Data preparation (for testing GAN-based SR models):

Please following our testing dataset preparation steps if you want to obtain the same results as our paper.
 
### 1.For bicubic degradation SR models.
We implement ESRGAN-SSL, RankSRGAN-SSL, SPSR-SSL, LDL-SSL, BebyGAN-SSL, ELANGAN-SSL, SwinIRGAN-SSL with bicubic degradation.

1.1 You are supposed to download the following testing dataset:

 - [DIV2K(801-900 images for testing)](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

 - [Set5](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)

 - [Set14](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)

 - [Urban100](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)

 - [BSDS100](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)
   
 - [Manga109](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)

 - [General100](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)
   

1.2. Please use [Matlab code](../scripts/matlab_scripts/generate_bicubic_img.m) or the [python bicubic](../basicsr/utils/matlab_functions.py) 
function to obtain x4 down-sampling low-resolution images. **Note that, before down-sample the GT images,
we crop the GT images so that their height/width could be divisible by 12.** The crop codes could be seen in
the [Matlab code](../scripts/matlab_scripts/generate_bicubic_img.m).


