## Data preparation (for training GAN-based SR models):

Please following our dataset preparation steps since we need to generate edge mask in an off-line manner.
 
### Generate GT patches together with the corresponding edge mask.
1. You are supposed to download the training dataset:

 - [DIV2K(1-800 images for training)](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

 - [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)

 - [OST](https://github.com/xinntao/SFTGAN)

 - [FFHQ](https://github.com/NVlabs/ffhq-dataset)

 - [DIV8K](https://competitions.codalab.org/competitions/22217#participate-get-data)
   
   FFHQ and DIV8K are utilized to train DM-based models. 
2. We following the data augmentation strategy in [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/scripts/generate_multiscale_DF2K.py)
to generate multi-scale images for training.

```bash
cd GAN-Based-SR/scripts/data_preparation
python generate_multiscale_img.py --input [path to your downloaded dataset] --save [path to your save path]
```

3. Extract 512*512 patches from the multi-scale images.
```bash
cd GAN-Based-SR/scripts/data_preparation
python extract_subimages.py --input [path to your multi-scale images] --save [path to your save path]
```

4. We extract the edge mask with a Laplacian operator from the 512*512 patches. We save edge mask into "A.mat". "A" means the image patch's name.
In trainging progress, we read in "A.mat" and calculate SSG. 
And we also save the visualization of edge mask into "A.png" format. You could observe the detailed edge mask through "A.png".
```bash
cd GAN-Based-SR/scripts/data_preparation
python generate_mask.py --input [path to your patch images] --save [path to your save path]
```

### Generate bicubic Low-resolution training patches
1. Please use Matlab to generate x4 down-sampling LR images by
using the 512*512 GT patches in previous Section Step 4. The Matlab code could be seen [here](../scripts/matlab_scripts/generate_bicubic_img.m).
Or you could just use the [Python code](../basicsr/utils/matlab_functions.py) to generate LR patches. It has the same bicubic function as Matlab.

## Data preparation (for testing GAN-based SR models):

