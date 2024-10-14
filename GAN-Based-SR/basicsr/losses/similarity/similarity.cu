#include <stdio.h>
#include <cmath>
#include <curand_kernel.h>

extern "C" 
__global__ void _compute_similarity_cuda(
    const float *image,
    const int *pos,
    float *out,
    const int mc,
    const int psize,
    const int ksize,
    const int height,
    const int width,
    const int channel) {

    int halfp = (psize-1)/2;
    int halfk = (ksize-1)/2;
    int psize2 = psize*psize;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int patch_index = index%psize2;
    int mc_index = (index-patch_index)/psize2;
    if(mc_index>=mc || patch_index>=psize2)
        return;

    // position of patch
    int patch_h = pos[2*mc_index+0];
    int patch_w = pos[2*mc_index+1];

    // position of current pixel in the patch
    int pindex_w = patch_index%psize;
    int pindex_w_in_img = pindex_w-halfp+patch_w;
    int pindex_h = (patch_index-pindex_w)/psize;
    int pindex_h_in_img = pindex_h-halfp+patch_h;

    float tmp;
    int c_distance = height*width;
    int p = 0;
    for(int c=0; c<channel; c++){
        for(int kh=-halfk; kh <=halfk; kh++){
            for(int kw=-halfk; kw <=halfk; kw++){
		if(kh+pindex_h<0 || kh+pindex_h>=psize || kw+pindex_w<0 || kw+pindex_w>=psize){
	            tmp = image[p + (patch_h+kh)*width + patch_w+kw];
		}else{
	            tmp = image[p + (patch_h+kh)*width + patch_w+kw] - image[p + (pindex_h_in_img+kh)*width + pindex_w_in_img+kw];
		}
                // printf("p:%d, pindex_h:%d, pindex_w:%d, kh:%d, kw:%d, patch_h:%d, patch_w:%d, pindex_h_in_img:%d, pindex_w_in_img:%d, tmp:%f\n", p, pindex_h, pindex_w, kh, kw, patch_h, patch_w, pindex_h_in_img, pindex_w_in_img);
	        out[mc_index*psize2+pindex_h*psize+pindex_w] += tmp*tmp;
	    }
	}
	p += c_distance;
    }
}

void _compute_similarity(
    const float *image,
    const int *pos, // the position of each target pos[mask_index] = (mask_w, mask_h)
    float *out, // shape (mc, psize, psize)
    const int mc, // the count of mask
    const int psize,
    const int ksize,
    const int height,
    const int width,
    const int channel) {
    int threads = 16;
    dim3 grid(floor(mc*psize*psize/threads)+1, 1);
    dim3 block(threads, 1);
    _compute_similarity_cuda<<<grid, block>>>(image, pos, out, mc, psize, ksize, height, width, channel);
}


extern "C" 
__global__ void _compute_similarity_backward_cuda(
    const float *image,
    const float *grads,
    const int *pos,
    float *image_grads,
    const int mc,
    const int psize,
    const int ksize,
    const int height,
    const int width,
    const int channel) {

    int halfp = (psize-1)/2;
    int halfk = (ksize-1)/2;
    int psize2 = psize*psize;
    int ksize2 = ksize*ksize;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int patch_index = index%psize2;
    int mc_index = (index-patch_index)/psize2;
    if(mc_index>=mc || patch_index>=psize2)
        return;

    // position of patch
    int patch_h = pos[2*mc_index+0];
    int patch_w = pos[2*mc_index+1];

    // position of current pixel in the patch
    int pindex_w = patch_index%psize;
    int pindex_w_in_img = pindex_w-halfp+patch_w;
    int pindex_h = (patch_index-pindex_w)/psize;
    int pindex_h_in_img = pindex_h-halfp+patch_h;

    float tmp;
    int c_distance = height*width;
    int c;
    int kernel_p = 0;
    int patch_p = 0;
    int kh, kw;
    for(int i=0; i<channel*ksize2; i++){
	c = (i+patch_index)%channel;
	kw = ((int)((i+patch_index-c)/channel)%ksize2)%ksize;
	kh = (((int)(i+patch_index-c)/channel)%ksize2-kw)/ksize;
	kw -= halfk;
	kh -= halfk;

        kernel_p = c*c_distance + (patch_h+kh)*width + patch_w+kw;
        patch_p = c*c_distance + (pindex_h_in_img+kh)*width + pindex_w_in_img+kw;

        if(kh+pindex_h<0 || kh+pindex_h>=psize || kw+pindex_w<0 || kw+pindex_w>=psize){
	    atomicAdd(image_grads+kernel_p, 2*image[kernel_p]*grads[index]);
	}else{
            tmp = 2*(image[kernel_p] - image[patch_p])*grads[index];
	    atomicAdd(image_grads+kernel_p, tmp);
	    atomicAdd(image_grads+patch_p, -tmp);
	}
    }
}

void _compute_similarity_backward(
    const float *image,
    const float *grads,
    const int *pos, // the position of each target pos[mask_index] = (mask_w, mask_h)
    float *image_grads,
    const int mc, // the count of mask
    const int psize,
    const int ksize,
    const int height,
    const int width,
    const int channel) {
    int threads = 16;
    dim3 grid(floor(mc*psize*psize/threads)+1, 1);
    dim3 block(threads, 1);
    _compute_similarity_backward_cuda<<<grid, block>>>(image, grads, pos, image_grads, mc, psize, ksize, height, width, channel);
}
