#include "similarity.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void compute_similarity(
    torch::Tensor &image,
    torch::Tensor &pos,
    torch::Tensor &out,
    const int mc, 
    const int psize, 
    const int ksize, 
    const int height, 
    const int width, 
    const int channel){
	
    CHECK_INPUT(image);
    CHECK_INPUT(pos);
    CHECK_INPUT(out);

    // run the code at the cuda device same with the input
    const at::cuda::OptionalCUDAGuard device_guard(device_of(image));

    _compute_similarity(
        (const float *)image.data_ptr(),
        (const int *)pos.data_ptr(),
        (float *)out.data_ptr(),
        mc, 
        psize, 
        ksize, 
        height, 
        width, 
        channel);

}

void compute_similarity_backward(
    torch::Tensor &image,
    torch::Tensor &grads,
    torch::Tensor &pos,
    torch::Tensor &image_grads,
    const int mc, 
    const int psize, 
    const int ksize, 
    const int height, 
    const int width, 
    const int channel){

    CHECK_INPUT(image);
    CHECK_INPUT(pos);
    CHECK_INPUT(grads);
    CHECK_INPUT(image_grads);

    // run the code at the cuda device same with the input
    const at::cuda::OptionalCUDAGuard device_guard(device_of(image));

    _compute_similarity_backward(
        (const float *)image.data_ptr(),
        (const float *)grads.data_ptr(),
        (const int *)pos.data_ptr(),
        (float *)image_grads.data_ptr(),
        mc, 
        psize, 
        ksize, 
        height, 
        width, 
        channel);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_similarity",
          &compute_similarity,
          "cuda version wrapper");

    m.def("compute_similarity_backward",
          &compute_similarity_backward,
          "cuda version wrapper");
}
