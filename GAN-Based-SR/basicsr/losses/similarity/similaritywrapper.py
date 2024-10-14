import os
import sys
from loguru import logger
curdir = os.path.split(__file__)[0]

import torch
from torch.utils.cpp_extension import load
from torch.autograd import Function
from torch.autograd.function import once_differentiable

if not torch.cuda.is_available():
    logger.error(f'compute_similarity has to run undeer cuda environment')
    sys.exit()

SIMWrapper = load(name="compute_similarity",
    sources=[os.path.join(curdir, "similaritywrapper.cpp"), 
            os.path.join(curdir, "similarity.cu")],
            verbose=True)

SIMWrapper_bw = load(name="compute_similarity_backward",
    sources=[os.path.join(curdir, "similaritywrapper.cpp"), 
            os.path.join(curdir, "similarity.cu")],
            verbose=True)

class ComputeSimilarityCUDA(Function):
    @staticmethod
    def forward(ctx, image_pad, mask_pad, pos, psize, ksize):
        mc = pos.shape[0]
        out = torch.zeros(mc, psize, psize, dtype=torch.float32, device=image_pad.device).contiguous()
        channel, height, width = image_pad.shape

        SIMWrapper.compute_similarity(image_pad, pos, out, mc, psize, ksize, height, width, channel)
        ctx.save_for_backward(image_pad, mask_pad, pos)
        ctx.psize = psize
        ctx.ksize = ksize
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # print(grad_output)
        image_pad, mask_pad, pos = ctx.saved_tensors
        channel, height, width = image_pad.shape
        psize, ksize = ctx.psize, ctx.ksize
        mc = pos.shape[0]

        image_grad = torch.zeros(channel, height, width, dtype=torch.float32, device=image_pad.device).contiguous()

        import time
        torch.cuda.synchronize()
        st = time.time()
        SIMWrapper_bw.compute_similarity_backward(image_pad, grad_output, pos, image_grad, mc, psize, ksize, height, width, channel)
        torch.cuda.synchronize()
        et = time.time()
        # print(f'backward: {(et-st)*1000} ms')

        return (image_grad, None, None, None, None)

def compute_similarity(image, mask, psize=25, ksize=9): 
    if image.is_cuda == False or mask.is_cuda == False:
        logger.error(f'compute_similarity only accepts tensors on GPU memory but image({image.device}), mask({mask.device})')
        sys.exit()

    plen = psize//2
    image_pad = torch.nn.functional.pad(image, (plen, plen, plen, plen), mode='reflect').contiguous()
    mask_pad = torch.nn.functional.pad(mask, (plen, plen, plen, plen), mode='constant')
    pos = torch.nonzero(mask_pad==1).to(torch.int32).contiguous()
    mask_pad = mask_pad.to(torch.int32).contiguous()
    return ComputeSimilarityCUDA.apply(image_pad, mask_pad, pos, psize, ksize)
