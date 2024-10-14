
void _compute_similarity(
    const float *image,
    const int *pos, // the position of each target pos[mask_index] = (mask_w, mask_h)
    float *out, // shape (mc, psize, psize)
    const int mc, // the count of mask
    const int psize,
    const int ksize,
    const int height,
    const int width,
    const int channel);

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
    const int channel);
