// Empty stand-in: the upstream ms_deform_im2col_cuda.cuh includes this torch header, but its forward
// kernel only needs CUDA itself (atomicAdd for float / double is native). Keeps the plugin free of torch.
#pragma once
