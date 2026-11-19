total kernel time (ncu, 1 eye, clock-control none): 456.7 ms, kernels: 1877
| stage | kernels | ms | % | SM% | Mem% | DRAM% | TensorCore% | compute | memory | latency |
|---|---|---|---|---|---|---|---|---|---|---|
| S1_backbone | 680 | 273.5 | 59.9% | 59 | 72 | 51 | 0.0 | 66% | 30% | 4% |
| S2_pixel_decoder | 335 | 53.8 | 11.8% | 51 | 66 | 52 | 6.0 | 53% | 36% | 12% |
| S3_transformer_decoder | 588 | 46.8 | 10.2% | 35 | 75 | 67 | 0.0 | 25% | 58% | 18% |
| S4_mask_upsample_to_input | 1 | 4.3 | 0.9% | 26 | 76 | 76 | 0.0 | 0% | 100% | 0% |
| S5_upsample_crop_1080p | 1 | 11.9 | 2.6% | 19 | 79 | 79 | 0.0 | 0% | 100% | 0% |
| S6_semantic_inference | 4 | 33.7 | 7.4% | 43 | 87 | 87 | 0.0 | 25% | 75% | 0% |
| S7_panoptic_inference | 265 | 28.3 | 6.2% | 12 | 91 | 90 | 0.0 | 0% | 91% | 9% |
| S8_person_prob | 3 | 4.5 | 1.0% | 4 | 97 | 97 | 0.0 | 0% | 100% | 0% |

## S1_backbone top kernels
   144.11 ms  x91   SM   74%  Mem   65%  compute  ampere_sgemm_128x64_tn
    37.37 ms  x282  SM   24%  Mem   93%  memory   at::elementwise_kernel<128, 2, at::gpu_kernel_impl_nocast<
    36.92 ms  x48   SM   68%  Mem   57%  compute  ampere_sgemm_128x128_nn
    12.21 ms  x24   SM   33%  Mem   95%  memory   <unnamed>::softmax_warp_forward<float, float, float, 8, 0,
    11.11 ms  x8    SM   72%  Mem   72%  memory   ampere_sgemm_64x64_tn
    10.60 ms  x24   SM   14%  Mem   95%  memory   at::vectorized_elementwise_kernel<4, at::GeluCUDAKernelImp
## S2_pixel_decoder top kernels
    18.78 ms  x24   SM   74%  Mem   65%  compute  ampere_sgemm_128x64_tn
     7.65 ms  x6    SM   62%  Mem   41%  compute  ms_deformable_im2col_gpu_kernel<float>
     5.23 ms  x1    SM   48%  Mem   30%  latency  sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc
     4.51 ms  x19   SM    2%  Mem   97%  memory   at::vectorized_elementwise_kernel<4, at::CUDAFunctor_add<f
     4.19 ms  x7    SM    2%  Mem   95%  memory   at::vectorized_elementwise_kernel<4, at::<unnamed>::launch
     3.37 ms  x82   SM   28%  Mem   92%  memory   at::elementwise_kernel<128, 2, at::gpu_kernel_impl_nocast<
## S3_transformer_decoder top kernels
     9.42 ms  x10   SM   75%  Mem   66%  compute  ampere_sgemm_64x64_nn
     5.61 ms  x18   SM   46%  Mem   33%  latency  ampere_sgemm_128x128_nn
     3.36 ms  x15   SM   21%  Mem   88%  memory   at::unrolled_elementwise_kernel<at::direct_copy_kernel_cud
     3.27 ms  x18   SM   31%  Mem   91%  memory   ampere_sgemm_128x128_tn
     3.18 ms  x9    SM    4%  Mem   97%  memory   at::reduce_kernel<512, 1, at::ReduceOp<long, at::func_wrap
     3.15 ms  x9    SM    3%  Mem   96%  memory   at::vectorized_elementwise_kernel<4, at::<unnamed>::masked
## S4_mask_upsample_to_input top kernels
     4.29 ms  x1    SM   26%  Mem   76%  memory   at::<unnamed>::upsample_bilinear2d_out_frame<float, float>
## S5_upsample_crop_1080p top kernels
    11.88 ms  x1    SM   19%  Mem   79%  memory   at::<unnamed>::upsample_bilinear2d_out_frame<float, float>
## S6_semantic_inference top kernels
    13.14 ms  x1    SM   11%  Mem   95%  memory   at::vectorized_elementwise_kernel<4, at::sigmoid_kernel_cu
    12.15 ms  x1    SM   55%  Mem   96%  memory   ampere_sgemm_64x64_nt
     8.42 ms  x1    SM   75%  Mem   62%  compute  ampere_sgemm_128x64_nt
     0.00 ms  x1    SM    9%  Mem   17%  latency  <unnamed>::softmax_warp_forward<float, float, float, 8, 0,
## S7_panoptic_inference top kernels
    13.14 ms  x1    SM   11%  Mem   95%  memory   at::vectorized_elementwise_kernel<4, at::sigmoid_kernel_cu
     4.14 ms  x60   SM    3%  Mem   92%  memory   at::reduce_kernel<512, 1, at::ReduceOp<long, at::func_wrap
     2.64 ms  x41   SM    5%  Mem   96%  memory   at::vectorized_elementwise_kernel<4, at::AUnaryFunctor<lon
     2.56 ms  x60   SM   34%  Mem   54%  latency  at::unrolled_elementwise_kernel<at::direct_copy_kernel_cud
     1.35 ms  x41   SM    6%  Mem   93%  memory   at::vectorized_elementwise_kernel<4, at::compare_scalar_ke
     1.27 ms  x1    SM    5%  Mem   95%  memory   at::vectorized_gather_kernel<16, long>
## S8_person_prob top kernels
     4.36 ms  x1    SM    4%  Mem   97%  memory   at::reduce_kernel<128, 4, at::ReduceOp<float, at::func_wra
     0.07 ms  x1    SM    7%  Mem   95%  memory   at::vectorized_elementwise_kernel<4, at::BinaryFunctor<flo
     0.03 ms  x1    SM    4%  Mem   93%  memory   at::vectorized_elementwise_kernel<4, at::<unnamed>::launch

overall: {'latency': '5.9%', 'memory': '43.9%', 'compute': '50.2%'}
