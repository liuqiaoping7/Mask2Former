// Person post-processing: query weights (class softmax), the fused mask kernel and the vectorised
// panoptic rule. The fused kernel is ported from person_cuda.py (global / point / layered-linear);
// the other two replace the torch ops of seg._person_fused / seg._panoptic_person_rule.
#include <cuda_fp16.h>
#include "kernels.hpp"
#include "tex.cuh"

namespace k {
namespace {

__global__ void query_weights_kernel(const float* __restrict__ logits, int Q, int C, int person_id, float thr,
                                     float* wp, float* wd, float* sk, int* flag) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  const float* l = logits + (size_t)q * C;
  float mx = l[0];
  for (int c = 1; c < C; ++c) mx = fmaxf(mx, l[c]);
  float sum = 0.f;
  for (int c = 0; c < C; ++c) sum += expf(l[c] - mx);
  float best = -1.f, pp = 0.f, pv = 0.f;
  int label = 0;
  for (int c = 0; c < C; ++c) {
    float p = expf(l[c] - mx) / sum;
    if (p > best) { best = p; label = c; }
    if (c == person_id) pp = p;
    if (c == C - 1) pv = p;
  }
  bool keep = label != C - 1 && best > thr;
  wp[q] = pp;
  wd[q] = 1.f - pv;
  sk[q] = keep ? best : 0.f;
  flag[q] = keep && label == person_id;
}

__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }

// M: 0 global, 1 point fetches (1-D texture), 2 hardware bilinear (layered array)
template <int M>
__global__ void person_kernel(const __half* __restrict__ m, cudaTextureObject_t tex, const float* __restrict__ wp,
                              const float* __restrict__ wd, const float* __restrict__ sk, float* __restrict__ prob,
                              int* __restrict__ ids, uint8_t* __restrict__ ownat, int* __restrict__ owncnt, int Q,
                              int H4, int W4, int OH, int OW, float SY, float SX) {
  __shared__ int cnt[256];
  if (threadIdx.x < Q) cnt[threadIdx.x] = 0;
  __syncthreads();
  int offs = blockIdx.x * blockDim.x + threadIdx.x;
  bool inb = offs < OH * OW;
  int i = offs / OW, j = offs % OW;
  float iy = (i + 0.5f) * SY - 0.5f, ix = (j + 0.5f) * SX - 0.5f;
  float y0f = floorf(iy), x0f = floorf(ix);
  float wy1 = iy - y0f, wx1 = ix - x0f, wy0 = 1.f - wy1, wx0 = 1.f - wx1;
  int y0 = (int)y0f, x0 = (int)x0f, y1 = y0 + 1, x1 = x0 + 1;
  bool vy0 = y0 >= 0 && y0 < H4, vy1 = y1 >= 0 && y1 < H4, vx0 = x0 >= 0 && x0 < W4, vx1 = x1 >= 0 && x1 < W4;
  bool m00 = inb && vy0 && vx0, m01 = inb && vy0 && vx1, m10 = inb && vy1 && vx0, m11 = inb && vy1 && vx1;
  int o00 = y0 * W4 + x0, o01 = y0 * W4 + x1, o10 = y1 * W4 + x0, o11 = y1 * W4 + x1;
  float num = 0.f, den = 0.f, best = -1.f;
  int bid = 0, bown = 0, plane = H4 * W4;
  for (int q = 0; q < Q; ++q) {
    float s = 0.f;
    if (M == 2) {
      if (inb) s = tex2DLayered<float>(tex, ix + 0.5f, iy + 0.5f, q);
    } else {
      int b = q * plane;
      float v00 = 0.f, v01 = 0.f, v10 = 0.f, v11 = 0.f;
      if (M == 1) {
        if (m00) v00 = tex1Dfetch<float>(tex, b + o00);
        if (m01) v01 = tex1Dfetch<float>(tex, b + o01);
        if (m10) v10 = tex1Dfetch<float>(tex, b + o10);
        if (m11) v11 = tex1Dfetch<float>(tex, b + o11);
      } else {
        if (m00) v00 = to_f(m[b + o00]);
        if (m01) v01 = to_f(m[b + o01]);
        if (m10) v10 = to_f(m[b + o10]);
        if (m11) v11 = to_f(m[b + o11]);
      }
      s = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
    }
    float sig = 1.f / (1.f + expf(-s));
    num += wp[q] * sig;
    den += wd[q] * sig;
    bool own = s >= 0.f;
    unsigned mask = __ballot_sync(0xffffffffu, own && inb);
    if ((threadIdx.x & 31) == 0 && mask) atomicAdd(&cnt[q], __popc(mask));
    float p = sk[q] * sig;
    if (p > best) { best = p; bid = q; bown = own; }
  }
  if (inb) {
    prob[offs] = num / fmaxf(den, 1e-6f);
    ids[offs] = bid;
    ownat[offs] = bown;
  }
  __syncthreads();
  if (threadIdx.x < Q && cnt[threadIdx.x]) atomicAdd(&owncnt[threadIdx.x], cnt[threadIdx.x]);
}

// area[q] = pixels assigned to q, seg_area[q] = those inside q's own mask; per-block counts first
__global__ void query_hist_kernel(const int* __restrict__ ids, const uint8_t* __restrict__ own_at, int n, int Q,
                                  int* hist) {
  __shared__ int cnt[512];
  for (int i = threadIdx.x; i < 2 * Q; i += blockDim.x) cnt[i] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x * 16 + threadIdx.x, e = min(n, (int)(blockIdx.x + 1) * (int)blockDim.x * 16); i < e;
       i += blockDim.x) {
    atomicAdd(&cnt[ids[i]], 1);
    if (own_at[i]) atomicAdd(&cnt[Q + ids[i]], 1);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 2 * Q; i += blockDim.x)
    if (cnt[i]) atomicAdd(&hist[i], cnt[i]);
}

__global__ void valid_kernel(const int* hist, const int* own_cnt, const int* flag, float thr, int Q, uint8_t* valid) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  float area = hist[q], seg = hist[Q + q], own = own_cnt[q];
  valid[q] = flag[q] && area > 0 && own > 0 && seg > 0 && area >= thr * own;
}

__global__ void hard_kernel(const int* __restrict__ ids, const uint8_t* __restrict__ own_at,
                            const uint8_t* __restrict__ valid, int n, uint8_t* __restrict__ hard) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) hard[i] = valid[ids[i]] && own_at[i];
}

// the mask logits change every call: copy them into a layered array (block-linear texture memory)
cudaTextureObject_t layered_tex(const void* ptr, int Q, int H4, int W4, cudaStream_t s) {
  static std::map<std::tuple<int, int, int>, std::pair<cudaArray_t, cudaTextureObject_t>> arrays;
  auto& e = arrays[{Q, H4, W4}];
  if (!e.first) {
    cudaChannelFormatDesc d = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
    CU(cudaMalloc3DArray(&e.first, &d, make_cudaExtent(W4, H4, Q), cudaArrayLayered));
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = e.first;
    cudaTextureDesc td = {};
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeBorder;  // grid_sample's zero padding
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeElementType;
    CU(cudaCreateTextureObject(&e.second, &rd, &td, nullptr));
  }
  cudaMemcpy3DParms c = {};
  c.srcPtr = make_cudaPitchedPtr(const_cast<void*>(ptr), (size_t)W4 * 2, W4, H4);
  c.dstArray = e.first;
  c.extent = make_cudaExtent(W4, H4, Q);
  c.kind = cudaMemcpyDeviceToDevice;
  CU(cudaMemcpy3DAsync(&c, s));
  return e.second;
}

}  // namespace

void query_weights(const float* logits, int Q, int C, int person_id, float obj_thr, float* wp, float* wd, float* sk,
                   int* flag, cudaStream_t s) {
  query_weights_kernel<<<(Q + 31) / 32, 32, 0, s>>>(logits, Q, C, person_id, obj_thr, wp, wd, sk, flag);
}

void person(Mode mode, const void* masks, const float* wp, const float* wd, const float* sk, float* prob, int* ids,
            uint8_t* own_at, int* own_cnt, int Q, int H4, int W4, int OH, int OW, float sy, float sx, cudaStream_t s) {
  if (Q > 256) throw std::runtime_error("at most 256 queries");
  const __half* m = static_cast<const __half*>(masks);
  cudaTextureObject_t t = 0;
  if (mode == TEX_POINT)
    t = tex_1d<float>(masks, (size_t)Q * H4 * W4, cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat));
  if (mode == TEX_LINEAR) t = layered_tex(masks, Q, H4, W4, s);
  CU(cudaMemsetAsync(own_cnt, 0, Q * sizeof(int), s));
  int blocks = (OH * OW + 255) / 256;
#define RUN(M) person_kernel<M><<<blocks, 256, 0, s>>>(m, t, wp, wd, sk, prob, ids, own_at, own_cnt, Q, H4, W4, OH, OW, sy, sx)
  if (mode == GLOBAL) RUN(0);
  else if (mode == TEX_POINT) RUN(1);
  else RUN(2);
#undef RUN
  CU(cudaGetLastError());
}

void panoptic_person(const int* ids, const uint8_t* own_at, const int* own_cnt, const int* flag, float overlap_thr,
                     int Q, int n, int* hist, uint8_t* valid, uint8_t* hard, cudaStream_t s) {
  CU(cudaMemsetAsync(hist, 0, 2 * Q * sizeof(int), s));
  query_hist_kernel<<<(n + 256 * 16 - 1) / (256 * 16), 256, 0, s>>>(ids, own_at, n, Q, hist);
  valid_kernel<<<(Q + 31) / 32, 32, 0, s>>>(hist, own_cnt, flag, overlap_thr, Q, valid);
  hard_kernel<<<(n + 255) / 256, 256, 0, s>>>(ids, own_at, valid, n, hard);
  CU(cudaGetLastError());
}

}  // namespace k
