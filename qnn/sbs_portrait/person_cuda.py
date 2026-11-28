"""CUDA global / texture implementations of person_triton.fused_person (same arguments and outputs),
in the order cuda -> texture -> triton (the production kernel).

  cuda        global-memory loads of the four bilinear taps of each of the Q mask planes
  tex_point   the same loads through the texture unit (1-D texture over the logits, point fetches)
  tex_linear  hardware bilinear filtering: the logits are copied every call into a layered CUDA array
              (block-linear texture memory) and sampled with one fetch per query; border address mode
              reproduces grid_sample's zero padding. The filter fraction is 8-bit fixed point.

Own-area counts: warp ballot -> block shared counters -> one global atomic per query per block.
"""
import torch

from warp_box import load_ext

KINDS = ("cuda", "tex_point", "tex_linear")

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <map>
#include <tuple>

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }

// M: 0 global, 1 point fetches (1-D texture), 2 hardware bilinear (layered array)
template <int M, typename T>
__global__ void person_kernel(const T* __restrict__ m, cudaTextureObject_t tex, const float* __restrict__ wp,
                              const float* __restrict__ wd, const float* __restrict__ sk, float* __restrict__ prob,
                              int* __restrict__ ids, signed char* __restrict__ ownat, int* __restrict__ owncnt,
                              int Q, int H4, int W4, int OH, int OW, float SY, float SX) {
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
  if (inb) { prob[offs] = num / fmaxf(den, 1e-6f); ids[offs] = bid; ownat[offs] = bown; }
  __syncthreads();
  if (threadIdx.x < Q && cnt[threadIdx.x]) atomicAdd(&owncnt[threadIdx.x], cnt[threadIdx.x]);
}

static std::map<std::tuple<const void*, size_t, int>, cudaTextureObject_t> g_tex;
static std::map<std::tuple<int, int, int, int>, std::pair<cudaArray_t, cudaTextureObject_t>> g_arr;

static cudaChannelFormatDesc desc_of(bool half) {
  return cudaCreateChannelDesc(half ? 16 : 32, 0, 0, 0, cudaChannelFormatKindFloat);
}

static cudaTextureObject_t point_tex(const void* ptr, size_t n, bool half) {
  auto key = std::make_tuple(ptr, n, (int)half);
  auto it = g_tex.find(key);
  if (it != g_tex.end()) return it->second;
  cudaResourceDesc rd = {};
  rd.resType = cudaResourceTypeLinear;
  rd.res.linear.devPtr = (void*)ptr;
  rd.res.linear.desc = desc_of(half);
  rd.res.linear.sizeInBytes = n * (half ? 2 : 4);
  cudaTextureDesc td = {};
  td.filterMode = cudaFilterModePoint;
  td.readMode = cudaReadModeElementType;
  cudaTextureObject_t t = 0;
  TORCH_CHECK(cudaCreateTextureObject(&t, &rd, &td, nullptr) == cudaSuccess, "cudaCreateTextureObject failed");
  g_tex[key] = t;
  return t;
}

static cudaTextureObject_t layered_tex(const void* ptr, int Q, int H4, int W4, bool half) {
  auto& e = g_arr[std::make_tuple(Q, H4, W4, (int)half)];
  if (!e.first) {
    cudaChannelFormatDesc d = desc_of(half);
    TORCH_CHECK(cudaMalloc3DArray(&e.first, &d, make_cudaExtent(W4, H4, Q), cudaArrayLayered) == cudaSuccess,
                "cudaMalloc3DArray failed");
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = e.first;
    cudaTextureDesc td = {};
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeBorder;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeElementType;
    TORCH_CHECK(cudaCreateTextureObject(&e.second, &rd, &td, nullptr) == cudaSuccess, "cudaCreateTextureObject failed");
  }
  cudaMemcpy3DParms c = {};
  size_t row = (size_t)W4 * (half ? 2 : 4);
  c.srcPtr = make_cudaPitchedPtr((void*)ptr, row, W4, H4);
  c.dstArray = e.first;
  c.extent = make_cudaExtent(W4, H4, Q);
  c.kind = cudaMemcpyDeviceToDevice;
  TORCH_CHECK(cudaMemcpy3DAsync(&c, c10::cuda::getCurrentCUDAStream()) == cudaSuccess, "cudaMemcpy3DAsync failed");
  return e.second;
}

template <typename T>
static void run(torch::Tensor m, torch::Tensor wp, torch::Tensor wd, torch::Tensor sk, torch::Tensor prob,
                torch::Tensor ids, torch::Tensor ownat, torch::Tensor cnt, double sy, double sx, int mode) {
  int Q = m.size(0), H4 = m.size(1), W4 = m.size(2), OH = prob.size(0), OW = prob.size(1);
  TORCH_CHECK(Q <= 256, "at most 256 queries");
  const T* p = static_cast<const T*>(m.data_ptr());
  bool half = sizeof(T) == 2;
  cudaTextureObject_t t = mode == 1 ? point_tex(p, m.numel(), half) : mode == 2 ? layered_tex(p, Q, H4, W4, half) : 0;
  auto st = c10::cuda::getCurrentCUDAStream();
  int blocks = (OH * OW + 255) / 256;
#define RUN(M) person_kernel<M, T><<<blocks, 256, 0, st>>>(p, t, wp.data_ptr<float>(), wd.data_ptr<float>(), \
    sk.data_ptr<float>(), prob.data_ptr<float>(), ids.data_ptr<int>(), (signed char*)ownat.data_ptr<int8_t>(), \
    cnt.data_ptr<int>(), Q, H4, W4, OH, OW, (float)sy, (float)sx)
  if (mode == 0) RUN(0); else if (mode == 1) RUN(1); else RUN(2);
#undef RUN
}

void person(torch::Tensor m, torch::Tensor wp, torch::Tensor wd, torch::Tensor sk, torch::Tensor prob,
            torch::Tensor ids, torch::Tensor ownat, torch::Tensor cnt, double sy, double sx, int64_t mode) {
  if (m.scalar_type() == at::kHalf) run<__half>(m, wp, wd, sk, prob, ids, ownat, cnt, sy, sx, mode);
  else run<float>(m, wp, wd, sk, prob, ids, ownat, cnt, sy, sx, mode);
}
"""
_CPP_SRC = r"""
#include <torch/extension.h>
void person(torch::Tensor m, torch::Tensor wp, torch::Tensor wd, torch::Tensor sk, torch::Tensor prob,
            torch::Tensor ids, torch::Tensor ownat, torch::Tensor cnt, double sy, double sx, int64_t mode);
"""
_ext = None


def fused_person(kind, masks, w_person, w_den, score_keep, out_hw, scale_y, scale_x):
    """person_triton.fused_person with kind in KINDS."""
    global _ext
    if _ext is None:
        _ext = load_ext("person_cuda_ext", _CPP_SRC, _CUDA_SRC, ["person"])
    oh, ow = out_hw
    dev = masks.device
    prob = torch.empty(oh, ow, device=dev, dtype=torch.float32)
    ids = torch.empty(oh, ow, device=dev, dtype=torch.int32)
    own_at = torch.empty(oh, ow, device=dev, dtype=torch.int8)
    own_cnt = torch.zeros(masks.shape[0], device=dev, dtype=torch.int32)
    _ext.person(masks.contiguous(), w_person.float().contiguous(), w_den.float().contiguous(),
                score_keep.float().contiguous(), prob, ids, own_at, own_cnt, float(scale_y), float(scale_x),
                KINDS.index(kind))
    return prob, ids.long(), own_at.bool(), own_cnt.float()
