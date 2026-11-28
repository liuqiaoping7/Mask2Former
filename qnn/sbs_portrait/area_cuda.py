"""CUDA reference implementations of area_triton.area_resize_normalize (same arguments, same output),
for comparing the Triton kernel against global-memory loads and texture sampling.

  cuda        the same algorithm in CUDA, global-memory loads (control: Triton vs CUDA by itself)
  tex_point   the same taps and weights, every source byte read through the texture unit: a 1-D uchar
              texture over the frame's linear memory (3 interleaved channels need no conversion and
              any width / eye offset works; no pitch alignment requirement)
  tex_linear  hardware bilinear filtering: the eye is first converted BGR -> BGRA (textures need 1, 2
              or 4 channels) into a 512-byte-pitch plane, adjacent taps are merged into one linear
              fetch (K taps -> ceil(K / 2) per axis, one fetch returns all channels). The filter
              fraction is 8-bit fixed point, so values can differ from OpenCV by a gray level.
"""
import math
import os
import sys

import torch

KINDS = ("cuda", "tex_point", "tex_linear")

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <map>
#include <tuple>

struct Out {
  float* p; int nh, nw, flip; float mean[3], std[3];
  __device__ __forceinline__ void store(int offs, int c, float v) const {
    v = fminf(fmaxf(rintf(v), 0.f), 255.f);  // saturate_cast<uchar>
    int oc = flip ? 2 - c : c;
    p[(size_t)oc * nh * nw + offs] = (v - mean[oc]) / std[oc];
  }
};

// one output pixel per thread; TEX: the source bytes come from a linear-memory texture
template <bool TEX, int K>
__global__ void area_taps_kernel(const unsigned char* __restrict__ src, cudaTextureObject_t tex, int off, int row,
                                 const int* __restrict__ ix, const float* __restrict__ wx,
                                 const int* __restrict__ iy, const float* __restrict__ wy, Out o) {
  int offs = blockIdx.x * blockDim.x + threadIdx.x;
  if (offs >= o.nh * o.nw) return;
  int y = offs / o.nw, x = offs % o.nw;
  int sx[K]; float fx[K];
  for (int k = 0; k < K; ++k) { sx[k] = ix[x * K + k] * 3; fx[k] = wx[x * K + k]; }
  float acc[3] = {0.f, 0.f, 0.f};
  for (int ky = 0; ky < K; ++ky) {
    int base = iy[y * K + ky] * row; float fy = wy[y * K + ky];
    float r[3] = {0.f, 0.f, 0.f};
    for (int kx = 0; kx < K; ++kx) {
      int p = base + sx[kx];
      for (int c = 0; c < 3; ++c) {
        float v = TEX ? (float)tex1Dfetch<unsigned char>(tex, off + p + c) : (float)src[p + c];
        r[c] += fx[kx] * v;
      }
    }
    for (int c = 0; c < 3; ++c) acc[c] += fy * r[c];
  }
  for (int c = 0; c < 3; ++c) o.store(offs, c, acc[c]);
}

// M x M hardware-bilinear fetches per output pixel at the merged tap positions
template <int M>
__global__ void area_linear_kernel(cudaTextureObject_t tex, const float* __restrict__ px, const float* __restrict__ wx,
                                   const float* __restrict__ py, const float* __restrict__ wy, Out o) {
  int offs = blockIdx.x * blockDim.x + threadIdx.x;
  if (offs >= o.nh * o.nw) return;
  int y = offs / o.nw, x = offs % o.nw;
  float u[M], fx[M];
  for (int k = 0; k < M; ++k) { u[k] = px[x * M + k]; fx[k] = wx[x * M + k]; }
  float acc[3] = {0.f, 0.f, 0.f};
  for (int ky = 0; ky < M; ++ky) {
    float v_ = py[y * M + ky], fy = wy[y * M + ky];
    float r[3] = {0.f, 0.f, 0.f};
    for (int kx = 0; kx < M; ++kx) {
      float4 t = tex2D<float4>(tex, u[kx], v_);
      r[0] += fx[kx] * t.x; r[1] += fx[kx] * t.y; r[2] += fx[kx] * t.z;
    }
    for (int c = 0; c < 3; ++c) acc[c] += fy * r[c];
  }
  for (int c = 0; c < 3; ++c) o.store(offs, c, acc[c] * 255.f);  // normalized-float read mode
}

// BGR (H, W, 3) with a free row stride -> BGRA (H, Wp, 4)
__global__ void eye_to_rgba_kernel(const unsigned char* __restrict__ src, int row, uchar4* __restrict__ dst, int H, int W, int Wp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x >= W) return;
  const unsigned char* s = src + (size_t)y * row + x * 3;
  dst[(size_t)y * Wp + x] = make_uchar4(s[0], s[1], s[2], 255);
}

static std::map<std::tuple<const void*, size_t, int, size_t>, cudaTextureObject_t> g_tex;

// h == 0: 1-D uchar point texture over `w` bytes; else 2-D pitched uchar4 linear-filter texture
static cudaTextureObject_t tex_for(void* ptr, size_t w, int h, size_t pitch) {
  auto key = std::make_tuple((const void*)ptr, w, h, pitch);
  auto it = g_tex.find(key);
  if (it != g_tex.end()) return it->second;
  cudaResourceDesc rd = {};
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
  td.normalizedCoords = 0;
  if (h == 0) {
    rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr = ptr;
    rd.res.linear.desc = cudaCreateChannelDesc<unsigned char>();
    rd.res.linear.sizeInBytes = w;
    td.filterMode = cudaFilterModePoint;
    td.readMode = cudaReadModeElementType;
  } else {
    rd.resType = cudaResourceTypePitch2D;
    rd.res.pitch2D.devPtr = ptr;
    rd.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
    rd.res.pitch2D.width = w; rd.res.pitch2D.height = h; rd.res.pitch2D.pitchInBytes = pitch;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeNormalizedFloat;
  }
  cudaTextureObject_t t = 0;
  TORCH_CHECK(cudaCreateTextureObject(&t, &rd, &td, nullptr) == cudaSuccess, "cudaCreateTextureObject failed");
  g_tex[key] = t;
  return t;
}

static Out make_out(torch::Tensor out, bool flip, const std::vector<double>& mean, const std::vector<double>& std_) {
  Out o;
  o.p = out.data_ptr<float>(); o.nh = out.size(1); o.nw = out.size(2); o.flip = flip;
  for (int c = 0; c < 3; ++c) { o.mean[c] = mean[c]; o.std[c] = std_[c]; }
  return o;
}

// base: the uint8 allocation holding the eye (the texture covers it from its 512-byte aligned start),
// off: byte offset of the eye's first pixel in it
void area_taps(torch::Tensor eye, torch::Tensor base, int64_t off, torch::Tensor ix, torch::Tensor wx, torch::Tensor iy,
               torch::Tensor wy, torch::Tensor out, bool use_tex, bool flip, std::vector<double> mean, std::vector<double> std_) {
  Out o = make_out(out, flip, mean, std_);
  int k = ix.size(1), row = eye.stride(0);
  cudaTextureObject_t t = use_tex ? tex_for(base.data_ptr(), base.numel(), 0, 0) : 0;
  const unsigned char* s = eye.data_ptr<unsigned char>();
  const int *pix = ix.data_ptr<int>(), *piy = iy.data_ptr<int>();
  const float *pwx = wx.data_ptr<float>(), *pwy = wy.data_ptr<float>();
  int blocks = (o.nh * o.nw + 255) / 256;
  auto st = c10::cuda::getCurrentCUDAStream();
#define RUN(K) case K: \
    if (use_tex) area_taps_kernel<true, K><<<blocks, 256, 0, st>>>(s, t, off, row, pix, pwx, piy, pwy, o); \
    else area_taps_kernel<false, K><<<blocks, 256, 0, st>>>(s, t, off, row, pix, pwx, piy, pwy, o); \
    break;
  switch (k) { RUN(2) RUN(3) RUN(4) RUN(5) default: TORCH_CHECK(false, "unsupported tap count ", k); }
#undef RUN
}

void eye_to_rgba(torch::Tensor eye, torch::Tensor dst) {
  int H = eye.size(0), W = eye.size(1);
  dim3 blk(256), grd((W + 255) / 256, H);
  eye_to_rgba_kernel<<<grd, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(eye.data_ptr<unsigned char>(), eye.stride(0),
      (uchar4*)dst.data_ptr<unsigned char>(), H, W, dst.size(1));
}

void area_linear(torch::Tensor rgba, int64_t w, torch::Tensor px, torch::Tensor wx, torch::Tensor py, torch::Tensor wy,
                 torch::Tensor out, bool flip, std::vector<double> mean, std::vector<double> std_) {
  Out o = make_out(out, flip, mean, std_);
  cudaTextureObject_t t = tex_for(rgba.data_ptr(), w, rgba.size(0), rgba.stride(0));
  int m = px.size(1), blocks = (o.nh * o.nw + 255) / 256;
  auto st = c10::cuda::getCurrentCUDAStream();
#define RUN(M) case M: area_linear_kernel<M><<<blocks, 256, 0, st>>>(t, px.data_ptr<float>(), wx.data_ptr<float>(), \
      py.data_ptr<float>(), wy.data_ptr<float>(), o); break;
  switch (m) { RUN(1) RUN(2) RUN(3) default: TORCH_CHECK(false, "unsupported fetch count ", m); }
#undef RUN
}
"""
_CPP_SRC = r"""
#include <torch/extension.h>
void area_taps(torch::Tensor eye, torch::Tensor base, int64_t off, torch::Tensor ix, torch::Tensor wx, torch::Tensor iy,
               torch::Tensor wy, torch::Tensor out, bool use_tex, bool flip, std::vector<double> mean, std::vector<double> std_);
void eye_to_rgba(torch::Tensor eye, torch::Tensor dst);
void area_linear(torch::Tensor rgba, int64_t w, torch::Tensor px, torch::Tensor wx, torch::Tensor py, torch::Tensor wy,
                 torch::Tensor out, bool flip, std::vector<double> mean, std::vector<double> std_);
"""
_ext = None


def cuda_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
        os.environ["PATH"] = os.pathsep.join([os.path.dirname(sys.executable),
                                              os.path.join(os.environ["CUDA_HOME"], "bin"), os.environ["PATH"]])
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        _ext = load_inline("area_cuda_ext", _CPP_SRC, cuda_sources=_CUDA_SRC,
                           functions=["area_taps", "eye_to_rgba", "area_linear"], extra_cuda_cflags=["-O3"], verbose=False)
    return _ext


def merge_taps(taps):
    """Pair adjacent taps (i, a), (i + 1, b) into one linear fetch at texel coordinate
    i + 0.5 + b / (a + b) with weight a + b. taps: [(indices, weights)] per output index.
    Returns (positions, weights), each (n, M) float32 padded with zero-weight fetches."""
    merged = []
    for idx, w in taps:
        m = []
        for j in range(0, len(idx), 2):
            i, a = idx[j], w[j]
            i1, b = (idx[j + 1], w[j + 1]) if j + 1 < len(idx) else (i, 0.0)
            assert i1 - i in (0, 1), "taps must be consecutive source indices"
            s = a + b
            m.append((i + 0.5 + (b / s if s > 0 and i1 != i else 0.0), s))
        merged.append(m)
    n = max(len(m) for m in merged)
    merged = [m + [(m[-1][0], 0.0)] * (n - len(m)) for m in merged]
    t = torch.tensor(merged, dtype=torch.float32)
    return t[..., 0].contiguous().cuda(), t[..., 1].contiguous().cuda()


class AreaResize:
    """One (source size -> output size) resize with a CUDA kind. raw_x / raw_y: unpadded taps from
    seg._area_taps / _linear_area_taps; mean / std: three floats in model channel order."""

    def __init__(self, kind, h, w, raw_x, raw_y, mean, std, rgb):
        assert kind in KINDS
        self.kind, self.nh, self.nw, self.rgb = kind, len(raw_y), len(raw_x), rgb
        self.mean, self.std = [float(v) for v in mean], [float(v) for v in std]
        self.ext = cuda_ext()
        if kind == "tex_linear":
            self.x, self.y = merge_taps(raw_x), merge_taps(raw_y)
            m = max(self.x[0].shape[1], self.y[0].shape[1])
            assert self.x[0].shape[1] == self.y[0].shape[1] == m, "both axes need the same fetch count"
            self.w = w
            self.rgba = torch.empty(h, math.ceil(w / 128) * 128, 4, dtype=torch.uint8, device="cuda")  # 512-byte pitch
        else:
            k = max(len(t[0]) for t in raw_x + raw_y)
            pad = lambda taps: (torch.tensor([i + [i[-1]] * (k - len(i)) for i, _ in taps], dtype=torch.int32, device="cuda"),  # noqa: E731
                                torch.tensor([a + [0.0] * (k - len(a)) for _, a in taps], dtype=torch.float32, device="cuda"))
            self.x, self.y = pad(raw_x), pad(raw_y)

    def __call__(self, eye):
        """eye: (H, W, 3) uint8 CUDA, a view into a larger frame allowed. Returns (3, nh, nw) float32."""
        assert eye.dtype == torch.uint8 and eye.stride(2) == 1 and eye.stride(1) == 3
        out = torch.empty(3, self.nh, self.nw, device=eye.device, dtype=torch.float32)
        if self.kind == "tex_linear":
            self.ext.eye_to_rgba(eye, self.rgba)
            self.ext.area_linear(self.rgba, self.w, *self.x, *self.y, out, self.rgb, self.mean, self.std)
        else:
            storage = eye.untyped_storage()
            base = torch.empty(0, dtype=torch.uint8, device=eye.device).set_(storage)
            self.ext.area_taps(eye, base, eye.storage_offset(), *self.x, *self.y, out, self.kind == "tex_point",
                               self.rgb, self.mean, self.std)
        return out
