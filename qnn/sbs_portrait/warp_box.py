"""Bilinear warp and reflect-101 box filter of the alpha post-processing (gpu_post.py uses torch ops
for them) as hand-written kernels, in the order cuda -> texture -> triton:

  cuda        global-memory loads, one output pixel per thread
  tex_point   the same loads through the texture unit (1-D float texture, point fetches)
  tex_linear  hardware bilinear filtering over a pitched 2-D texture of the stacked planes (warp: one
              fetch per channel; box: adjacent equal-weight taps merged into one fetch, 9 -> 5 per
              1-D pass). Planes whose rows are not 32-byte multiples are first copied into a padded
              buffer. The filter fraction is 8-bit fixed point.
  triton      the cuda algorithm in Triton

warp(img, dx, dy): out(y, x) = img(clamp(y + dy), clamp(x + dx)), i.e. GPUPost._sample (grid_sample,
bilinear, border, align_corners=True). box(x, r): GPUPost.box (reflect pad + two 1-D means).
"""
import os
import sys

import torch
import triton
import triton.language as tl

KINDS = ("cuda", "tex_point", "tex_linear", "triton")

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <map>
#include <tuple>

// M: 0 global, 1 point fetches (1-D texture), 2 hardware bilinear (2-D texture)
template <int M, int C, bool HASY>
__global__ void warp_kernel(const float* __restrict__ img, cudaTextureObject_t tex, const float* __restrict__ dx,
                            const float* __restrict__ dy, int step, float* __restrict__ out, int H, int W) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H * W) return;
  int py = i / W, px = i % W;
  float x = fminf(fmaxf(px + dx[(size_t)i * step], 0.f), W - 1.f);
  float y = HASY ? fminf(fmaxf(py + dy[(size_t)i * step], 0.f), H - 1.f) : (float)py;
  if (M == 2) {
    for (int c = 0; c < C; ++c) out[(size_t)c * H * W + i] = tex2D<float>(tex, x + 0.5f, c * H + y + 0.5f);
    return;
  }
  int x0 = (int)x, y0 = (int)y;
  int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
  float lx = x - x0, ly = y - y0;
  for (int c = 0; c < C; ++c) {
    int b = c * H * W;
    int o00 = b + y0 * W + x0, o01 = b + y0 * W + x1, o10 = b + y1 * W + x0, o11 = b + y1 * W + x1;
    float v00, v01, v10, v11;
    if (M == 1) {
      v00 = tex1Dfetch<float>(tex, o00); v01 = tex1Dfetch<float>(tex, o01);
      v10 = tex1Dfetch<float>(tex, o10); v11 = tex1Dfetch<float>(tex, o11);
    } else {
      v00 = img[o00]; v01 = img[o01]; v10 = img[o10]; v11 = img[o11];
    }
    out[(size_t)c * H * W + i] = (1.f - ly) * ((1.f - lx) * v00 + lx * v01) + ly * ((1.f - lx) * v10 + lx * v11);
  }
}

__device__ __forceinline__ int reflect101(int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); }

// one 1-D mean of 2R + 1 taps over the (N, H, W) planes, vertical (V) or horizontal
template <int M, int R, bool V>
__global__ void box_kernel(const float* __restrict__ in, cudaTextureObject_t tex, float* __restrict__ out, int N, int H, int W) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * H * W) return;
  int n = i / (H * W), y = i / W - n * H, x = i % W;
  int len = V ? H : W, pos = V ? y : x, base = n * H * W;
  float acc = 0.f;
  if (M == 2) {
    for (int k = -R; k <= R; k += 2) {
      bool pair = k < R;  // taps k, k + 1 stay adjacent under reflect-101: fetch their midpoint
      float a = reflect101(pos + k, len), b = pair ? reflect101(pos + k + 1, len) : a;
      float p = 0.5f * (a + b) + 0.5f;
      float v = V ? tex2D<float>(tex, x + 0.5f, n * H + p) : tex2D<float>(tex, p, n * H + y + 0.5f);
      acc += pair ? 2.f * v : v;
    }
  } else {
    for (int k = -R; k <= R; ++k) {
      int q = reflect101(pos + k, len);
      int o = base + (V ? q * W + x : y * W + q);
      acc += M == 1 ? tex1Dfetch<float>(tex, o) : in[o];
    }
  }
  out[i] = acc * (1.f / (2 * R + 1));
}

static std::map<std::tuple<const void*, int, int, size_t, int>, cudaTextureObject_t> g_tex;
static std::map<std::pair<int, int>, void*> g_pad;

static cudaTextureObject_t make_tex(const void* ptr, int w, int h, size_t pitch, bool linear) {
  auto key = std::make_tuple(ptr, w, h, pitch, (int)linear);
  auto it = g_tex.find(key);
  if (it != g_tex.end()) return it->second;
  cudaResourceDesc rd = {};
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
  td.readMode = cudaReadModeElementType;
  if (linear) {
    rd.resType = cudaResourceTypePitch2D;
    rd.res.pitch2D.devPtr = (void*)ptr;
    rd.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    rd.res.pitch2D.width = w; rd.res.pitch2D.height = h; rd.res.pitch2D.pitchInBytes = pitch;
    td.filterMode = cudaFilterModeLinear;
  } else {
    rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr = (void*)ptr;
    rd.res.linear.desc = cudaCreateChannelDesc<float>();
    rd.res.linear.sizeInBytes = (size_t)w * h * sizeof(float);
    td.filterMode = cudaFilterModePoint;
  }
  cudaTextureObject_t t = 0;
  TORCH_CHECK(cudaCreateTextureObject(&t, &rd, &td, nullptr) == cudaSuccess, "cudaCreateTextureObject failed");
  g_tex[key] = t;
  return t;
}

// texture over the (C, H, W) float planes stacked into one W x (C H) image
static cudaTextureObject_t tex_planes(torch::Tensor t, int mode) {
  int W = t.size(-1), rows = t.numel() / W;
  const char* p = (const char*)t.data_ptr<float>();
  if (mode == 1) return make_tex(p, W, rows, W * sizeof(float), false);
  size_t pitch = W * sizeof(float);
  if (pitch % 32 == 0 && (uintptr_t)p % 512 == 0) return make_tex(p, W, rows, pitch, true);
  size_t ap = (pitch + 31) / 32 * 32;
  void*& buf = g_pad[{W, rows}];
  if (!buf) TORCH_CHECK(cudaMalloc(&buf, ap * rows) == cudaSuccess, "cudaMalloc failed");
  cudaMemcpy2DAsync(buf, ap, p, pitch, pitch, rows, cudaMemcpyDeviceToDevice, c10::cuda::getCurrentCUDAStream());
  return make_tex(buf, W, rows, ap, true);
}

void warp(torch::Tensor img, torch::Tensor dx, torch::Tensor dy, torch::Tensor out, int64_t mode, bool has_y) {
  int C = img.size(0), H = img.size(1), W = img.size(2), step = dx.stride(1);
  cudaTextureObject_t t = mode ? tex_planes(img, mode) : 0;
  int blocks = (H * W + 255) / 256;
  auto st = c10::cuda::getCurrentCUDAStream();
  const float *p = img.data_ptr<float>(), *px = dx.data_ptr<float>(), *py = has_y ? dy.data_ptr<float>() : nullptr;
  float* o = out.data_ptr<float>();
#define W3(M, C_, Y) warp_kernel<M, C_, Y><<<blocks, 256, 0, st>>>(p, t, px, py, step, o, H, W)
#define W2(M) if (C == 1) { if (has_y) W3(M, 1, true); else W3(M, 1, false); } else { if (has_y) W3(M, 2, true); else W3(M, 2, false); }
  TORCH_CHECK(C == 1 || C == 2, "1 or 2 channels");
  if (mode == 0) { W2(0) } else if (mode == 1) { W2(1) } else { W2(2) }
#undef W2
#undef W3
}

void box(torch::Tensor in, torch::Tensor tmp, torch::Tensor out, int64_t r, int64_t mode) {
  int N = in.size(0), H = in.size(1), W = in.size(2), blocks = (N * H * W + 255) / 256;
  auto st = c10::cuda::getCurrentCUDAStream();
  const float* p = in.data_ptr<float>();
  float *t = tmp.data_ptr<float>(), *o = out.data_ptr<float>();
  cudaTextureObject_t t0 = mode ? tex_planes(in, mode) : 0;
#define B(M, R) box_kernel<M, R, true><<<blocks, 256, 0, st>>>(p, t0, t, N, H, W); { \
    cudaTextureObject_t t1 = M ? tex_planes(tmp, mode) : 0; box_kernel<M, R, false><<<blocks, 256, 0, st>>>(t, t1, o, N, H, W); }
#define BR(R) case R: if (mode == 0) { B(0, R) } else if (mode == 1) { B(1, R) } else { B(2, R) } break;
  switch (r) { BR(1) BR(2) BR(3) BR(4) BR(5) BR(6) default: TORCH_CHECK(false, "unsupported radius ", r); }
#undef BR
#undef B
}
"""
_CPP_SRC = r"""
#include <torch/extension.h>
void warp(torch::Tensor img, torch::Tensor dx, torch::Tensor dy, torch::Tensor out, int64_t mode, bool has_y);
void box(torch::Tensor in, torch::Tensor tmp, torch::Tensor out, int64_t r, int64_t mode);
"""
_ext = None
MODE = {"cuda": 0, "tex_point": 1, "tex_linear": 2}


def load_ext(name, cpp, cuda, functions):
    from torch.utils.cpp_extension import load_inline

    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    os.environ["PATH"] = os.pathsep.join([os.path.dirname(sys.executable),
                                          os.path.join(os.environ["CUDA_HOME"], "bin"), os.environ["PATH"]])
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    return load_inline(name, cpp, cuda_sources=cuda, functions=functions, extra_cuda_cflags=["-O3"], verbose=False)


def cuda_ext():
    global _ext
    if _ext is None:
        _ext = load_ext("warp_box_ext", _CPP_SRC, _CUDA_SRC, ["warp", "box"])
    return _ext


@triton.jit
def _warp_kernel(IMG, DX, DY, STEP, OUT, H, W, C: tl.constexpr, HASY: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < H * W
    py = offs // W
    px = offs % W
    x = px.to(tl.float32) + tl.load(DX + offs * STEP, mask=inb, other=0.0)
    x = tl.minimum(tl.maximum(x, 0.0), (W - 1) * 1.0)
    y = py.to(tl.float32)
    if HASY:
        y = tl.minimum(tl.maximum(y + tl.load(DY + offs * STEP, mask=inb, other=0.0), 0.0), (H - 1) * 1.0)
    x0 = x.to(tl.int32)
    y0 = y.to(tl.int32)
    x1 = tl.minimum(x0 + 1, W - 1)
    y1 = tl.minimum(y0 + 1, H - 1)
    lx = x - x0.to(tl.float32)
    ly = y - y0.to(tl.float32)
    for c in tl.static_range(C):
        b = IMG + c * H * W
        v00 = tl.load(b + y0 * W + x0, mask=inb, other=0.0)
        v01 = tl.load(b + y0 * W + x1, mask=inb, other=0.0)
        v10 = tl.load(b + y1 * W + x0, mask=inb, other=0.0)
        v11 = tl.load(b + y1 * W + x1, mask=inb, other=0.0)
        v = (1.0 - ly) * ((1.0 - lx) * v00 + lx * v01) + ly * ((1.0 - lx) * v10 + lx * v11)
        tl.store(OUT + c * H * W + offs, v, mask=inb)


@triton.jit
def _box_kernel(IN, OUT, N, H, W, R: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < N * H * W
    n = offs // (H * W)
    y = offs // W - n * H
    x = offs % W
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(2 * R + 1):
        if V:
            q = y + (k - R)
            q = tl.where(q < 0, -q, q)
            q = tl.where(q >= H, 2 * H - 2 - q, q)
            o = n * H * W + q * W + x
        else:
            q = x + (k - R)
            q = tl.where(q < 0, -q, q)
            q = tl.where(q >= W, 2 * W - 2 - q, q)
            o = n * H * W + y * W + q
        acc += tl.load(IN + o, mask=inb, other=0.0)
    tl.store(OUT + offs, acc * (1.0 / (2 * R + 1)), mask=inb)


def warp(kind, img, dx, dy=None):
    """img (C, H, W) float32 contiguous, C in {1, 2}; dx, dy (H, W) float32, any element stride
    (flow[..., 0] / flow[..., 1] views work). Returns (C, H, W)."""
    out = torch.empty_like(img)
    assert img.is_contiguous() and (dy is None or dy.stride() == dx.stride())
    if kind == "triton":
        C, H, W = img.shape
        _warp_kernel[(triton.cdiv(H * W, 256),)](img, dx, dx if dy is None else dy, dx.stride(1), out, H, W,
                                                  C=C, HASY=dy is not None, BLOCK=256)
    else:
        cuda_ext().warp(img, dx, dx if dy is None else dy, out, MODE[kind], dy is not None)
    return out


def box(kind, x, r):
    """(2 r + 1)^2 mean with reflect-101 borders of each plane of x (N, H, W) float32 contiguous."""
    assert x.is_contiguous()
    tmp, out = torch.empty_like(x), torch.empty_like(x)
    if kind == "triton":
        N, H, W = x.shape
        grid = (triton.cdiv(N * H * W, 256),)
        _box_kernel[grid](x, tmp, N, H, W, R=r, V=True, BLOCK=256)
        _box_kernel[grid](tmp, out, N, H, W, R=r, V=False, BLOCK=256)
    else:
        cuda_ext().box(x, tmp, out, r, MODE[kind])
    return out
