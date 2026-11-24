"""SBS compositing (alpha upsample + virtual-background blend) with a picture-in-picture overlay.

PiP: the local SBS image (the input frame itself, already on the GPU) or a static SBS 3D icon
(RGBA), each eye scaled to (H/8, W/8) and placed in the top-right corner of the same eye, shifted
by +-d/2 so it is seen at disparity d (in front of the person: no depth conflict). Reference
definition (two steps, = pipeline.compose + cv2): composite -> uint8; overlay eye resized with
OpenCV INTER_AREA -> uint8; out = trunc(p * a_p + comp * (1 - a_p)), a_p = icon alpha / 255 (1 for
the frame).

Implementations (same inputs: frame (H, 2W, 3) u8, background (H, 2W, 3) u8, alphas (2, h, w) f32):
  torch        GPUPost.compose (torch ops, float intermediates) + PiP from two small matmuls with
               OpenCV's area weight matrices (exact INTER_AREA up to rounding ties)
  triton       one fused compose kernel (bilinear alpha + blend + truncation, u8 in / out) and one PiP
               kernel over the two PiP rectangles only (OpenCV area taps)
  cuda_fused   one CUDA kernel for everything: 4 pixels (12 bytes) per thread with 32-bit loads /
               stores, per-pixel PiP branch with OpenCV area taps, global-memory loads
  cuda         the same compose kernel without the branch + a PiP kernel over the two rectangles
               only (one thread per PiP pixel: no load imbalance)
  cuda_tex     the two CUDA kernels with texture sampling: alpha maps as float textures (hardware bilinear =
               align_corners=False upsample), overlay as RGBA8 textures read as normalized float with
               linear filtering; the area filter is n x n bilinear fetches at texel junctions (n =
               round(scale / 2): for the 8x case 16 fetches = the exact 8x8 box). Frame sources are
               first converted BGR -> RGBA into 512-byte-aligned per-eye planes (textures need 1, 2
               or 4 channels; separate eye planes keep the filter from bleeding across eyes)
  cuda_tex1    as cuda_tex with a single bilinear fetch per PiP pixel (2x2 of each 8x8 box: aliased)
Texture weights are 8-bit fixed point (hardware), so cuda_tex* differ from the reference by up to
a few gray levels; the others match it up to float rounding / rounding ties (+-1).
"""
import math
import os

import cv2
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

KINDS = ("torch", "triton", "cuda_fused", "cuda", "cuda_tex", "cuda_tex1")


# --------------------------------------------------------------------------- layout / reference
def pip_layout(H, W, disparity=0.0, margin=None):
    """PiP rectangle per eye: size (H // 8, W // 8), top-right corner with a margin, the left eye
    shifted right and the right eye left by disparity / 2 (x_left = x_right + d)."""
    ph, pw = H // 8, W // 8
    m = int(round(H / 54.0)) if margin is None else margin
    x = W - pw - m
    d = int(round(disparity / 2.0))
    x0 = [min(max(x + d, 0), W - pw), min(max(x - d, 0), W - pw)]
    return {"ph": ph, "pw": pw, "y0": m, "x0": x0}


def area_matrix(src, dst):
    """Dense OpenCV INTER_AREA weights (dst, src) float32 (cv::computeResizeAreaTab)."""
    from seg import _area_taps

    M = np.zeros((dst, src), np.float32)
    for d, (idx, w) in enumerate(_area_taps(src, dst)):
        M[d, idx] = w
    return M


def reference(frame, bg, aL, aR, src, lay):
    """CPU reference (numpy / OpenCV). frame / bg (H, 2W, 3) u8; src (Hs, 2Ws, 3 or 4) u8 SBS."""
    H, W2 = frame.shape[:2]
    W = W2 // 2
    out = np.empty_like(frame)
    for e, a in enumerate((aL, aR)):
        A = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        sl = slice(e * W, (e + 1) * W)
        out[:, sl] = (frame[:, sl].astype(np.float32) * A + bg[:, sl].astype(np.float32) * (1.0 - A)).clip(0, 255).astype(np.uint8)
    return out if src is None else apply_pip_cpu(out, src, lay)


def apply_pip_cpu(out, src, lay):
    """PiP on a CPU composite (in place): each eye of src resized with INTER_AREA, alpha-blended (RGBA)
    or copied (BGR) into its rectangle."""
    W = out.shape[1] // 2
    Ws = src.shape[1] // 2
    ph, pw, y0 = lay["ph"], lay["pw"], lay["y0"]
    for e in range(2):
        p = cv2.resize(np.ascontiguousarray(src[:, e * Ws:(e + 1) * Ws]), (pw, ph), interpolation=cv2.INTER_AREA)
        x0 = e * W + lay["x0"][e]
        reg = out[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
        if p.shape[2] == 4:
            ap = p[..., 3:4].astype(np.float32) / 255.0
            out[y0:y0 + ph, x0:x0 + pw] = (p[..., :3].astype(np.float32) * ap + reg * (1.0 - ap)).clip(0, 255).astype(np.uint8)
        else:
            out[y0:y0 + ph, x0:x0 + pw] = p
    return out


# --------------------------------------------------------------------------- triton
@triton.jit
def _compose_kernel(FR, BG, A, OUT, H, W, h, w, SY, SX, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    W2 = 2 * W
    inb = offs < H * W2
    y = offs // W2
    x2 = offs % W2
    e = (x2 >= W).to(tl.int32)
    x = x2 - e * W
    fy = tl.maximum((y.to(tl.float32) + 0.5) * SY - 0.5, 0.0)
    fx = tl.maximum((x.to(tl.float32) + 0.5) * SX - 0.5, 0.0)
    y0 = fy.to(tl.int32)
    x0 = fx.to(tl.int32)
    y1 = tl.minimum(y0 + 1, h - 1)
    x1 = tl.minimum(x0 + 1, w - 1)
    ly = fy - y0.to(tl.float32)
    lx = fx - x0.to(tl.float32)
    base = A + e * h * w
    a00 = tl.load(base + y0 * w + x0, mask=inb, other=0.0)
    a01 = tl.load(base + y0 * w + x1, mask=inb, other=0.0)
    a10 = tl.load(base + y1 * w + x0, mask=inb, other=0.0)
    a11 = tl.load(base + y1 * w + x1, mask=inb, other=0.0)
    a = (1.0 - ly) * ((1.0 - lx) * a00 + lx * a01) + ly * ((1.0 - lx) * a10 + lx * a11)
    for c in tl.static_range(3):
        f = tl.load(FR + offs * 3 + c, mask=inb, other=0).to(tl.float32)
        b = tl.load(BG + offs * 3 + c, mask=inb, other=0).to(tl.float32)
        v = tl.minimum(tl.maximum(f * a + b * (1.0 - a), 0.0), 255.0)
        tl.store(OUT + offs * 3 + c, v.to(tl.uint8), mask=inb)


@triton.jit
def _pip_kernel(SRC, SROW, OUT, W, IX, WX, IY, WY, X00, X01, Y0, XOFF1, PH, PW,
                K: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n = PH * PW
    inb = offs < 2 * n
    e = (offs >= n).to(tl.int32)
    r = offs - e * n
    py = r // PW
    px = r % PW
    xoff = e * XOFF1
    acc0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK], dtype=tl.float32)
    for ky in tl.static_range(K):
        sy = tl.load(IY + py * K + ky, mask=inb, other=0)
        wy = tl.load(WY + py * K + ky, mask=inb, other=0.0)
        r0 = tl.zeros([BLOCK], dtype=tl.float32)
        r1 = tl.zeros([BLOCK], dtype=tl.float32)
        r2 = tl.zeros([BLOCK], dtype=tl.float32)
        r3 = tl.zeros([BLOCK], dtype=tl.float32)
        for kx in tl.static_range(K):
            sx = tl.load(IX + px * K + kx, mask=inb, other=0)
            wx = tl.load(WX + px * K + kx, mask=inb, other=0.0)
            p = SRC + sy * SROW + (xoff + sx) * C
            r0 += wx * tl.load(p, mask=inb, other=0).to(tl.float32)
            r1 += wx * tl.load(p + 1, mask=inb, other=0).to(tl.float32)
            r2 += wx * tl.load(p + 2, mask=inb, other=0).to(tl.float32)
            if C == 4:
                r3 += wx * tl.load(p + 3, mask=inb, other=0).to(tl.float32)
        acc0 += wy * r0
        acc1 += wy * r1
        acc2 += wy * r2
        acc3 += wy * r3
    x0 = tl.where(e == 0, X00, W + X01)
    o = OUT + ((Y0 + py) * 2 * W + x0 + px) * 3
    v0 = tl.minimum(tl.maximum(libdevice.rint(acc0), 0.0), 255.0)
    v1 = tl.minimum(tl.maximum(libdevice.rint(acc1), 0.0), 255.0)
    v2 = tl.minimum(tl.maximum(libdevice.rint(acc2), 0.0), 255.0)
    if C == 4:
        ap = tl.minimum(tl.maximum(libdevice.rint(acc3), 0.0), 255.0) / 255.0
        v0 = tl.minimum(tl.maximum(v0 * ap + tl.load(o, mask=inb, other=0).to(tl.float32) * (1.0 - ap), 0.0), 255.0)
        v1 = tl.minimum(tl.maximum(v1 * ap + tl.load(o + 1, mask=inb, other=0).to(tl.float32) * (1.0 - ap), 0.0), 255.0)
        v2 = tl.minimum(tl.maximum(v2 * ap + tl.load(o + 2, mask=inb, other=0).to(tl.float32) * (1.0 - ap), 0.0), 255.0)
    tl.store(o, v0.to(tl.uint8), mask=inb)
    tl.store(o + 1, v1.to(tl.uint8), mask=inb)
    tl.store(o + 2, v2.to(tl.uint8), mask=inb)


# --------------------------------------------------------------------------- CUDA (global / texture)
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <map>
#include <tuple>

struct PipArgs {
  int on, ph, pw, y0, x0[2], has_alpha;
  // global variant
  const unsigned char* src; int src_row, src_c, xoff1, K;
  const int* ix; const float* wx; const int* iy; const float* wy;
  // texture variant
  cudaTextureObject_t tex[2]; float fy, fx; int ny, nx;
};

__device__ __forceinline__ float alpha_global(const float* a, int h, int w, float sy, float sx, int y, int x) {
  float fy = fmaxf(__fsub_rn(__fmul_rn(y + 0.5f, sy), 0.5f), 0.f);
  float fx = fmaxf(__fsub_rn(__fmul_rn(x + 0.5f, sx), 0.5f), 0.f);
  int y0 = (int)fy, x0 = (int)fx;
  int y1 = min(y0 + 1, h - 1), x1 = min(x0 + 1, w - 1);
  float ly = fy - y0, lx = fx - x0;
  float t = __fadd_rn(__fmul_rn(1.f - lx, a[y0 * w + x0]), __fmul_rn(lx, a[y0 * w + x1]));
  float b = __fadd_rn(__fmul_rn(1.f - lx, a[y1 * w + x0]), __fmul_rn(lx, a[y1 * w + x1]));
  return __fadd_rn(__fmul_rn(1.f - ly, t), __fmul_rn(ly, b));
}

__device__ __forceinline__ unsigned char blend_u8(float f, float b, float a) {
  float v = __fadd_rn(__fmul_rn(f, a), __fmul_rn(b, __fsub_rn(1.f, a)));
  return (unsigned char)fminf(fmaxf(v, 0.f), 255.f);
}

template <bool TEX>
__device__ __forceinline__ void pip_pixel(const PipArgs& p, int e, int px, int py, unsigned char* c) {
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (TEX) {
    float bx = px * p.fx, by = py * p.fy;
    for (int ky = 0; ky < p.ny; ++ky) {
      float u_y = by + (ky + 0.5f) * p.fy / p.ny;
      for (int kx = 0; kx < p.nx; ++kx) {
        float4 t = tex2D<float4>(p.tex[e], bx + (kx + 0.5f) * p.fx / p.nx, u_y);
        v[0] += t.x; v[1] += t.y; v[2] += t.z; v[3] += t.w;
      }
    }
    float s = 255.f / (p.ny * p.nx);
    for (int k = 0; k < 4; ++k) v[k] = fminf(fmaxf(rintf(v[k] * s), 0.f), 255.f);  // planes hold B,G,R,A
  } else {
    int xo = e * p.xoff1;
    for (int ky = 0; ky < p.K; ++ky) {
      int sy_ = p.iy[py * p.K + ky]; float wy = p.wy[py * p.K + ky];
      float r[4] = {0.f, 0.f, 0.f, 0.f};
      for (int kx = 0; kx < p.K; ++kx) {
        int sx_ = p.ix[px * p.K + kx]; float wx = p.wx[px * p.K + kx];
        const unsigned char* s = p.src + (size_t)sy_ * p.src_row + (size_t)(xo + sx_) * p.src_c;
        for (int k = 0; k < p.src_c; ++k) r[k] += wx * s[k];
      }
      for (int k = 0; k < 4; ++k) v[k] += wy * r[k];
    }
    for (int k = 0; k < 4; ++k) v[k] = fminf(fmaxf(rintf(v[k]), 0.f), 255.f);
  }
  if (p.has_alpha) {
    float ap = v[3] / 255.f;
    for (int k = 0; k < 3; ++k) c[k] = blend_u8(v[k], c[k], ap);
  } else {
    for (int k = 0; k < 3; ++k) c[k] = (unsigned char)v[k];
  }
}

// 4 consecutive pixels (12 bytes, 4-byte aligned: W even) per thread: three 32-bit loads from the
// frame and the background, three 32-bit stores; the PiP pixels take the overlay branch.
template <bool TEX>
__global__ void compose_pip_kernel(const unsigned char* __restrict__ fr, const unsigned char* __restrict__ bg,
                                   const float* __restrict__ A, cudaTextureObject_t ta0, cudaTextureObject_t ta1,
                                   unsigned char* __restrict__ out, int H, int W, int h, int w, float sy, float sx,
                                   PipArgs p) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;  // quad index in the row
  int y = blockIdx.y;
  if (q * 4 >= 2 * W) return;
  size_t o = ((size_t)y * 2 * W + q * 4) * 3;
  uint32_t f32[3], b32[3], r32[3];
  const uint32_t* fp = reinterpret_cast<const uint32_t*>(fr + o);
  const uint32_t* bp = reinterpret_cast<const uint32_t*>(bg + o);
  for (int k = 0; k < 3; ++k) { f32[k] = __ldg(fp + k); b32[k] = __ldg(bp + k); }
  const unsigned char* f8 = reinterpret_cast<const unsigned char*>(f32);
  const unsigned char* b8 = reinterpret_cast<const unsigned char*>(b32);
  unsigned char* r8 = reinterpret_cast<unsigned char*>(r32);
  for (int i = 0; i < 4; ++i) {
    int x2 = q * 4 + i, e = x2 >= W, x = x2 - e * W;
    float a;
    if (TEX) a = tex2D<float>(e ? ta1 : ta0, (x + 0.5f) * sx, (y + 0.5f) * sy);
    else a = alpha_global(A + e * h * w, h, w, sy, sx, y, x);
    unsigned char* c = r8 + 3 * i;
    for (int k = 0; k < 3; ++k) c[k] = blend_u8(f8[3 * i + k], b8[3 * i + k], a);
    if (p.on == 1) {  // fused variant: per-pixel PiP branch (load imbalance: PiP pixels cost ~64x)
      int px = x - p.x0[e], py = y - p.y0;
      if (px >= 0 && px < p.pw && py >= 0 && py < p.ph) pip_pixel<TEX>(p, e, px, py, c);
    }
  }
  uint32_t* op = reinterpret_cast<uint32_t*>(out + o);
  for (int k = 0; k < 3; ++k) op[k] = r32[k];
}

// PiP only: one thread per PiP pixel of both eyes, reads / writes the composite in place
template <bool TEX>
__global__ void pip_kernel(unsigned char* __restrict__ out, int W, PipArgs p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, n = p.ph * p.pw;
  if (i >= 2 * n) return;
  int e = i >= n, r = i - e * n, py = r / p.pw, px = r % p.pw;
  unsigned char* o = out + ((size_t)(p.y0 + py) * 2 * W + e * W + p.x0[e] + px) * 3;
  unsigned char c[3] = {o[0], o[1], o[2]};
  pip_pixel<TEX>(p, e, px, py, c);
  o[0] = c[0]; o[1] = c[1]; o[2] = c[2];
}

// BGR (H, 2W, 3) -> per-eye BGRA planes (2, H, Wp, 4), A = 255
__global__ void to_rgba_kernel(const unsigned char* __restrict__ fr, uchar4* __restrict__ dst, int H, int W, int Wp) {
  int x2 = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x2 >= 2 * W) return;
  int e = x2 >= W, x = x2 - e * W;
  const unsigned char* s = fr + ((size_t)y * 2 * W + x2) * 3;
  dst[((size_t)e * H + y) * Wp + x] = make_uchar4(s[0], s[1], s[2], 255);
}

static std::map<std::tuple<void*, int, int, size_t, int>, cudaTextureObject_t> g_tex;

static cudaTextureObject_t tex_for(void* ptr, int w, int h, size_t pitch, bool rgba) {
  auto key = std::make_tuple(ptr, w, h, pitch, (int)rgba);
  auto it = g_tex.find(key);
  if (it != g_tex.end()) return it->second;
  cudaResourceDesc rd = {};
  rd.resType = cudaResourceTypePitch2D;
  rd.res.pitch2D.devPtr = ptr;
  rd.res.pitch2D.desc = rgba ? cudaCreateChannelDesc<uchar4>() : cudaCreateChannelDesc<float>();
  rd.res.pitch2D.width = w; rd.res.pitch2D.height = h; rd.res.pitch2D.pitchInBytes = pitch;
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
  td.filterMode = cudaFilterModeLinear;
  td.readMode = rgba ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
  td.normalizedCoords = 0;
  cudaTextureObject_t t = 0;
  TORCH_CHECK(cudaCreateTextureObject(&t, &rd, &td, nullptr) == cudaSuccess, "cudaCreateTextureObject failed");
  g_tex[key] = t;
  return t;
}

void to_rgba(torch::Tensor fr, torch::Tensor dst) {
  int H = fr.size(0), W = fr.size(1) / 2, Wp = dst.size(2);
  dim3 blk(256), grd((2 * W + 255) / 256, H);
  to_rgba_kernel<<<grd, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(fr.data_ptr<unsigned char>(),
      (uchar4*)dst.data_ptr<unsigned char>(), H, W, Wp);
}

// mode 0: global, 1: texture. a: (2, h, w) float (global) or (2, h, wp) pitched float (texture).
// pip_i: [on, ph, pw, y0, x00, x01, has_alpha, src_c, xoff1, K, ny, nx]; pip_f: [fy, fx]
void compose_pip(torch::Tensor fr, torch::Tensor bg, torch::Tensor a, int h, int w, torch::Tensor out, int mode,
                 bool fused, std::vector<int64_t> pip_i, std::vector<double> pip_f, torch::Tensor src,
                 torch::Tensor ix, torch::Tensor wx, torch::Tensor iy, torch::Tensor wy) {
  int H = fr.size(0), W = fr.size(1) / 2;
  float sy = (float)h / H, sx = (float)w / W;
  PipArgs p = {};
  p.on = pip_i[0]; p.ph = pip_i[1]; p.pw = pip_i[2]; p.y0 = pip_i[3]; p.x0[0] = pip_i[4]; p.x0[1] = pip_i[5];
  p.has_alpha = pip_i[6]; p.src_c = pip_i[7]; p.xoff1 = pip_i[8]; p.K = pip_i[9]; p.ny = pip_i[10]; p.nx = pip_i[11];
  p.fy = pip_f[0]; p.fx = pip_f[1];
  cudaTextureObject_t ta0 = 0, ta1 = 0;
  if (mode == 1) {
    size_t pitch = a.stride(1) * sizeof(float);
    ta0 = tex_for(a[0].data_ptr(), w, h, pitch, false);
    ta1 = tex_for(a[1].data_ptr(), w, h, pitch, false);
    if (p.on) {  // src: (2, Hs, Wp, 4) u8 planes
      size_t sp = src.stride(1);
      int Hs = src.size(1), Ws = pip_i[8];
      p.tex[0] = tex_for(src[0].data_ptr(), Ws, Hs, sp, true);
      p.tex[1] = tex_for(src[1].data_ptr(), Ws, Hs, sp, true);
    }
  } else if (p.on) {
    p.src = src.data_ptr<unsigned char>(); p.src_row = src.stride(0);
    p.ix = ix.data_ptr<int>(); p.wx = wx.data_ptr<float>(); p.iy = iy.data_ptr<int>(); p.wy = wy.data_ptr<float>();
  }
  TORCH_CHECK(W % 2 == 0, "eye width must be even (4-pixel vector access)");
  dim3 blk(128), grd((2 * W / 4 + 127) / 128, H);
  auto st = c10::cuda::getCurrentCUDAStream();
  int on = p.on;
  p.on = (on && fused) ? 1 : 0;
  if (mode == 1)
    compose_pip_kernel<true><<<grd, blk, 0, st>>>(fr.data_ptr<unsigned char>(), bg.data_ptr<unsigned char>(), nullptr,
                                                  ta0, ta1, out.data_ptr<unsigned char>(), H, W, h, w, sy, sx, p);
  else
    compose_pip_kernel<false><<<grd, blk, 0, st>>>(fr.data_ptr<unsigned char>(), bg.data_ptr<unsigned char>(),
                                                   a.data_ptr<float>(), 0, 0, out.data_ptr<unsigned char>(), H, W, h, w, sy, sx, p);
  if (on && !fused) {
    int n = 2 * p.ph * p.pw;
    if (mode == 1) pip_kernel<true><<<(n + 63) / 64, 64, 0, st>>>(out.data_ptr<unsigned char>(), W, p);
    else pip_kernel<false><<<(n + 63) / 64, 64, 0, st>>>(out.data_ptr<unsigned char>(), W, p);
  }
}
"""
_CPP_SRC = r"""
#include <torch/extension.h>
void to_rgba(torch::Tensor fr, torch::Tensor dst);
void compose_pip(torch::Tensor fr, torch::Tensor bg, torch::Tensor a, int h, int w, torch::Tensor out, int mode,
                 bool fused, std::vector<int64_t> pip_i, std::vector<double> pip_f, torch::Tensor src,
                 torch::Tensor ix, torch::Tensor wx, torch::Tensor iy, torch::Tensor wy);
"""
_ext = None


def cuda_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        import sys

        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
        # ninja / nvcc of this interpreter's environment (non-interactive shells do not activate it)
        os.environ["PATH"] = os.pathsep.join([os.path.dirname(sys.executable),
                                              os.path.join(os.environ["CUDA_HOME"], "bin"), os.environ["PATH"]])
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        _ext = load_inline("pip_compose_ext", _CPP_SRC, cuda_sources=_CUDA_SRC, functions=["to_rgba", "compose_pip"],
                           extra_cuda_cflags=["-O3"], verbose=False)
    return _ext


# --------------------------------------------------------------------------- composer
class Composer:
    """Composite + optional PiP for one eye size. src: None, "frame" (the local SBS input frame) or an
    SBS RGBA icon (numpy (Hi, 2Wi, 4) u8, uploaded once)."""

    def __init__(self, H, W, ws, bg_hwc, kind="cuda", src=None, disparity=0.0, device="cuda"):
        assert kind in KINDS
        self.H, self.W, self.kind = H, W, kind
        self.w, self.h = ws
        self.bg = bg_hwc  # (H, 2W, 3) u8 on the GPU
        self.out = torch.empty(H, 2 * W, 3, dtype=torch.uint8, device=device)
        self.lay = pip_layout(H, W, disparity)
        self.apad = None  # pitched alpha planes (texture variants)
        self.src_kind = None if src is None else ("frame" if isinstance(src, str) else "icon")
        if self.src_kind is None:
            return
        if self.src_kind == "icon":
            self.icon = torch.from_numpy(np.ascontiguousarray(src)).to(device)
            Hs, Ws, C = src.shape[0], src.shape[1] // 2, 4
        else:
            Hs, Ws, C = H, W, 3
        self.Hs, self.Ws, self.C = Hs, Ws, C
        ph, pw = self.lay["ph"], self.lay["pw"]
        if kind in ("torch",):
            self.Ry = torch.from_numpy(area_matrix(Hs, ph)).to(device)
            self.Rx = torch.from_numpy(area_matrix(Ws, pw)).to(device)
        if kind in ("triton", "cuda", "cuda_fused"):
            from seg import _area_taps
            tx, ty = _area_taps(Ws, pw), _area_taps(Hs, ph)
            k = max(len(t[0]) for t in tx + ty)
            pad = lambda taps: (torch.tensor([i + [i[-1]] * (k - len(i)) for i, _ in taps], dtype=torch.int32, device=device),  # noqa: E731
                                torch.tensor([a + [0.0] * (k - len(a)) for _, a in taps], dtype=torch.float32, device=device))
            (self.ix, self.wx), (self.iy, self.wy), self.K = pad(tx), pad(ty), k
        if kind.startswith("cuda_tex"):
            wp = -(-Ws // 128) * 128  # eye planes start 512-byte aligned (texture base alignment)
            self.rgba = torch.zeros(2, Hs, wp, 4, dtype=torch.uint8, device=device)
            if self.src_kind == "icon":
                for e in range(2):
                    self.rgba[e, :, :Ws] = self.icon[:, e * Ws:(e + 1) * Ws]
            self.fy, self.fx = Hs / ph, Ws / pw
            n = 1 if kind == "cuda_tex1" else None
            self.ny = n or max(1, int(round(self.fy / 2)))
            self.nx = n or max(1, int(round(self.fx / 2)))

    def __call__(self, frame_t, aL, aR):
        k, lay = self.kind, self.lay
        if k == "torch":
            return self._torch(frame_t, aL, aR)
        if k == "triton":
            a = torch.stack([aL, aR]).contiguous()
            n = self.H * 2 * self.W
            _compose_kernel[(triton.cdiv(n, 1024),)](frame_t, self.bg, a, self.out, self.H, self.W, self.h, self.w,
                                                      self.h / self.H, self.w / self.W, BLOCK=1024)
            if self.src_kind:
                src = self.icon if self.src_kind == "icon" else frame_t
                m = 2 * lay["ph"] * lay["pw"]
                _pip_kernel[(triton.cdiv(m, 64),)](src, src.stride(0), self.out, self.W, self.ix, self.wx, self.iy, self.wy,
                                                     lay["x0"][0], lay["x0"][1], lay["y0"], self.Ws, lay["ph"], lay["pw"],
                                                     K=self.K, C=self.C, BLOCK=64)
            return self.out
        ext = cuda_ext()
        on = self.src_kind is not None
        pi = [int(on), lay["ph"], lay["pw"], lay["y0"], lay["x0"][0], lay["x0"][1], int(self.src_kind == "icon"),
              getattr(self, "C", 3), getattr(self, "Ws", 0), getattr(self, "K", 0), getattr(self, "ny", 1), getattr(self, "nx", 1)]
        pf = [getattr(self, "fy", 1.0), getattr(self, "fx", 1.0)]
        e = torch.empty(0, device=frame_t.device)
        if k in ("cuda", "cuda_fused"):
            a = torch.stack([aL, aR]).contiguous()
            src = (self.icon if self.src_kind == "icon" else frame_t) if on else e
            ext.compose_pip(frame_t, self.bg, a, self.h, self.w, self.out, 0, k == "cuda_fused", pi, pf, src,
                            *(t for t in (getattr(self, "ix", e), getattr(self, "wx", e), getattr(self, "iy", e), getattr(self, "wy", e))))
            return self.out
        # texture: alphas into a pitched buffer (rows 512-byte multiples), frame -> RGBA eye planes
        if self.apad is None:
            wp = -(-self.w // 128) * 128
            self.apad = torch.zeros(2, self.h, wp, dtype=torch.float32, device=frame_t.device)
        self.apad[0, :, :self.w].copy_(aL)
        self.apad[1, :, :self.w].copy_(aR)
        if on and self.src_kind == "frame":
            ext.to_rgba(frame_t, self.rgba)
        ext.compose_pip(frame_t, self.bg, self.apad, self.h, self.w, self.out, 1, False, pi, pf, self.rgba if on else e,
                        e, e, e, e)
        return self.out

    def _torch(self, frame_t, aL, aR):
        import torch.nn.functional as F

        H, W2 = frame_t.shape[:2]
        W = W2 // 2
        img = frame_t.permute(2, 0, 1).reshape(3, H, 2, W).permute(2, 0, 1, 3).float()
        a = F.interpolate(torch.stack([aL, aR])[:, None], size=(H, W), mode="bilinear", align_corners=False)
        bg = self.bg.permute(2, 0, 1).reshape(3, H, 2, W).permute(2, 0, 1, 3).float()
        out = (img * a + bg * (1.0 - a)).clamp(0, 255).to(torch.uint8)
        out = out.permute(1, 2, 0, 3).reshape(3, H, W2).permute(1, 2, 0)
        if self.src_kind is None:
            return out
        src = self.icon if self.src_kind == "icon" else frame_t
        lay = self.lay
        ph, pw, y0 = lay["ph"], lay["pw"], lay["y0"]
        for e in range(2):
            s = src[:, e * self.Ws:(e + 1) * self.Ws].float()  # Hs, Ws, C
            t = (self.Ry @ s.reshape(self.Hs, -1)).reshape(ph, self.Ws, self.C)
            p = torch.einsum("xw,ywc->yxc", self.Rx, t).round().clamp(0, 255)
            x0 = e * W + lay["x0"][e]
            reg = out[y0:y0 + ph, x0:x0 + pw]
            if self.C == 4:
                ap = p[..., 3:] / 255.0
                reg.copy_((p[..., :3] * ap + reg.float() * (1.0 - ap)).clamp(0, 255).to(torch.uint8))
            else:
                reg.copy_(p.to(torch.uint8))
        return out


def make_icon(size=512):
    """Synthetic SBS 3D icon (RGBA, (size, 2 size, 4)): a rounded tile with "3D" text, the text drawn
    with a 6 % horizontal disparity between the eyes so it floats in front of the tile."""
    eyes = []
    for e in range(2):
        im = np.zeros((size, size, 4), np.uint8)
        r = size // 6
        cv2.rectangle(im, (r, 0), (size - r, size - 1), (60, 40, 20, 230), -1)
        cv2.rectangle(im, (0, r), (size - 1, size - r), (60, 40, 20, 230), -1)
        for cx, cy in ((r, r), (size - r - 1, r), (r, size - r - 1), (size - r - 1, size - r - 1)):
            cv2.circle(im, (cx, cy), r, (60, 40, 20, 230), -1)
        d = int(size * 0.03) * (1 if e == 0 else -1)
        cv2.putText(im, "3D", (int(size * 0.17) + d, int(size * 0.68)), cv2.FONT_HERSHEY_DUPLEX, size / 110.0,
                    (255, 220, 120, 255), max(2, size // 40), cv2.LINE_AA)
        eyes.append(im)
    return np.hstack(eyes)
