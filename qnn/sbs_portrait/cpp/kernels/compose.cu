// Fused compositing (bilinear alpha upsample + blend + truncation) and the picture-in-picture
// overlay. Ported from pip_compose.py "cuda" (global) and "cuda_tex" (linear); TEX_POINT is the
// global kernel with its alpha / source loads going through 1-D textures.
#include "kernels.hpp"
#include "tex.cuh"

namespace k {
namespace {

struct Args {
  Pip p;
  cudaTextureObject_t a[2];    // TEX_LINEAR: alpha planes; TEX_POINT: a[0] = both planes (1-D)
  cudaTextureObject_t src[2];  // TEX_LINEAR: BGRA eye planes; TEX_POINT: src[0] = SBS bytes (1-D)
  float fy, fx;                // TEX_LINEAR: PiP source footprint per output pixel
  int ny, nx;
};

// torch's bilinear upsample (align_corners=False) without fused multiply-add
template <int M>
__device__ __forceinline__ float alpha_at(const float* a, cudaTextureObject_t t, int base, int h, int w, float sy,
                                          float sx, int y, int x) {
  float fy = fmaxf(__fsub_rn(__fmul_rn(y + 0.5f, sy), 0.5f), 0.f);
  float fx = fmaxf(__fsub_rn(__fmul_rn(x + 0.5f, sx), 0.5f), 0.f);
  int y0 = (int)fy, x0 = (int)fx;
  int y1 = min(y0 + 1, h - 1), x1 = min(x0 + 1, w - 1);
  float ly = fy - y0, lx = fx - x0;
  auto at = [&](int o) { return M == 1 ? tex1Dfetch<float>(t, base + o) : a[base + o]; };
  float top = __fadd_rn(__fmul_rn(1.f - lx, at(y0 * w + x0)), __fmul_rn(lx, at(y0 * w + x1)));
  float bot = __fadd_rn(__fmul_rn(1.f - lx, at(y1 * w + x0)), __fmul_rn(lx, at(y1 * w + x1)));
  return __fadd_rn(__fmul_rn(1.f - ly, top), __fmul_rn(ly, bot));
}

__device__ __forceinline__ uint8_t blend_u8(float f, float b, float a) {
  float v = __fadd_rn(__fmul_rn(f, a), __fmul_rn(b, __fsub_rn(1.f, a)));
  return (uint8_t)fminf(fmaxf(v, 0.f), 255.f);
}

__global__ void alpha_pad_kernel(const float* __restrict__ a, float* __restrict__ pad, int h, int w, int apitch) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;  // y over both planes
  if (x < w) pad[(size_t)y * apitch + x] = a[(size_t)y * w + x];
}

template <int M>
__global__ void compose_kernel(const uint8_t* __restrict__ fr, const uint8_t* __restrict__ bg, const float* __restrict__ A,
                               uint8_t* __restrict__ out, int H, int W, int h, int w, float sy, float sx, Args g) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;  // quad index in the row: 4 pixels = 12 bytes
  int y = blockIdx.y;
  if (q * 4 >= 2 * W) return;
  size_t o = ((size_t)y * 2 * W + q * 4) * 3;
  uint32_t f32[3], b32[3], r32[3];
  const uint32_t* fp = reinterpret_cast<const uint32_t*>(fr + o);
  const uint32_t* bp = reinterpret_cast<const uint32_t*>(bg + o);
  for (int k = 0; k < 3; ++k) { f32[k] = __ldg(fp + k); b32[k] = __ldg(bp + k); }
  const uint8_t* f8 = reinterpret_cast<const uint8_t*>(f32);
  const uint8_t* b8 = reinterpret_cast<const uint8_t*>(b32);
  uint8_t* r8 = reinterpret_cast<uint8_t*>(r32);
  for (int i = 0; i < 4; ++i) {
    int x2 = q * 4 + i, e = x2 >= W, x = x2 - e * W;
    float a = M == 2 ? tex2D<float>(g.a[e], (x + 0.5f) * sx, (y + 0.5f) * sy)
                     : alpha_at<M>(A, g.a[0], e * h * w, h, w, sy, sx, y, x);
    for (int k = 0; k < 3; ++k) r8[3 * i + k] = blend_u8(f8[3 * i + k], b8[3 * i + k], a);
  }
  uint32_t* op = reinterpret_cast<uint32_t*>(out + o);
  for (int k = 0; k < 3; ++k) op[k] = r32[k];
}

// one thread per PiP pixel of both eyes, reads / writes the composite in place
template <int M>
__global__ void pip_kernel(uint8_t* __restrict__ out, int W, Args g) {
  const Pip& p = g.p;
  int i = blockIdx.x * blockDim.x + threadIdx.x, n = p.ph * p.pw;
  if (i >= 2 * n) return;
  int e = i >= n, r = i - e * n, py = r / p.pw, px = r % p.pw;
  uint8_t* o = out + ((size_t)(p.y0 + py) * 2 * W + e * W + p.x0[e] + px) * 3;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (M == 2) {
    float bx = px * g.fx, by = py * g.fy;
    for (int ky = 0; ky < g.ny; ++ky) {
      float uy = by + (ky + 0.5f) * g.fy / g.ny;
      for (int kx = 0; kx < g.nx; ++kx) {
        float4 t = tex2D<float4>(g.src[e], bx + (kx + 0.5f) * g.fx / g.nx, uy);
        v[0] += t.x; v[1] += t.y; v[2] += t.z; v[3] += t.w;
      }
    }
    float s = 255.f / (g.ny * g.nx);
    for (int k = 0; k < 4; ++k) v[k] = fminf(fmaxf(rintf(v[k] * s), 0.f), 255.f);
  } else {
    int xo = e * p.Ws;
    for (int ky = 0; ky < p.K; ++ky) {
      int sy = p.y.i[py * p.K + ky];
      float wy = p.y.w[py * p.K + ky];
      float row[4] = {0.f, 0.f, 0.f, 0.f};
      for (int kx = 0; kx < p.K; ++kx) {
        int sx = p.x.i[px * p.K + kx];
        float wx = p.x.w[px * p.K + kx];
        int off = sy * p.src_row + (xo + sx) * p.src_c;
        for (int k = 0; k < p.src_c; ++k)
          row[k] += wx * (M == 1 ? tex1Dfetch<unsigned char>(g.src[0], off + k) : p.src[off + k]);
      }
      for (int k = 0; k < 4; ++k) v[k] += wy * row[k];
    }
    for (int k = 0; k < 4; ++k) v[k] = fminf(fmaxf(rintf(v[k]), 0.f), 255.f);
  }
  if (p.has_alpha) {
    float ap = v[3] / 255.f;
    for (int k = 0; k < 3; ++k) o[k] = blend_u8(v[k], o[k], ap);
  } else {
    for (int k = 0; k < 3; ++k) o[k] = (uint8_t)v[k];
  }
}

__global__ void to_rgba_kernel(const uint8_t* __restrict__ src, int row, int c, uchar4* __restrict__ dst, int Hs, int Ws,
                               int wp) {
  int x2 = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x2 >= 2 * Ws) return;
  int e = x2 >= Ws, x = x2 - e * Ws;
  const uint8_t* s = src + (size_t)y * row + (size_t)x2 * c;
  dst[((size_t)e * Hs + y) * wp + x] = make_uchar4(s[0], s[1], s[2], c == 4 ? s[3] : 255);
}

}  // namespace

void compose(Mode mode, const uint8_t* frame, const uint8_t* bg, const float* alpha, float* apad, int apitch,
             uint8_t* out, int H, int W, int h, int w, const Pip& pip, cudaStream_t s) {
  if (W % 2) throw std::runtime_error("eye width must be even (4-pixel vector access)");
  float sy = (float)h / H, sx = (float)w / W;
  Args g = {};
  g.p = pip;
  if (mode == TEX_POINT) {
    g.a[0] = tex_1d<float>(alpha, (size_t)2 * h * w);
    if (pip.on) g.src[0] = tex_1d<unsigned char>(pip.src, (size_t)pip.Hs * pip.src_row);
  } else if (mode == TEX_LINEAR) {
    alpha_pad_kernel<<<dim3((w + 255) / 256, 2 * h), 256, 0, s>>>(alpha, apad, h, w, apitch);
    for (int e = 0; e < 2; ++e)
      g.a[e] = tex_2d<float>(apad + (size_t)e * h * apitch, w, h, (size_t)apitch * 4, cudaReadModeElementType);
    if (pip.on) {
      if (!pip.rgba_ready)
        to_rgba_kernel<<<dim3((2 * pip.Ws + 255) / 256, pip.Hs), 256, 0, s>>>(pip.src, pip.src_row, pip.src_c,
                                                                           (uchar4*)pip.rgba, pip.Hs, pip.Ws, pip.wp);
      for (int e = 0; e < 2; ++e)
        g.src[e] = tex_2d<uchar4>(pip.rgba + (size_t)e * pip.Hs * pip.wp * 4, pip.Ws, pip.Hs, (size_t)pip.wp * 4,
                                  cudaReadModeNormalizedFloat);
      g.fy = (float)pip.Hs / pip.ph;
      g.fx = (float)pip.Ws / pip.pw;
      g.ny = (int)fmaxf(1.f, rintf(g.fy / 2));
      g.nx = (int)fmaxf(1.f, rintf(g.fx / 2));
    }
  }
  dim3 blk(128), grd((2 * W / 4 + 127) / 128, H);
  int n = 2 * pip.ph * pip.pw;
#define RUN(M)                                                                     \
  compose_kernel<M><<<grd, blk, 0, s>>>(frame, bg, alpha, out, H, W, h, w, sy, sx, g); \
  if (pip.on) pip_kernel<M><<<(n + 63) / 64, 64, 0, s>>>(out, W, g)
  if (mode == GLOBAL) { RUN(0); }
  else if (mode == TEX_POINT) { RUN(1); }
  else { RUN(2); }
#undef RUN
  CU(cudaGetLastError());
}

}  // namespace k
