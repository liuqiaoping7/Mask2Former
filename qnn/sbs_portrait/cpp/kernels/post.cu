// Alpha post-processing on the work grid (gpu_post.py). warp / box are ported from warp_box.py
// (global / point / linear); everything else replaces torch ops of GPUPost with one kernel each.
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include "kernels.hpp"
#include "tex.cuh"

namespace k {
namespace {

// one thread per element
#define FOR_EACH_ELEMENT(n)                          \
  int i = blockIdx.x * blockDim.x + threadIdx.x;     \
  if (i >= (n)) return;
#define LAUNCH(kernel, n, s) kernel<<<((n) + 255) / 256, 256, 0, s>>>

__global__ void u8_kernel(const uint8_t* __restrict__ in, float* __restrict__ out, int n) {
  FOR_EACH_ELEMENT(n) out[i] = in[i];
}
__global__ void deinterleave_kernel(const float2* __restrict__ in, float* __restrict__ out, int n) {
  FOR_EACH_ELEMENT(n) out[i] = in[i].x; out[n + i] = in[i].y;
}

// M: 0 global, 1 point fetches (1-D texture), 2 hardware bilinear (2-D texture)
template <int M, int C, bool HASY>
__global__ void warp_kernel(const float* __restrict__ img, cudaTextureObject_t tex, const float* __restrict__ dx,
                            const float* __restrict__ dy, int step, float* __restrict__ out, int H, int W) {
  FOR_EACH_ELEMENT(H * W)
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
  FOR_EACH_ELEMENT(N * H * W)
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

__global__ void validity_kernel(const float* __restrict__ g, const float2* __restrict__ fb, const float* __restrict__ ff,
                                const float* __restrict__ prev_w, uint8_t* __restrict__ valid, float* __restrict__ photo,
                                int n) {
  FOR_EACH_ELEMENT(n)
  float2 b = fb[i];
  float fx = ff[i], fy = ff[n + i];
  float sx = b.x + fx, sy = b.y + fy;
  float mag = (b.x * b.x + b.y * b.y) + (fx * fx + fy * fy);
  bool ok = sx * sx + sy * sy < 0.01f * mag + 0.5f;
  float ph = fabsf(rintf(prev_w[i]) - g[i]);  // cv2.remap on uint8 rounds
  photo[i] = ph;
  valid[i] = ok && ph < 20.f;
}

__global__ void or_above_kernel(const uint8_t* h, const float* p, float thr, uint8_t* out, int n) {
  FOR_EACH_ELEMENT(n) out[i] = h[i] || p[i] > thr;
}

// distance from each pixel to the nearest foreground pixel of its row: two scans per row
__global__ void row_distance_kernel(const uint8_t* __restrict__ m, int* __restrict__ dh, int H, int W) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= H) return;
  const uint8_t* row = m + (size_t)y * W;
  int* d = dh + (size_t)y * W;
  int last = -4 * W;
  for (int x = 0; x < W; ++x) {
    if (row[x]) last = x;
    d[x] = x - last;
  }
  int next = 5 * W;
  for (int x = W - 1; x >= 0; --x) {
    if (row[x]) next = x;
    d[x] = min(d[x], next - x);
  }
}
__global__ void ellipse_kernel(const int* __restrict__ dh, const int* __restrict__ hw, int r, uint8_t* __restrict__ out,
                               int H, int W) {
  FOR_EACH_ELEMENT(H * W)
  int y = i / W, x = i % W;
  uint8_t on = 0;
  for (int dy = -r; dy <= r && !on; ++dy) {
    int q = y + dy;
    if (q >= 0 && q < H) on = dh[q * W + x] <= hw[dy + r];
  }
  out[i] = on;
}

// separable minimum filter; samples outside the image are ignored
template <bool V>
__global__ void min_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int H, int W, int r) {
  FOR_EACH_ELEMENT(H * W)
  int y = i / W, x = i % W;
  uint8_t v = 1;
  for (int k = -r; k <= r; ++k) {
    int q = (V ? y : x) + k;
    if (q >= 0 && q < (V ? H : W)) v &= in[V ? q * W + x : y * W + q];
  }
  out[i] = v;
}

__global__ void fill_kernel(const float* d, const uint8_t* band, float dp, float dd, float* out, int n) {
  FOR_EACH_ELEMENT(n) out[i] = isnan(d[i]) ? (band[i] ? dp : dd) : d[i];
}
__global__ void soft_kernel(const float* p, float lo, float hi, float* out, int n) {
  FOR_EACH_ELEMENT(n)
  float t = fminf(fmaxf((p[i] - lo) / (hi - lo), 0.f), 1.f);
  out[i] = t * t * (3.f - 2.f * t);
}
__global__ void guided_stack_kernel(const float* __restrict__ g255, const float* __restrict__ p, float* __restrict__ g,
                                    float* __restrict__ s4, int n) {
  FOR_EACH_ELEMENT(n)
  float gv = g255[i] / 255.f, pv = p[i];
  g[i] = gv;
  s4[i] = gv;
  s4[n + i] = pv;
  s4[2 * n + i] = gv * pv;
  s4[3 * n + i] = gv * gv;
}
__global__ void guided_ab_kernel(const float* __restrict__ b4, float eps, float* __restrict__ ab, int n) {
  FOR_EACH_ELEMENT(n)
  float mi = b4[i], mp = b4[n + i], cov = b4[2 * n + i] - mi * mp, var = b4[3 * n + i] - mi * mi;
  float a = cov / (var + eps);
  ab[i] = a;
  ab[n + i] = mp - a * mi;
}
__global__ void guided_out_kernel(const float* __restrict__ b2, const float* __restrict__ g, float* __restrict__ q, int n) {
  FOR_EACH_ELEMENT(n) q[i] = fminf(fmaxf(b2[i] * g[i] + b2[n + i], 0.f), 1.f);
}
__global__ void ema_kernel(const float* __restrict__ p, const float* __restrict__ prev_w, const uint8_t* __restrict__ valid,
                           const float* __restrict__ photo, float lam, float sigma, float gate, float* __restrict__ out,
                           int n) {
  FOR_EACH_ELEMENT(n)
  float w = lam * (valid[i] ? 1.f : 0.f) * expf(-photo[i] / sigma);
  if (gate != 0.f) w *= fminf(fmaxf(1.f - fabsf(p[i] - prev_w[i]) / gate, 0.f), 1.f);
  out[i] = w * prev_w[i] + (1.f - w) * p[i];
}
__global__ void negate_kernel(const float* in, float* out, int n) {
  FOR_EACH_ELEMENT(n) out[i] = -in[i];
}
__global__ void average_kernel(const float* a, const float* b, float* out, int n) {
  FOR_EACH_ELEMENT(n) out[i] = 0.5f * (a[i] + b[i]);
}

// W x rows float planes as a texture; TEX_LINEAR needs 32-byte pitch and 512-byte base alignment
cudaTextureObject_t planes_tex(Mode mode, const float* p, int W, int rows, float* pad, cudaStream_t s) {
  if (mode == TEX_POINT) return tex_1d<float>(p, (size_t)W * rows);
  size_t pitch = (size_t)W * 4;
  if (pitch % 32 == 0 && (uintptr_t)p % 512 == 0) return tex_2d<float>(p, W, rows, pitch, cudaReadModeElementType);
  size_t ap = (pitch + 31) / 32 * 32;
  CU(cudaMemcpy2DAsync(pad, ap, p, pitch, pitch, rows, cudaMemcpyDeviceToDevice, s));
  return tex_2d<float>(pad, W, rows, ap, cudaReadModeElementType);
}

struct NotNan {
  __host__ __device__ bool operator()(float v) const { return v == v; }
};

struct NonZero {
  __host__ __device__ bool operator()(uint8_t v) const { return v != 0; }
};
struct IsNan {
  __host__ __device__ bool operator()(float v) const { return v != v; }
};

}  // namespace

size_t pitched_bytes(int W, int rows) { return (size_t)((W * 4 + 31) / 32 * 32) * rows; }

void u8_to_f32(const uint8_t* in, float* out, int n, cudaStream_t s) { LAUNCH(u8_kernel, n, s)(in, out, n); }
void deinterleave2(const float* hwc, float* chw, int n, cudaStream_t s) {
  LAUNCH(deinterleave_kernel, n, s)((const float2*)hwc, chw, n);
}

void warp(Mode mode, const float* img, int C, int H, int W, const float* dx, const float* dy, int step, float* out,
          float* pad, cudaStream_t s) {
  if (C != 1 && C != 2) throw std::runtime_error("1 or 2 channels");
  cudaTextureObject_t t = mode ? planes_tex(mode, img, W, C * H, pad, s) : 0;
  bool has_y = dy != nullptr;
  const float* py = has_y ? dy : dx;
#define W3(M, C_, Y) LAUNCH((warp_kernel<M, C_, Y>), H * W, s)(img, t, dx, py, step, out, H, W)
#define W2(M)                                              \
  if (C == 1) { if (has_y) W3(M, 1, true); else W3(M, 1, false); } \
  else { if (has_y) W3(M, 2, true); else W3(M, 2, false); }
  if (mode == GLOBAL) { W2(0) } else if (mode == TEX_POINT) { W2(1) } else { W2(2) }
#undef W2
#undef W3
  CU(cudaGetLastError());
}

void box(Mode mode, const float* in, float* tmp, float* out, int N, int H, int W, int r, float* pad, cudaStream_t s) {
  int n = N * H * W;
  cudaTextureObject_t t0 = mode ? planes_tex(mode, in, W, N * H, pad, s) : 0;
#define B(M, R)                                                                     \
  LAUNCH((box_kernel<M, R, true>), n, s)(in, t0, tmp, N, H, W);                     \
  {                                                                                 \
    cudaTextureObject_t t1 = M ? planes_tex(mode, tmp, W, N * H, pad, s) : 0;       \
    LAUNCH((box_kernel<M, R, false>), n, s)(tmp, t1, out, N, H, W);                 \
  }
#define BR(R)                                \
  case R:                                    \
    if (mode == GLOBAL) { B(0, R) }          \
    else if (mode == TEX_POINT) { B(1, R) }  \
    else { B(2, R) }                         \
    break;
  switch (r) {
    BR(1) BR(2) BR(3) BR(4) BR(5) BR(6)
    default: throw std::runtime_error("unsupported box radius");
  }
#undef BR
#undef B
  CU(cudaGetLastError());
}

void flow_validity(const float* g_cur, const float* fb_hwc, const float* ff_at, const float* prev_w, uint8_t* valid,
                   float* photo, int n, cudaStream_t s) {
  LAUNCH(validity_kernel, n, s)(g_cur, (const float2*)fb_hwc, ff_at, prev_w, valid, photo, n);
}
void or_above(const uint8_t* h, const float* p, float thr, uint8_t* out, int n, cudaStream_t s) {
  LAUNCH(or_above_kernel, n, s)(h, p, thr, out, n);
}
void dilate_ellipse(const uint8_t* m, int* dh, const int* half_w, int r, uint8_t* out, int H, int W, cudaStream_t s) {
  row_distance_kernel<<<(H + 31) / 32, 32, 0, s>>>(m, dh, H, W);
  LAUNCH(ellipse_kernel, H * W, s)(dh, half_w, r, out, H, W);
}
void erode(const uint8_t* m, uint8_t* tmp, uint8_t* out, int H, int W, int ksize, cudaStream_t s) {
  LAUNCH(min_kernel<true>, H * W, s)(m, tmp, H, W, ksize / 2);
  LAUNCH(min_kernel<false>, H * W, s)(tmp, out, H, W, ksize / 2);
}

bool nanmedian(const float* d, const uint8_t* mask, int n, float* scratch, float* result, cudaStream_t s) {
  auto policy = thrust::cuda::par.on(s);
  float* end;
  if (mask) {  // keep the masked values, then drop the NaNs among them
    end = thrust::copy_if(policy, d, d + n, mask, scratch, NonZero());
    end = thrust::remove_if(policy, scratch, end, IsNan());
  } else {
    end = thrust::copy_if(policy, d, d + n, scratch, NotNan());
  }
  int m = (int)(end - scratch);
  if (m == 0) return false;
  thrust::sort(policy, scratch, end);
  double pos = 0.5 * (m - 1);
  int lo = (int)pos;
  float v[2] = {0.f, 0.f};
  CU(cudaMemcpyAsync(v, scratch + lo, (lo + 1 < m ? 2 : 1) * sizeof(float), cudaMemcpyDeviceToHost, s));
  CU(cudaStreamSynchronize(s));
  if (lo + 1 >= m) v[1] = v[0];
  *result = v[0] + (float)(pos - lo) * (v[1] - v[0]);
  return true;
}

void fill_disparity(const float* d, const uint8_t* band, float dp, float dd, float* out, int n, cudaStream_t s) {
  LAUNCH(fill_kernel, n, s)(d, band, dp, dd, out, n);
}
void soft_alpha(const float* p, float lo, float hi, float* out, int n, cudaStream_t s) {
  LAUNCH(soft_kernel, n, s)(p, lo, hi, out, n);
}
void guided_stack(const float* g255, const float* p, float* g, float* stack4, int n, cudaStream_t s) {
  LAUNCH(guided_stack_kernel, n, s)(g255, p, g, stack4, n);
}
void guided_ab(const float* box4, float eps, float* ab, int n, cudaStream_t s) {
  LAUNCH(guided_ab_kernel, n, s)(box4, eps, ab, n);
}
void guided_out(const float* box2, const float* g, float* q, int n, cudaStream_t s) {
  LAUNCH(guided_out_kernel, n, s)(box2, g, q, n);
}
void ema(const float* p, const float* prev_w, const uint8_t* valid, const float* photo, float lam, float sigma,
         float gate, float* out, int n, cudaStream_t s) {
  LAUNCH(ema_kernel, n, s)(p, prev_w, valid, photo, lam, sigma, gate, out, n);
}
void negate(const float* in, float* out, int n, cudaStream_t s) { LAUNCH(negate_kernel, n, s)(in, out, n); }
void average(const float* a, const float* b, float* out, int n, cudaStream_t s) {
  LAUNCH(average_kernel, n, s)(a, b, out, n);
}

void clear_textures() {
  for (auto& kv : tex_cache()) cudaDestroyTextureObject(kv.second);
  tex_cache().clear();
}

}  // namespace k
