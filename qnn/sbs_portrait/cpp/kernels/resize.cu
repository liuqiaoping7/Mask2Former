// OpenCV INTER_AREA resize + cvRound + BGR->RGB + mean/std normalisation, one pass.
// Ported from area_cuda.py (Python reference of the production Triton kernel area_triton.py);
// the output here is written straight into the padded model-input buffer.
#include "kernels.hpp"
#include "tex.cuh"

namespace k {
namespace {

struct Out {
  float* p;
  int nh, nw, Hp, Wp;
  float mean[3], std_[3];
  __device__ __forceinline__ void store(int y, int x, int c, float v) const {
    v = fminf(fmaxf(rintf(v), 0.f), 255.f);  // saturate_cast<uchar>
    int oc = 2 - c;                          // BGR -> RGB
    p[((size_t)oc * Hp + y) * Wp + x] = (v - mean[oc]) / std_[oc];
  }
};

template <bool TEX, int K>
__global__ void taps_kernel(const uint8_t* __restrict__ eye, cudaTextureObject_t tex, int off, int row,
                            const int* __restrict__ ix, const float* __restrict__ wx, const int* __restrict__ iy,
                            const float* __restrict__ wy, Out o) {
  int offs = blockIdx.x * blockDim.x + threadIdx.x;
  if (offs >= o.nh * o.nw) return;
  int y = offs / o.nw, x = offs % o.nw;
  int sx[K];
  float fx[K];
  for (int k = 0; k < K; ++k) {
    sx[k] = ix[x * K + k] * 3;
    fx[k] = wx[x * K + k];
  }
  float acc[3] = {0.f, 0.f, 0.f};
  for (int ky = 0; ky < K; ++ky) {
    int base = iy[y * K + ky] * row;
    float fy = wy[y * K + ky];
    float r[3] = {0.f, 0.f, 0.f};
    for (int kx = 0; kx < K; ++kx) {
      int p = base + sx[kx];
      for (int c = 0; c < 3; ++c) {
        float v = TEX ? (float)tex1Dfetch<unsigned char>(tex, off + p + c) : (float)eye[p + c];
        r[c] += fx[kx] * v;
      }
    }
    for (int c = 0; c < 3; ++c) acc[c] += fy * r[c];
  }
  for (int c = 0; c < 3; ++c) o.store(y, x, c, acc[c]);
}

template <int M>
__global__ void linear_kernel(cudaTextureObject_t tex, const float* __restrict__ px, const float* __restrict__ wx,
                              const float* __restrict__ py, const float* __restrict__ wy, Out o) {
  int offs = blockIdx.x * blockDim.x + threadIdx.x;
  if (offs >= o.nh * o.nw) return;
  int y = offs / o.nw, x = offs % o.nw;
  float u[M], fx[M];
  for (int k = 0; k < M; ++k) {
    u[k] = px[x * M + k];
    fx[k] = wx[x * M + k];
  }
  float acc[3] = {0.f, 0.f, 0.f};
  for (int ky = 0; ky < M; ++ky) {
    float v = py[y * M + ky], fy = wy[y * M + ky];
    float r[3] = {0.f, 0.f, 0.f};
    for (int kx = 0; kx < M; ++kx) {
      float4 t = tex2D<float4>(tex, u[kx], v);
      r[0] += fx[kx] * t.x;
      r[1] += fx[kx] * t.y;
      r[2] += fx[kx] * t.z;
    }
    for (int c = 0; c < 3; ++c) acc[c] += fy * r[c];
  }
  for (int c = 0; c < 3; ++c) o.store(y, x, c, acc[c] * 255.f);  // normalized-float read mode
}

__global__ void eye_to_rgba_kernel(const uint8_t* __restrict__ src, int row, uchar4* __restrict__ dst, int W, int Wp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x >= W) return;
  const uint8_t* s = src + (size_t)y * row + x * 3;
  dst[(size_t)y * Wp + x] = make_uchar4(s[0], s[1], s[2], 255);
}

Out make_out(float* out, int nh, int nw, int Hp, int Wp, const float mean[3], const float std_[3]) {
  Out o{out, nh, nw, Hp, Wp, {mean[0], mean[1], mean[2]}, {std_[0], std_[1], std_[2]}};
  return o;
}

}  // namespace

std::map<TexKey, cudaTextureObject_t>& tex_cache() {
  static std::map<TexKey, cudaTextureObject_t> m;
  return m;
}

void resize_taps(Mode mode, const uint8_t* eye, int row, const uint8_t* frame, size_t frame_bytes, Taps x, Taps y,
                 int K, int nh, int nw, int Hp, int Wp, const float mean[3], const float std_[3], float* out,
                 cudaStream_t s) {
  Out o = make_out(out, nh, nw, Hp, Wp, mean, std_);
  bool tex = mode == TEX_POINT;
  cudaTextureObject_t t = tex ? tex_1d<unsigned char>(frame, frame_bytes) : 0;
  int off = (int)(eye - frame), blocks = (nh * nw + 255) / 256;
#define RUN(K_)                                                                                   \
  case K_:                                                                                        \
    if (tex) taps_kernel<true, K_><<<blocks, 256, 0, s>>>(eye, t, off, row, x.i, x.w, y.i, y.w, o); \
    else taps_kernel<false, K_><<<blocks, 256, 0, s>>>(eye, t, off, row, x.i, x.w, y.i, y.w, o);   \
    break;
  switch (K) {
    RUN(2) RUN(3) RUN(4) RUN(5)
    default: throw std::runtime_error("unsupported tap count");
  }
#undef RUN
  CU(cudaGetLastError());
}

void resize_linear(const uint8_t* eye, int row, int h, int w, uint8_t* rgba, int wp, LinTaps x, LinTaps y, int M,
                   int nh, int nw, int Hp, int Wp, const float mean[3], const float std_[3], float* out, cudaStream_t s) {
  eye_to_rgba_kernel<<<dim3((w + 255) / 256, h), 256, 0, s>>>(eye, row, (uchar4*)rgba, w, wp);
  cudaTextureObject_t t = tex_2d<uchar4>(rgba, w, h, (size_t)wp * 4, cudaReadModeNormalizedFloat);
  Out o = make_out(out, nh, nw, Hp, Wp, mean, std_);
  int blocks = (nh * nw + 255) / 256;
#define RUN(M_) \
  case M_: linear_kernel<M_><<<blocks, 256, 0, s>>>(t, x.pos, x.w, y.pos, y.w, o); break;
  switch (M) {
    RUN(1) RUN(2) RUN(3)
    default: throw std::runtime_error("unsupported fetch count");
  }
#undef RUN
  CU(cudaGetLastError());
}

}  // namespace k
