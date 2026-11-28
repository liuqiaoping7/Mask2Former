// GPU entry points of the C++ pipeline. Plain device pointers + a stream; no framework.
// Mode selects the memory path of the sampling / neighbourhood operators:
//   GLOBAL      global-memory loads
//   TEX_POINT   the same loads through the texture unit (point fetches)
//   TEX_LINEAR  hardware bilinear filtering (8-bit fixed-point weights)
// Point-wise operators and reductions have a single (global) version: they read each input once
// at the thread's own index, so a texture path has nothing to add.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace k {

enum Mode { GLOBAL = 0, TEX_POINT = 1, TEX_LINEAR = 2 };

void clear_textures();  // drop all cached texture objects / arrays (call when buffers are freed)

// ---------------------------------------------------------------- model-input resize (resize.cu)
struct Taps {      // per output index: K source indices and weights (device pointers)
  const int* i;
  const float* w;
};
struct LinTaps {   // TEX_LINEAR: M merged fetch positions and weights per output index
  const float* pos;
  const float* w;
};
// eye: BGR u8 inside the frame [frame, frame + frame_bytes), rows `row` bytes apart.
// out: (3, Hp, Wp) float, only the (nh, nw) top-left region is written (the padding stays).
void resize_taps(Mode mode, const uint8_t* eye, int row, const uint8_t* frame, size_t frame_bytes, Taps x, Taps y,
                 int K, int nh, int nw, int Hp, int Wp, const float mean[3], const float std_[3], float* out,
                 cudaStream_t s);
// rgba: (h, wp, 4) work buffer, wp a multiple of 128
void resize_linear(const uint8_t* eye, int row, int h, int w, uint8_t* rgba, int wp, LinTaps x, LinTaps y, int M,
                   int nh, int nw, int Hp, int Wp, const float mean[3], const float std_[3], float* out, cudaStream_t s);

// ---------------------------------------------------------------- person post-processing (person.cu)
// logits (Q, C) -> softmax; wp = p(person), wd = 1 - p(void), sk = score * keep, flag = keep && label == person
void query_weights(const float* logits, int Q, int C, int person_id, float obj_thr, float* wp, float* wd, float* sk,
                   int* flag, cudaStream_t s);
// masks: (Q, H4, W4) half logits; own_cnt: Q ints, zeroed by the callee
void person(Mode mode, const void* masks, const float* wp, const float* wd, const float* sk, float* prob, int* ids,
            uint8_t* own_at, int* own_cnt, int Q, int H4, int W4, int OH, int OW, float sy, float sx, cudaStream_t s);
// vectorised panoptic rule for the person class; hist: 2 Q ints, valid: Q bytes (work buffers)
void panoptic_person(const int* ids, const uint8_t* own_at, const int* own_cnt, const int* flag, float overlap_thr,
                     int Q, int n, int* hist, uint8_t* valid, uint8_t* hard, cudaStream_t s);

// ---------------------------------------------------------------- alpha post-processing (post.cu)
void u8_to_f32(const uint8_t* in, float* out, int n, cudaStream_t s);
void deinterleave2(const float* hwc, float* chw, int n, cudaStream_t s);
// out(c, p) = img(c, p + (dx[p step], dy[p step])), bilinear, border replicate; dy may be null.
// img: (C, H, W), C in {1, 2}; pad: pitched scratch of warp_pad_bytes() for TEX_LINEAR
size_t pitched_bytes(int W, int rows);
void warp(Mode mode, const float* img, int C, int H, int W, const float* dx, const float* dy, int step, float* out,
          float* pad, cudaStream_t s);
// (2r+1)^2 mean, reflect-101, of N planes; tmp: same size as in
void box(Mode mode, const float* in, float* tmp, float* out, int N, int H, int W, int r, float* pad, cudaStream_t s);
void flow_validity(const float* g_cur, const float* fb_hwc, const float* ff_at, const float* prev_w, uint8_t* valid,
                   float* photo, int n, cudaStream_t s);
void or_above(const uint8_t* h, const float* p, float thr, uint8_t* out, int n, cudaStream_t s);
// binary dilation by an ellipse given as the half width of each of its 2r+1 rows; dh: H W ints
void dilate_ellipse(const uint8_t* m, int* dh, const int* half_w, int r, uint8_t* out, int H, int W, cudaStream_t s);
void erode(const uint8_t* m, uint8_t* tmp, uint8_t* out, int H, int W, int k, cudaStream_t s);
// median (linear interpolation) of d where mask (null: everywhere) and not NaN; false if empty.
// Synchronises with the stream (thrust). scratch: n floats
bool nanmedian(const float* d, const uint8_t* mask, int n, float* scratch, float* result, cudaStream_t s);
void fill_disparity(const float* d, const uint8_t* band, float d_person, float d_default, float* out, int n, cudaStream_t s);
void soft_alpha(const float* p, float lo, float hi, float* out, int n, cudaStream_t s);
// guided filter: stack4 = [g, p, g p, g g]; ab = [a, b] from the four box means; q = clamp(ma g + mb)
void guided_stack(const float* g255, const float* p, float* g, float* stack4, int n, cudaStream_t s);
void guided_ab(const float* box4, float eps, float* ab, int n, cudaStream_t s);
void guided_out(const float* box2, const float* g, float* q, int n, cudaStream_t s);
void ema(const float* p, const float* prev_w, const uint8_t* valid, const float* photo, float lam, float sigma,
         float gate, float* out, int n, cudaStream_t s);
void negate(const float* in, float* out, int n, cudaStream_t s);
void average(const float* a, const float* b, float* out, int n, cudaStream_t s);

// ---------------------------------------------------------------- compositing + PiP (compose.cu)
struct Pip {
  int on, ph, pw, y0, x0[2], has_alpha;
  const uint8_t* src;       // SBS source (Hs, 2 Ws, src_c)
  int src_row, src_c, Hs, Ws, K;
  Taps x, y;                // INTER_AREA taps Ws -> pw, Hs -> ph
  uint8_t* rgba;            // TEX_LINEAR: (2, Hs, wp, 4) planes of the source
  int wp;
  int rgba_ready;           // rgba already holds the source (a static icon is converted once)
};
// alpha: (2, h, w); apad: (2, h, apitch) float work buffer for TEX_LINEAR (apitch multiple of 128)
void compose(Mode mode, const uint8_t* frame, const uint8_t* bg, const float* alpha, float* apad, int apitch,
             uint8_t* out, int H, int W, int h, int w, const Pip& pip, cudaStream_t s);

}  // namespace k
