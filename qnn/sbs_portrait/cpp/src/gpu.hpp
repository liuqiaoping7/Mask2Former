// Minimal RAII wrappers over the CUDA runtime (what torch's allocator / streams / events and
// gpu_post.Pinned do for the Python pipeline). Every buffer is allocated once per pipeline.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#define CU(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess)                                                                           \
      throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " +        \
                               cudaGetErrorString(e_));                                              \
  } while (0)

template <class T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  Dev() = default;
  explicit Dev(size_t count) { alloc(count); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  ~Dev() { if (p) cudaFree(p); }
  void alloc(size_t count) {
    if (p) CU(cudaFree(p));
    p = nullptr;
    n = count;
    CU(cudaMalloc(&p, count * sizeof(T)));
  }
  size_t bytes() const { return n * sizeof(T); }
  operator T*() const { return p; }
};

// page-locked host buffer for async H2D / D2H copies
template <class T>
struct Pin {
  T* p = nullptr;
  size_t n = 0;
  Pin() = default;
  explicit Pin(size_t count) { alloc(count); }
  Pin(const Pin&) = delete;
  Pin& operator=(const Pin&) = delete;
  ~Pin() { if (p) cudaFreeHost(p); }
  void alloc(size_t count) {
    if (p) CU(cudaFreeHost(p));
    p = nullptr;
    n = count;
    CU(cudaMallocHost(&p, count * sizeof(T)));
  }
  size_t bytes() const { return n * sizeof(T); }
  operator T*() const { return p; }
};

struct Stream {
  cudaStream_t s = nullptr;
  Stream() { CU(cudaStreamCreate(&s)); }
  Stream(const Stream&) = delete;
  ~Stream() { cudaStreamDestroy(s); }
  operator cudaStream_t() const { return s; }
};

struct Event {
  cudaEvent_t e = nullptr;
  explicit Event(bool timing = false) { CU(cudaEventCreateWithFlags(&e, timing ? cudaEventDefault : cudaEventDisableTiming)); }
  Event(const Event&) = delete;
  ~Event() { cudaEventDestroy(e); }
  void record(cudaStream_t s) { CU(cudaEventRecord(e, s)); }
  void wait_in(cudaStream_t s) { CU(cudaStreamWaitEvent(s, e, 0)); }
};

// make stream `waiter` wait for everything queued on `signal` so far (torch Stream.wait_stream)
inline void wait_stream(cudaStream_t waiter, cudaStream_t signal, Event& ev) {
  ev.record(signal);
  ev.wait_in(waiter);
}
