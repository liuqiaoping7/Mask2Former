// Texture objects cached by (address, shape); the pipeline's buffers live as long as the pipeline.
#pragma once
#include <cuda_runtime.h>
#include <map>
#include <tuple>
#include "gpu.hpp"

namespace k {

// resource kind, address, width, height, pitch, filter, address mode
using TexKey = std::tuple<int, const void*, size_t, size_t, size_t, int, int>;
std::map<TexKey, cudaTextureObject_t>& tex_cache();

// 1-D point-fetch texture over `count` elements of linear memory
template <class T>
cudaTextureObject_t tex_1d(const void* ptr, size_t count, cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>()) {
  TexKey key{0, ptr, count, 0, 0, 0, desc.x};
  auto it = tex_cache().find(key);
  if (it != tex_cache().end()) return it->second;
  cudaResourceDesc rd = {};
  rd.resType = cudaResourceTypeLinear;
  rd.res.linear.devPtr = const_cast<void*>(ptr);
  rd.res.linear.desc = desc;
  rd.res.linear.sizeInBytes = count * (desc.x / 8);
  cudaTextureDesc td = {};
  td.filterMode = cudaFilterModePoint;
  td.readMode = cudaReadModeElementType;
  cudaTextureObject_t t = 0;
  CU(cudaCreateTextureObject(&t, &rd, &td, nullptr));
  return tex_cache()[key] = t;
}

// 2-D linear-filter texture over pitched memory (pitch multiple of 32, base 512-byte aligned)
template <class T>
cudaTextureObject_t tex_2d(const void* ptr, size_t w, size_t h, size_t pitch, cudaTextureReadMode read) {
  TexKey key{1, ptr, w, h, pitch, (int)read, (int)sizeof(T)};
  auto it = tex_cache().find(key);
  if (it != tex_cache().end()) return it->second;
  cudaResourceDesc rd = {};
  rd.resType = cudaResourceTypePitch2D;
  rd.res.pitch2D.devPtr = const_cast<void*>(ptr);
  rd.res.pitch2D.desc = cudaCreateChannelDesc<T>();
  rd.res.pitch2D.width = w;
  rd.res.pitch2D.height = h;
  rd.res.pitch2D.pitchInBytes = pitch;
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
  td.filterMode = cudaFilterModeLinear;
  td.readMode = read;
  cudaTextureObject_t t = 0;
  CU(cudaCreateTextureObject(&t, &rd, &td, nullptr));
  return tex_cache()[key] = t;
}

}  // namespace k
