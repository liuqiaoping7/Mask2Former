// Where the composited frame goes instead of the pinned host buffer (implemented by
// render::Presenter). The pipeline writes its last kernel's output into the sink's buffer.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct FrameSink {
  virtual ~FrameSink() = default;
  // false: the consumer is gone, stop. dst: where to write (nullptr: use your own device buffer).
  virtual bool begin(uint8_t** dst, cudaStream_t s) = 0;
  // the frame written to device_frame is complete in stream order
  virtual void end(const uint8_t* device_frame, cudaStream_t s) = 0;
};
