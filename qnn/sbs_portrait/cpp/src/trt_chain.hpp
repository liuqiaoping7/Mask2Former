// The three TensorRT engines (backbone -> pixel decoder -> transformer decoder) run as one chain in a
// ping-pong arena: C++ counterpart of trt_common.TRTRunner.profile_for + TRTChain (see its docstring
// for the arena layout). The plans are the ones built by the Python pipeline.
#pragma once
#include <NvInferRuntime.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "gpu.hpp"

struct ChainOutput {
  const void* data;
  std::vector<int64_t> shape;
  nvinfer1::DataType type;
};

class TrtChain {
 public:
  // plans in chain order; links[i] (i >= 1): input name of stage i -> output name of stage i - 1
  TrtChain(const std::vector<std::string>& plans, const std::vector<std::map<std::string, std::string>>& links);
  ~TrtChain();
  // x: (1, 3, H, W) float. Outputs are views into the arena: consume them on `stream` before the next call.
  std::map<std::string, ChainOutput> operator()(const float* x, int H, int W, cudaStream_t stream);
  size_t arena_bytes() const { return arena_.n; }

 private:
  struct Stage {
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<std::string> inputs, outputs;
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> ctx;  // per optimization profile
  };
  struct StageLayout {
    int profile;
    std::map<std::string, std::vector<int64_t>> in_shape, out_shape;
    std::map<std::string, size_t> out_bytes, out_off;
    size_t out_size = 0, scratch = 0, scratch_off = 0;
  };
  struct Layout {
    size_t total = 0;
    std::vector<StageLayout> st;
  };
  nvinfer1::IExecutionContext* context(size_t i, int profile, cudaStream_t stream);
  Layout make_layout(int H, int W, cudaStream_t stream);

  std::unique_ptr<nvinfer1::ILogger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::vector<Stage> stages_;
  std::vector<std::map<std::string, std::string>> links_;
  std::map<std::pair<int, int>, Layout> layouts_;
  std::map<std::pair<size_t, int>, std::pair<std::pair<int, int>, int>> bound_;  // (stage, profile) -> (shape, arena version)
  Dev<uint8_t> arena_;
  int version_ = 0;
};
