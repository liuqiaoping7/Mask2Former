#include "trt_chain.hpp"
#include <NvInferPlugin.h>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace nvinfer1;

namespace {

constexpr size_t ALIGN = 512;  // setDeviceMemory requires 512-byte aligned scratch (outputs need 256)
size_t align_up(size_t n) { return (n + ALIGN - 1) / ALIGN * ALIGN; }

struct Logger : ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "[trt] %s\n", msg);
  }
};

std::vector<int64_t> to_vec(const Dims& d) { return {d.d, d.d + d.nbDims}; }
Dims to_dims(const std::vector<int64_t>& v) {
  Dims d;
  d.nbDims = (int)v.size();
  for (size_t i = 0; i < v.size(); ++i) d.d[i] = v[i];
  return d;
}
int64_t volume(const std::vector<int64_t>& v) {
  int64_t n = 1;
  for (auto x : v) n *= x;
  return n;
}
size_t type_size(DataType t) {
  switch (t) {
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    default: throw std::runtime_error("unexpected engine dtype");
  }
}

// narrowest profile (smallest max - min volume) whose range contains all input shapes
int profile_for(const ICudaEngine& e, const std::vector<std::string>& inputs,
                const std::map<std::string, std::vector<int64_t>>& shapes) {
  int best = -1;
  int64_t best_v = 0;
  for (int p = 0; p < e.getNbOptimizationProfiles(); ++p) {
    bool ok = true;
    int64_t v = 0;
    for (auto& n : inputs) {
      auto lo = to_vec(e.getProfileShape(n.c_str(), p, OptProfileSelector::kMIN));
      auto hi = to_vec(e.getProfileShape(n.c_str(), p, OptProfileSelector::kMAX));
      auto& s = shapes.at(n);
      for (size_t i = 0; i < s.size(); ++i) ok = ok && lo[i] <= s[i] && s[i] <= hi[i];
      v += volume(hi) - volume(lo);
    }
    if (ok && (best < 0 || v < best_v)) { best = p; best_v = v; }
  }
  if (best < 0) throw std::runtime_error("no optimization profile covers the input shape");
  return best;
}

}  // namespace

TrtChain::TrtChain(const std::vector<std::string>& plans, const std::vector<std::map<std::string, std::string>>& links)
    : logger_(new Logger), links_(links) {
  initLibNvInferPlugins(logger_.get(), "");  // MultiScaleDeformableAttnPlugin of the pixel decoder
  runtime_.reset(createInferRuntime(*logger_));
  for (auto& path : plans) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Stage st;
    st.engine.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!st.engine) throw std::runtime_error("deserialize failed: " + path);
    for (int i = 0; i < st.engine->getNbIOTensors(); ++i) {
      const char* n = st.engine->getIOTensorName(i);
      (st.engine->getTensorIOMode(n) == TensorIOMode::kINPUT ? st.inputs : st.outputs).push_back(n);
    }
    st.ctx.resize(st.engine->getNbOptimizationProfiles());
    stages_.push_back(std::move(st));
  }
  for (size_t i = 1; i < stages_.size(); ++i)
    for (auto& [in, out] : links_[i])
      if (stages_[i].engine->getTensorDataType(in.c_str()) != stages_[i - 1].engine->getTensorDataType(out.c_str()))
        throw std::runtime_error("dtype of a chain link differs: " + in);
}

TrtChain::~TrtChain() {
  for (auto& s : stages_) s.ctx.clear();  // contexts before their engines, engines before the runtime
  stages_.clear();
}

IExecutionContext* TrtChain::context(size_t i, int p, cudaStream_t stream) {
  auto& c = stages_[i].ctx[p];
  if (!c) {
    c.reset(stages_[i].engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!c || !c->setOptimizationProfileAsync(p, stream)) throw std::runtime_error("cannot create execution context");
  }
  return c.get();
}

TrtChain::Layout TrtChain::make_layout(int H, int W, cudaStream_t stream) {
  Layout lay;
  std::map<std::string, std::vector<int64_t>> shapes{{stages_[0].inputs[0], {1, 3, H, W}}};
  for (size_t i = 0; i < stages_.size(); ++i) {
    auto& st = stages_[i];
    StageLayout s;
    s.profile = profile_for(*st.engine, st.inputs, shapes);
    s.in_shape = shapes;
    auto* ctx = context(i, s.profile, stream);
    for (auto& n : st.inputs)
      if (!ctx->setInputShape(n.c_str(), to_dims(shapes.at(n)))) throw std::runtime_error("setInputShape " + n);
    for (auto& n : st.outputs) {
      s.out_shape[n] = to_vec(ctx->getTensorShape(n.c_str()));
      s.out_bytes[n] = align_up(volume(s.out_shape[n]) * type_size(st.engine->getTensorDataType(n.c_str())));
      s.out_size += s.out_bytes[n];
    }
    s.scratch = align_up(ctx->updateDeviceMemorySizeForShapes());
    for (auto it = bound_.begin(); it != bound_.end();)  // shapes changed on this context: rebind
      it = (it->first.first == i && it->first.second == s.profile) ? bound_.erase(it) : std::next(it);
    if (i + 1 < stages_.size()) {
      shapes.clear();
      for (auto& [in, out] : links_[i + 1]) shapes[in] = s.out_shape.at(out);
    }
    lay.st.push_back(std::move(s));
  }
  for (size_t i = 0; i < lay.st.size(); ++i) {
    size_t prev = i ? lay.st[i - 1].out_size : 0;
    lay.total = std::max(lay.total, prev + lay.st[i].scratch + lay.st[i].out_size);
  }
  for (size_t i = 0; i < lay.st.size(); ++i) {
    auto& s = lay.st[i];
    bool bottom = i % 2 == 0;
    size_t prev = i ? lay.st[i - 1].out_size : 0, o = bottom ? 0 : lay.total - s.out_size;
    for (auto& n : stages_[i].outputs) {
      s.out_off[n] = o;
      o += s.out_bytes[n];
    }
    s.scratch_off = bottom ? s.out_size : prev;
  }
  return lay;
}

std::map<std::string, ChainOutput> TrtChain::operator()(const float* x, int H, int W, cudaStream_t stream) {
  auto key = std::make_pair(H, W);
  auto it = layouts_.find(key);
  if (it == layouts_.end()) it = layouts_.emplace(key, make_layout(H, W, stream)).first;
  const Layout& lay = it->second;
  if (!arena_.p || arena_.n < lay.total) {
    CU(cudaDeviceSynchronize());
    arena_.alloc(lay.total);
    ++version_;
  }
  uint8_t* base = arena_.p;
  for (size_t i = 0; i < stages_.size(); ++i) {
    auto& st = stages_[i];
    const StageLayout& s = lay.st[i];
    auto* ctx = context(i, s.profile, stream);
    auto tag = std::make_pair(key, version_);
    auto bk = std::make_pair(i, s.profile);
    if (!bound_.count(bk) || bound_[bk] != tag) {
      for (auto& n : st.inputs) ctx->setInputShape(n.c_str(), to_dims(s.in_shape.at(n)));
      ctx->setDeviceMemoryV2(base + s.scratch_off, (int64_t)s.scratch);
      for (auto& [n, off] : s.out_off) ctx->setTensorAddress(n.c_str(), base + off);
      if (i)
        for (auto& [in, out] : links_[i]) ctx->setTensorAddress(in.c_str(), base + lay.st[i - 1].out_off.at(out));
      bound_[bk] = tag;
    }
    if (i == 0) ctx->setTensorAddress(st.inputs[0].c_str(), const_cast<float*>(x));
    if (!ctx->enqueueV3(stream)) throw std::runtime_error("TensorRT execution failed");
  }
  std::map<std::string, ChainOutput> out;
  auto& last = stages_.back();
  for (auto& n : last.outputs)
    out[n] = {base + lay.st.back().out_off.at(n), lay.st.back().out_shape.at(n), last.engine->getTensorDataType(n.c_str())};
  return out;
}
