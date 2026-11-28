// Per-module timer (pipeline.Timer), raw tensor dumps and small helpers.
#pragma once
#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <vector>
#include "gpu.hpp"

inline double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

class Timer {
 public:
  explicit Timer(bool sync = false) : sync_(sync) {}
  void new_frame() { std::lock_guard<std::mutex> lk(m_); frames_.emplace_back(); }
  struct Scope {
    Timer* t;
    const char* key;
    bool gpu;
    double t0;
    Scope(Timer* t, const char* key, bool gpu) : t(t), key(key), gpu(gpu) {
      if (t && gpu && t->sync_) cudaDeviceSynchronize();
      t0 = now_ms();
    }
    ~Scope() {
      if (!t) return;
      if (gpu && t->sync_) cudaDeviceSynchronize();
      t->add(key, now_ms() - t0);
    }
  };
  void add(const char* key, double ms) {
    std::lock_guard<std::mutex> lk(m_);
    frames_.back()[key] += ms;
  }
  // median over frames of the per-frame total of each module
  std::map<std::string, double> summary(size_t skip) const {
    std::vector<std::map<std::string, double>> f;
    for (auto& x : frames_)
      if (!x.empty()) f.push_back(x);
    if (f.size() > skip) f.erase(f.begin(), f.begin() + skip);
    std::map<std::string, std::vector<double>> col;
    for (auto& x : f)
      for (auto& kv : x) col[kv.first];
    for (auto& [k, v] : col)
      for (auto& x : f) v.push_back(x.count(k) ? x.at(k) : 0.0);
    std::map<std::string, double> out;
    for (auto& [k, v] : col) out[k] = median(v);
    return out;
  }

 private:
  bool sync_;
  std::mutex m_;
  std::vector<std::map<std::string, double>> frames_{1};
};

// <dir>/<name>.bin: one header line "dtype d0 d1 ..." followed by the raw data
inline void write_bin(const std::string& dir, const std::string& name, const char* dtype,
                      const std::vector<int64_t>& shape, const void* data, size_t bytes) {
  std::ofstream f(dir + "/" + name + ".bin", std::ios::binary);
  f << dtype;
  for (auto s : shape) f << ' ' << s;
  f << '\n';
  f.write((const char*)data, bytes);
}

// download a device buffer (after everything queued on `s`) and write it
inline void dump_dev(const std::string& dir, const std::string& name, const char* dtype,
                     const std::vector<int64_t>& shape, const void* dev, size_t elem, cudaStream_t s) {
  size_t n = elem;
  for (auto v : shape) n *= v;
  std::vector<char> host(n);
  CU(cudaMemcpyAsync(host.data(), dev, n, cudaMemcpyDeviceToHost, s));
  CU(cudaStreamSynchronize(s));
  write_bin(dir, name, dtype, shape, host.data(), n);
}

// --op-times: CUDA events around every GPU operator (main thread only). end_frame() after the
// device is idle turns them into per-frame totals per operator.
struct OpTimes {
  bool on = false;
  std::vector<std::pair<const char*, std::pair<cudaEvent_t, cudaEvent_t>>> open;
  std::vector<std::pair<cudaEvent_t, cudaEvent_t>> pool;
  std::map<std::string, std::vector<double>> ms;
  void end_frame(bool keep) {
    std::map<std::string, double> sum;
    for (auto& [name, ev] : open) {
      float t = 0;
      CU(cudaEventElapsedTime(&t, ev.first, ev.second));
      sum[name] += t;
      pool.push_back(ev);
    }
    open.clear();
    if (keep)
      for (auto& [name, t] : sum) ms[name].push_back(t);
  }
};
inline OpTimes g_ops;

struct OpScope {
  cudaStream_t s;
  cudaEvent_t end = nullptr;
  OpScope(const char* name, cudaStream_t s) : s(s) {
    if (!g_ops.on) return;
    std::pair<cudaEvent_t, cudaEvent_t> ev;
    if (g_ops.pool.empty()) {
      CU(cudaEventCreate(&ev.first));
      CU(cudaEventCreate(&ev.second));
    } else {
      ev = g_ops.pool.back();
      g_ops.pool.pop_back();
    }
    CU(cudaEventRecord(ev.first, s));
    end = ev.second;
    g_ops.open.push_back({name, ev});
  }
  ~OpScope() {
    if (end) cudaEventRecord(end, s);
  }
};
