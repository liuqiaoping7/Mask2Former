// The API-independent half: slot queue, copy-mode buffers, correctness check, statistics.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include "gpu.hpp"
#include "renderer.hpp"

namespace render {
namespace {
double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}
double p95(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[(size_t)(0.95 * (v.size() - 1))];
}
}  // namespace

Presenter::Presenter(const Config& c) : c_(c), bytes_((size_t)c.width * c.height * 3) {
  if (!glfwInit()) throw std::runtime_error("glfwInit failed (is DISPLAY set?)");
  int w = c.win_w ? c.win_w : std::min(c.width, 1920);
  int h = c.win_h ? c.win_h : (int)((long long)w * c.height / c.width);
  glfwDefaultWindowHints();
  if (c.api == "gl")
    gl_window_hints();
  else if (c.api == "vulkan")
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  else
    throw std::runtime_error("unknown render api " + c.api);
  std::string title = "SBS portrait - " + c.api + (c.zerocopy ? " zero-copy" : " host copy");
  win_ = glfwCreateWindow(w, h, title.c_str(), nullptr, nullptr);
  if (!win_) throw std::runtime_error("cannot create a window");
  backend_ = c.api == "gl" ? make_gl(c_, win_) : make_vulkan(c_, win_);
  q_ = std::make_unique<SlotQueue>(c.slots, c.drop);
  slots_.resize(c.slots);
  for (auto& s : slots_) {
    if (!c.zerocopy) CU(cudaMallocHost(&s.host, bytes_));
    if (c.check) CU(cudaMallocHost(&s.expect, bytes_));
    CU(cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming));
  }
}

Presenter::~Presenter() {
  cudaDeviceSynchronize();
  backend_.reset();
  for (auto& s : slots_) {
    if (s.host) cudaFreeHost(s.host);
    if (s.expect) cudaFreeHost(s.expect);
    cudaEventDestroy(s.ready);
  }
  if (win_) glfwDestroyWindow(win_);
  glfwTerminate();
}

bool Presenter::begin(uint8_t** dst, cudaStream_t s) {
  double t0 = now_ms();
  cur_ = q_->acquire();
  wait_ms_.push_back(now_ms() - t0);
  if (cur_ < 0) return false;
  *dst = c_.zerocopy ? backend_->cuda_begin(cur_, s) : nullptr;
  return true;
}

void Presenter::end(const uint8_t* frame, cudaStream_t s) {
  Slot& sl = slots_[cur_];
  sl.frame = produced_++;
  sl.check = sl.frame < c_.check;
  if (sl.check) CU(cudaMemcpyAsync(sl.expect, frame, bytes_, cudaMemcpyDeviceToHost, s));
  if (!c_.zerocopy) CU(cudaMemcpyAsync(sl.host, frame, bytes_, cudaMemcpyDeviceToHost, s));
  if (sl.check || !c_.zerocopy) CU(cudaEventRecord(sl.ready, s));
  if (c_.zerocopy) backend_->cuda_end(cur_, s);
  sl.published = now_ms();
  q_->publish(cur_);
}

void Presenter::run() {
  std::vector<uint8_t> rgba(c_.check ? (size_t)c_.width * c_.height * 4 : 0);
  while (!glfwWindowShouldClose(win_)) {
    glfwPollEvents();
    if (glfwGetKey(win_, GLFW_KEY_ESCAPE) == GLFW_PRESS || glfwGetKey(win_, GLFW_KEY_Q) == GLFW_PRESS) break;
    int i = q_->take(10);
    if (i == -2) break;
    if (i < 0) continue;
    Slot& sl = slots_[i];
    if (sl.check || !c_.zerocopy) CU(cudaEventSynchronize(sl.ready));
    double t0 = now_ms();
    backend_->draw(i, c_.zerocopy ? nullptr : sl.host, sl.check ? rgba.data() : nullptr);
    double t1 = now_ms();
    if (sl.check) {  // the buffer is BGR, the read-back image R G B A
      const uint8_t* e = sl.expect;
      for (size_t p = 0, n = (size_t)c_.width * c_.height; p < n; ++p)
        for (int k = 0; k < 3; ++k) {
          int d = std::abs((int)rgba[4 * p + k] - (int)e[3 * p + 2 - k]);
          check_max_ = std::max(check_max_, d);
          check_bad_ += d != 0;
        }
      ++checked_;
      if (on_checked) on_checked(sl.frame, rgba.data());
    } else {  // checked frames carry the read-back time: not a latency sample
      latency_ms_.push_back(t1 - sl.published);
      draw_ms_.push_back(t1 - t0);
    }
    if (backend_->gpu_draw_ms() >= 0) gpu_ms_.push_back(backend_->gpu_draw_ms());
    ++presented_;
    q_->release(i);
  }
  q_->close();
}

std::string Presenter::stats_json() const {
  std::ostringstream j;
  j << "{\"api\": \"" << c_.api << "\", \"present\": \"" << (c_.zerocopy ? "zerocopy" : "copy") << "\", \"vsync\": "
    << c_.vsync << ", \"queue\": \"" << (c_.drop ? "drop" : "block") << "\", \"slots\": " << c_.slots
    << ", \"produced\": " << produced_ << ", \"presented\": " << presented_ << ", \"dropped\": " << q_->dropped()
    << ", \"acquire_wait_ms\": {\"median\": " << median(wait_ms_) << ", \"p95\": " << p95(wait_ms_)
    << "}, \"latency_ms\": {\"median\": " << median(latency_ms_) << ", \"p95\": " << p95(latency_ms_)
    << "}, \"draw_cpu_ms\": " << median(draw_ms_) << ", \"draw_gpu_ms\": " << median(gpu_ms_) << ", \"checked\": "
    << checked_ << ", \"check_max_diff\": " << check_max_ << ", \"check_diff_values\": " << check_bad_ << "}";
  return j.str();
}

}  // namespace render
