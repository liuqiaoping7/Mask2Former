// CUDA -> OpenGL / Vulkan presentation, decoupled from the producer (inference) by a slot queue.
//
//   producer thread                              consumer (main) thread
//   dst = begin(stream)   take a slot            take a ready slot
//   <kernel writes dst>                          draw it (OpenGL / Vulkan), present
//   end(stream)           publish it             release it
//
// zero-copy: dst is the graphics API's own buffer mapped into CUDA, nothing is copied afterwards.
// copy     : the baseline: end() copies device -> pinned host, the consumer uploads host -> GPU.
#pragma once
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "queue.hpp"
#include "sink.hpp"

namespace render {

struct Config {
  std::string api = "gl";  // "gl" | "vulkan"
  int width = 0, height = 0;  // source frame (SBS: 2W x H), BGR u8
  int slots = 3;
  bool zerocopy = true, vsync = true, drop = false, validate = false;
  int win_w = 0, win_h = 0;  // 0: fit the source into 1920 columns
  int check = 0;             // first N frames: render at source size, read back, compare with CUDA
  std::string shader_dir;
};

// API-specific half. ctor / dtor / draw run on the consumer thread (which owns the window);
// cuda_begin / cuda_end run on the producer thread and only enqueue work on its stream.
class Backend {
 public:
  virtual ~Backend() = default;
  // zero-copy: wait (on the stream) until the graphics API released the slot, return its CUDA pointer
  virtual uint8_t* cuda_begin(int slot, cudaStream_t s) = 0;
  // zero-copy: signal (on the stream) that the CUDA writes of the slot are complete
  virtual void cuda_end(int slot, cudaStream_t s) = 0;
  // host: copy-mode source (null when zero-copy). rgba: if not null, also render at source size and
  // read the result back into it (width * height * 4, top row first, R G B A)
  virtual void draw(int slot, const uint8_t* host, uint8_t* rgba) = 0;
  virtual double gpu_draw_ms() const = 0;  // last measured GPU time of one draw (< 0: none yet)
  virtual std::string description() const = 0;
};

std::unique_ptr<Backend> make_gl(const Config& c, GLFWwindow* win);
std::unique_ptr<Backend> make_vulkan(const Config& c, GLFWwindow* win);
void gl_window_hints();

class Presenter : public FrameSink {
 public:
  explicit Presenter(const Config& c);  // consumer thread: window, backend, slot buffers
  ~Presenter() override;
  // ---- producer thread
  // returns false once the window is closed. dst: device pointer to write the frame to, or nullptr
  // (copy mode: write to your own buffer and pass it to end()).
  bool begin(uint8_t** dst, cudaStream_t s) override;
  void end(const uint8_t* device_frame, cudaStream_t s) override;
  void finish() { q_->finish(); }
  // ---- consumer thread: render until the producer finishes or the window is closed
  void run();
  // called for every checked frame with the read-back image (R G B A, source size)
  std::function<void(int frame, const uint8_t* rgba)> on_checked;
  std::string stats_json() const;
  std::string description() const { return backend_->description(); }

 private:
  struct Slot {
    uint8_t *host = nullptr, *expect = nullptr;  // pinned: copy-mode frame, check reference
    cudaEvent_t ready = nullptr;                 // host copies finished
    double published = 0;
    int frame = 0;
    bool check = false;
  };
  Config c_;
  size_t bytes_;
  GLFWwindow* win_ = nullptr;
  std::unique_ptr<Backend> backend_;
  std::unique_ptr<SlotQueue> q_;
  std::vector<Slot> slots_;
  int cur_ = -1, produced_ = 0;
  // statistics
  std::vector<double> wait_ms_, latency_ms_, draw_ms_, gpu_ms_;
  int presented_ = 0, checked_ = 0, check_max_ = 0;
  long long check_bad_ = 0;
};

}  // namespace render
