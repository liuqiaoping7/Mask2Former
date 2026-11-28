// Person segmentation (seg.py: preprocess_gpu + person() with the TensorRT chain).
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "gpu.hpp"
#include "kernels.hpp"
#include "trt_chain.hpp"
#include "util.hpp"

// values exported from the Python model config (tools/export_assets.py)
struct Assets {
  std::string backbone, pixel_decoder, decoder, icon;
  float mean[3], std_[3], object_mask_threshold, overlap_threshold;
  int min_size, max_size, size_divisibility, person_id;
  static Assets load(const std::string& path);
};

// device copy of OpenCV INTER_AREA taps (src -> dst): K taps per output index, or M merged fetches
struct DevTaps {
  Dev<int> i;
  Dev<float> w, pos, mw;
  int K = 0, M = 0;
  k::Taps taps() const { return {i.p, w.p}; }
  k::LinTaps lin() const { return {pos.p, mw.p}; }
};
using RawTaps = std::vector<std::vector<std::pair<int, float>>>;
RawTaps area_taps(int src, int dst);         // cv::computeResizeAreaTab (downscale)
RawTaps linear_area_taps(int src, int dst);  // cv::resize INTER_AREA when upscaling
void upload_taps(const RawTaps& x, const RawTaps& y, DevTaps& dx, DevTaps& dy, bool merged);

// model input of one eye size
struct Input {
  int h, w, nh, nw, Hp, Wp;
  DevTaps tx, ty;
  Dev<float> data[2];  // (1, 3, Hp, Wp) per eye
  Dev<uint8_t> rgba;   // TEX_LINEAR scratch
  int rgba_wp = 0;
};

// per output size work buffers of the person post-processing
struct PersonOut {
  int w, h;
  Dev<float> prob[2];
  Dev<uint8_t> hard[2];
  Dev<int> ids;
  Dev<uint8_t> own_at;
};

class Segmenter {
 public:
  Segmenter(const Assets& a, k::Mode mode);
  std::unique_ptr<Input> make_input(int h, int w);
  std::unique_ptr<PersonOut> make_output(int w, int h);
  // eye `e` of a BGR u8 frame on the device (rows `row` bytes apart), on stream `s`
  void preprocess(Input& in, int e, const uint8_t* eye, int row, const uint8_t* frame, size_t frame_bytes, cudaStream_t s);
  // model + post-processing on the segmentation stream, ordered after / before `caller`
  void person(const Input& in, int e, PersonOut& out, cudaStream_t caller, Timer* tm);
  size_t arena_bytes() const { return chain_->arena_bytes(); }
  const std::string& stage_dir() const { return stage_dir_; }
  void set_dump(const std::string& dir, int frame) { stage_dir_ = dir; frame_ = frame; }
  const Assets assets;
  const k::Mode mode;

 private:
  std::unique_ptr<TrtChain> chain_;
  Stream stream_;
  Event ev_in_, ev_out_;
  Dev<float> wp_{256}, wd_{256}, sk_{256};
  Dev<int> flag_{256}, own_cnt_{256}, hist_{512};
  Dev<uint8_t> valid_{256};
  std::string stage_dir_;
  int frame_ = 0;
};
