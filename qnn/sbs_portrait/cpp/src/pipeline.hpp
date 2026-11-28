// The production pipeline (pipeline.SBSPortraitPipeline, mode "opt" + gpu_post.GPUPost):
//   CPU worker: downscale, SGBM disparity, DIS flows, upload on its own stream
//   GPU: segmentation, flow validity, disparity fill, soft / guided / EMA / fusion, compositing
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include "geometry.hpp"
#include "segmenter.hpp"

struct PipelineConfig {
  float soft_lo = 0.4f, soft_hi = 0.6f, gf_eps = 1e-4f, lam = 0.7f, gate = 0.5f, sigma = 10.0f;
  double bg_margin = 4.0, geo_late_ms = 5.0;
  int geo_threads = 4;
  std::string pip;  // "", "frame" or the path of an SBS RGBA png
};

struct Geometry {
  cv::Mat gL, gR, dL, dR, flow[4];  // flow: fbL (cur->prev), ffL, fbR, ffR
  bool has_flow = false;
};

class Pipeline {
 public:
  Pipeline(Segmenter& seg, const cv::Mat& first_frame, const cv::Mat& bg_sbs, Timer* timer, const PipelineConfig& cfg);
  ~Pipeline();
  // returns the composited SBS frame (H, 2W, 3) in a reused pinned buffer
  const uint8_t* process(const cv::Mat& frame);
  int W() const { return W_; }
  int H() const { return H_; }
  geo::Work work() const { return ws_; }
  geo::Params params() const { return prm_; }
  bool geo_parallel() const { return geo_par_; }
  double init_person_disp() const { return init_person_disp_; }
  double bg_shift() const { return bg_shift_; }
  std::pair<int, int> model_input_hw() const { return {in_->nh, in_->nw}; }
  std::pair<int, int> padded_hw() const { return {in_->Hp, in_->Wp}; }
  void dump_outputs(const std::string& dir, int t);  // final aL/aR/pL/pR/hL/hR (not timed)

 private:
  Geometry geometry(const cv::Mat& L, const cv::Mat& R);
  void upload_geometry(const Geometry& g);
  void post(const Geometry& g);
  void dilate_band(const uint8_t* hard, const float* prob, uint8_t* band);

  Segmenter& seg_;
  Timer* timer_;
  PipelineConfig cfg_;
  int W_, H_, n_;  // eye size, work pixels
  geo::Work ws_;
  geo::Params prm_;
  std::unique_ptr<geo::Stereo> stereo_, stereo_r_;
  geo::Flow flow_, flow_r_;
  std::unique_ptr<geo::Pool> worker_, gpool_;
  cv::Mat prev_gray_[2];
  bool geo_par_ = false;
  double late_ema_ = 0, init_person_disp_ = 0, bg_shift_ = 0;
  float person_disp_ = 0;
  int frames_ = 0;

  std::unique_ptr<Input> in_;
  std::unique_ptr<PersonOut> seg_out_;
  Stream main_, up_;
  Event ev_up_, ev_seg_done_{true}, ev_now_{true};

  // frame / output
  Pin<uint8_t> frame_pin_, out_pin_;
  Dev<uint8_t> frame_, bg_, out_;
  // geometry uploads
  Pin<uint8_t> gray_pin_[2];
  Pin<float> disp_pin_[2], flow_pin_[4];
  Dev<uint8_t> gray_[2];
  Dev<float> disp_[2], flow_dev_[4];
  // post-processing
  Dev<float> g_[2][2];  // float gray of the current / previous frame, per eye
  int cur_ = 0;
  bool have_prev_ = false;
  Dev<float> ff_chw_, ff_at_, prev_w_, photo_[2], warped_, filled_[2], scratch_, soft_, guide_, stack4_, tmp4_, box4_,
      ab_, box2_, q_, ema_[2][2], neg_, alpha_, pad_, apad_;
  int ema_cur_[2] = {0, 0};
  bool have_ema_[2] = {false, false};
  Dev<uint8_t> valid_[2], tmp8_, core_, band_[2];
  Dev<int> dh_, half_w_;
  int ell_r_ = 0, apitch_ = 0;
  // picture in picture
  k::Pip pip_ = {};
  DevTaps pip_x_, pip_y_;
  Dev<uint8_t> pip_icon_, pip_rgba_;
  bool pip_from_frame_ = false;
};
