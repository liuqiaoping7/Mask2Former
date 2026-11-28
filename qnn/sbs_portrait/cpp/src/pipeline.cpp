#include "pipeline.hpp"
#include <cmath>
#include <cstring>
#include <opencv2/imgcodecs.hpp>

using cv::Mat;

namespace {

Mat down(const Mat& x, cv::Size size, int interp = cv::INTER_AREA) {
  Mat out;
  cv::resize(x, out, size, 0, 0, interp);
  return out;
}
Mat gray(const Mat& bgr) {
  Mat g;
  cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
  return g;
}
template <class T>
void upload(Dev<T>& dev, Pin<T>& pin, const Mat& m, cudaStream_t s) {
  size_t row = m.cols * m.elemSize(), bytes = row * m.rows;
  if (bytes != pin.bytes()) throw std::runtime_error("upload: size mismatch");
  for (int y = 0; y < m.rows; ++y) std::memcpy((char*)pin.p + y * row, m.ptr(y), row);
  CU(cudaMemcpyAsync(dev.p, pin.p, bytes, cudaMemcpyHostToDevice, s));
}

}  // namespace

Pipeline::Pipeline(Segmenter& seg, const Mat& first_frame, const Mat& bg_sbs, Timer* timer, const PipelineConfig& cfg)
    : seg_(seg), timer_(timer), cfg_(cfg) {
  W_ = first_frame.cols / 2;
  H_ = first_frame.rows;
  ws_ = geo::work_size(W_, H_);
  prm_ = geo::work_params(ws_.h);
  const int w = ws_.w, h = ws_.h, W = W_, H = H_;
  n_ = w * h;
  cv::Size ws(w, h);
  size_t frame_bytes = (size_t)H * 2 * W * 3;

  in_ = seg_.make_input(H, W);
  seg_out_ = seg_.make_output(w, h);
  frame_pin_.alloc(frame_bytes);
  out_pin_.alloc(frame_bytes);
  frame_.alloc(frame_bytes);
  bg_.alloc(frame_bytes);
  out_.alloc(frame_bytes);

  Mat L0 = first_frame.colRange(0, W), R0 = first_frame.colRange(W, 2 * W);
  CU(cudaMemcpy2D(frame_.p, (size_t)2 * W * 3, first_frame.data, first_frame.step, (size_t)2 * W * 3, H,
                  cudaMemcpyHostToDevice));
  auto segment_eyes = [&] {
    for (int e = 0; e < 2; ++e) {
      seg_.preprocess(*in_, e, frame_.p + (size_t)e * W * 3, 2 * W * 3, frame_.p, frame_.n, main_);
      seg_.person(*in_, e, *seg_out_, main_, nullptr);
    }
    CU(cudaStreamSynchronize(main_));
  };
  for (int i = 0; i < 2; ++i) segment_eyes();  // warm-up

  Mat gL0 = gray(down(L0, ws)), gR0 = gray(down(R0, ws));
  auto range = geo::disparity_range(gL0, gR0);
  stereo_ = std::make_unique<geo::Stereo>(range.first, range.second);
  if (cfg_.geo_threads > 1) {  // thread-private matchers
    stereo_r_ = std::make_unique<geo::Stereo>(range.first, range.second);
    gpool_ = std::make_unique<geo::Pool>(cfg_.geo_threads - 1);
  }
  worker_ = std::make_unique<geo::Pool>(1);

  // virtual background + depth placement
  Mat bgL = down(bg_sbs.colRange(0, bg_sbs.cols / 2).clone(), cv::Size(W, H), cv::INTER_LINEAR);
  Mat bgR = down(bg_sbs.colRange(bg_sbs.cols / 2, bg_sbs.cols).clone(), cv::Size(W, H), cv::INTER_LINEAR);
  Mat gbL = gray(down(bgL, ws)), gbR = gray(down(bgR, ws));
  auto brange = geo::disparity_range(gbL, gbR);
  Mat bd_l = geo::Stereo(brange.first, brange.second)(gbL, gbR).first;
  segment_eyes();
  Mat h0(h, w, CV_8U);
  CU(cudaMemcpy(h0.data, seg_out_->hard[0].p, n_, cudaMemcpyDeviceToHost));
  Mat dL0 = (*stereo_)(gL0, gR0).first;
  Mat core;
  cv::erode(h0, core, Mat::ones(prm_.core, prm_.core, CV_8U));
  double person_p5 = geo::nanpct(dL0, core, 5, 0.0) / ws_.s;
  init_person_disp_ = geo::nanpct(dL0, core, 50, 0.0);
  person_disp_ = (float)init_person_disp_;
  double bg_p95 = geo::nanpct(bd_l, Mat(), 95, 0.0) / ws_.s;
  bg_shift_ = std::max(0.0, bg_p95 - person_p5 + cfg_.bg_margin * W / 1920.0);
  if (bg_shift_ > 0) bgR = geo::warp_h(bgR, Mat(H, W, CV_32F, cv::Scalar((float)-bg_shift_)));
  {
    Mat both;
    cv::hconcat(bgL, bgR, both);
    CU(cudaMemcpy(bg_.p, both.data, frame_bytes, cudaMemcpyHostToDevice));
  }

  // GPU post-processing buffers
  for (int e = 0; e < 2; ++e) {
    gray_pin_[e].alloc(n_);
    gray_[e].alloc(n_);
    disp_pin_[e].alloc(n_);
    disp_[e].alloc(n_);
    for (auto& b : g_[e]) b.alloc(n_);
    for (auto& b : ema_[e]) b.alloc(n_);
    photo_[e].alloc(n_);
    valid_[e].alloc(n_);
    filled_[e].alloc(n_);
    band_[e].alloc(n_);
  }
  for (int i = 0; i < 4; ++i) {
    flow_pin_[i].alloc(2 * (size_t)n_);
    flow_dev_[i].alloc(2 * (size_t)n_);
  }
  ff_chw_.alloc(2 * n_);
  ff_at_.alloc(2 * n_);
  for (auto* b : {&prev_w_, &warped_, &scratch_, &soft_, &guide_, &q_, &neg_}) b->alloc(n_);
  for (auto* b : {&stack4_, &tmp4_, &box4_}) b->alloc(4 * (size_t)n_);
  ab_.alloc(2 * n_);
  box2_.alloc(2 * n_);
  alpha_.alloc(2 * n_);
  tmp8_.alloc(n_);
  core_.alloc(n_);
  dh_.alloc(n_);
  pad_.alloc(k::pitched_bytes(w, 4 * h) / sizeof(float));
  apitch_ = (w + 127) / 128 * 128;
  apad_.alloc((size_t)2 * h * apitch_);

  // half width of each row of OpenCV's ellipse (rows are runs centred on the middle column)
  Mat ell = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(prm_.ellipse, prm_.ellipse));
  ell_r_ = prm_.ellipse / 2;
  std::vector<int> hw(ell.rows);
  for (int y = 0; y < ell.rows; ++y) {
    int on = cv::countNonZero(ell.row(y));
    hw[y] = on ? on / 2 : -1;
  }
  half_w_.alloc(hw.size());
  CU(cudaMemcpy(half_w_.p, hw.data(), hw.size() * sizeof(int), cudaMemcpyHostToDevice));

  // picture in picture: top-right corner of each eye, (H / 8, W / 8), at the person's depth
  if (!cfg_.pip.empty()) {
    double disp = std::max(0.0, init_person_disp_ / ws_.s) + cfg_.bg_margin * W / 1920.0;
    pip_.on = 1;
    pip_.ph = H / 8;
    pip_.pw = W / 8;
    pip_.y0 = (int)std::nearbyint(H / 54.0);
    pip_.rgba_ready = 0;
    int x = W - pip_.pw - pip_.y0, d = (int)std::nearbyint(disp / 2.0);
    pip_.x0[0] = std::min(std::max(x + d, 0), W - pip_.pw);
    pip_.x0[1] = std::min(std::max(x - d, 0), W - pip_.pw);
    pip_from_frame_ = cfg_.pip == "frame";
    if (pip_from_frame_) {
      pip_.Hs = H;
      pip_.Ws = W;
      pip_.src_c = 3;
      pip_.src = frame_.p;
    } else {
      Mat icon = cv::imread(cfg_.pip, cv::IMREAD_UNCHANGED);
      if (icon.empty() || icon.channels() != 4) throw std::runtime_error("PiP icon must be an SBS RGBA png");
      pip_.Hs = icon.rows;
      pip_.Ws = icon.cols / 2;
      pip_.src_c = 4;
      pip_icon_.alloc(icon.total() * 4);
      CU(cudaMemcpy(pip_icon_.p, icon.data, icon.total() * 4, cudaMemcpyHostToDevice));
      pip_.src = pip_icon_.p;
    }
    pip_.has_alpha = !pip_from_frame_;
    pip_.src_row = 2 * pip_.Ws * pip_.src_c;
    upload_taps(area_taps(pip_.Ws, pip_.pw), area_taps(pip_.Hs, pip_.ph), pip_x_, pip_y_, false);
    pip_.x = pip_x_.taps();
    pip_.y = pip_y_.taps();
    pip_.K = pip_x_.K;
    if (seg_.mode == k::TEX_LINEAR) {
      pip_.wp = (pip_.Ws + 127) / 128 * 128;
      pip_rgba_.alloc((size_t)2 * pip_.Hs * pip_.wp * 4);
      pip_.rgba = pip_rgba_.p;
    }
  }
}

Pipeline::~Pipeline() {
  worker_.reset();
  gpool_.reset();
  cudaDeviceSynchronize();
  k::clear_textures();  // the cached texture objects point into buffers freed with this pipeline
}

// ------------------------------------------------------------------ CPU worker
Geometry Pipeline::geometry(const Mat& L, const Mat& R) {
  Geometry g;
  {
    Timer::Scope t(timer_, "geo.downscale", false);
    cv::Size ws(ws_.w, ws_.h);
    g.gL = gray(down(L, ws));
    g.gR = gray(down(R, ws));
  }
  const Mat pL = prev_gray_[0], pR = prev_gray_[1];
  g.has_flow = !pL.empty();
  if (geo_par_) {
    Timer::Scope t(timer_, "geo.parallel", false);  // left / right SGBM and the two flow pairs concurrently
    auto fr = gpool_->submit([&] { return stereo_r_->right(g.gL, g.gR); });
    std::future<void> ffl, ffr;
    if (g.has_flow) {
      ffl = gpool_->submit([&] { g.flow[0] = flow_(g.gL, pL); g.flow[1] = flow_(pL, g.gL); });
      ffr = gpool_->submit([&] { g.flow[2] = flow_r_(g.gR, pR); g.flow[3] = flow_r_(pR, g.gR); });
    }
    g.dL = stereo_->left(g.gL, g.gR);
    g.dR = fr.get();
    geo::Stereo::check(g.dL, g.dR);
    if (g.has_flow) {
      ffl.get();
      ffr.get();
    }
  } else {
    {
      Timer::Scope t(timer_, "geo.disparity", false);
      std::tie(g.dL, g.dR) = (*stereo_)(g.gL, g.gR);
    }
    Timer::Scope t(timer_, "geo.flow", false);
    if (g.has_flow) {
      g.flow[0] = flow_(g.gL, pL);
      g.flow[1] = flow_(pL, g.gL);
      g.flow[2] = flow_(g.gR, pR);
      g.flow[3] = flow_(pR, g.gR);
    }
  }
  prev_gray_[0] = g.gL;
  prev_gray_[1] = g.gR;
  upload_geometry(g);
  return g;
}

void Pipeline::upload_geometry(const Geometry& g) {
  Timer::Scope t(timer_, "geo.upload", false);
  upload(gray_[0], gray_pin_[0], g.gL, up_);
  upload(gray_[1], gray_pin_[1], g.gR, up_);
  upload(disp_[0], disp_pin_[0], g.dL, up_);
  upload(disp_[1], disp_pin_[1], g.dR, up_);
  if (g.has_flow)
    for (int i = 0; i < 4; ++i) {
      Mat flat = g.flow[i].reshape(1);  // (h, 2 w)
      upload(flow_dev_[i], flow_pin_[i], flat, up_);
    }
  ev_up_.record(up_);
}

// ------------------------------------------------------------------ GPU post-processing (gpu_post.py)
void Pipeline::dilate_band(const uint8_t* hard, const float* prob, uint8_t* band) {
  { OpScope op_("or_above", main_); k::or_above(hard, prob, 0.3f, tmp8_, n_, main_); }
  { OpScope op_("dilate_ellipse", main_); k::dilate_ellipse(tmp8_, dh_, half_w_, ell_r_, band, ws_.h, ws_.w, main_); }
}

void Pipeline::post(const Geometry& g) {
  const int w = ws_.w, h = ws_.h, n = n_;
  const k::Mode mode = seg_.mode;
  cudaStream_t s = main_;
  const std::string& dir = seg_.stage_dir();
  std::string t = "_" + std::to_string(frames_ - 1);
  PersonOut& o = *seg_out_;

  for (int e = 0; e < 2; ++e) { OpScope op_("u8_to_f32", s); k::u8_to_f32(gray_[e], g_[e][cur_], n, s); }
  bool flows = g.has_flow && have_prev_;
  if (flows)
    for (int e = 0; e < 2; ++e) {  // fw-bw consistency + photometric check
      const float *fb = flow_dev_[2 * e], *ff = flow_dev_[2 * e + 1];
      { OpScope op_("deinterleave2", s); k::deinterleave2(ff, ff_chw_, n, s); }
      { OpScope op_("warp", s); k::warp(mode, ff_chw_, 2, h, w, fb, fb + 1, 2, ff_at_, pad_, s); }
      { OpScope op_("warp", s); k::warp(mode, g_[e][1 - cur_], 1, h, w, fb, fb + 1, 2, prev_w_, pad_, s); }
      { OpScope op_("flow_validity", s); k::flow_validity(g_[e][cur_], fb, ff_at_, prev_w_, valid_[e], photo_[e], n, s); }
      if (!dir.empty()) {
        dump_dev(dir, "valid" + t + "_" + std::to_string(e), "u1", {h, w}, valid_[e].p, 1, s);
        dump_dev(dir, "photo" + t + "_" + std::to_string(e), "f4", {h, w}, photo_[e].p, 4, s);
      }
    }
  have_prev_ = true;

  // disparity fill: person disparity inside the dilated person band, median disparity elsewhere
  for (int e = 0; e < 2; ++e) dilate_band(o.hard[e], o.prob[e], band_[e]);
  { OpScope op_("erode", s); k::erode(o.hard[0], tmp8_, core_, h, w, prm_.core, s); }
  float pd = 0.f, dd = 0.f;
  {
    OpScope op_("nanmedian", s);
    if (k::nanmedian(disp_[0], core_, n, scratch_, &pd, s)) person_disp_ = pd;
    if (!k::nanmedian(disp_[0], nullptr, n, scratch_, &dd, s)) dd = 0.f;
  }
  for (int e = 0; e < 2; ++e) { OpScope op_("fill_disparity", s); k::fill_disparity(disp_[e], band_[e], person_disp_, dd, filled_[e], n, s); }

  float* tcur[2];
  for (int e = 0; e < 2; ++e) {
    // soft alpha + guided filter
    { OpScope op_("soft_alpha", s); k::soft_alpha(o.prob[e], cfg_.soft_lo, cfg_.soft_hi, soft_, n, s); }
    { OpScope op_("guided_stack", s); k::guided_stack(g_[e][cur_], soft_, guide_, stack4_, n, s); }
    { OpScope op_("box", s); k::box(mode, stack4_, tmp4_, box4_, 4, h, w, prm_.gf_r, pad_, s); }
    { OpScope op_("guided_ab", s); k::guided_ab(box4_, cfg_.gf_eps, ab_, n, s); }
    { OpScope op_("box", s); k::box(mode, ab_, tmp4_, box2_, 2, h, w, prm_.gf_r, pad_, s); }
    { OpScope op_("guided_out", s); k::guided_out(box2_, guide_, q_, n, s); }
    // motion-compensated temporal EMA
    float* prev = ema_[e][ema_cur_[e]];
    float* next = ema_[e][1 - ema_cur_[e]];
    if (have_ema_[e] && flows) {
      const float* fb = flow_dev_[2 * e];
      { OpScope op_("warp", s); k::warp(mode, prev, 1, h, w, fb, fb + 1, 2, prev_w_, pad_, s); }
      { OpScope op_("ema", s); k::ema(q_, prev_w_, valid_[e], photo_[e], cfg_.lam, cfg_.sigma, cfg_.gate, next, n, s); }
    } else {
      CU(cudaMemcpyAsync(next, q_.p, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    have_ema_[e] = true;
    ema_cur_[e] = 1 - ema_cur_[e];
    tcur[e] = next;
  }
  cur_ = 1 - cur_;

  // stereo fusion: average with the other eye's alpha warped through the filled disparity
  float *aL = alpha_, *aR = alpha_ + n;
  { OpScope op_("negate", s); k::negate(filled_[0], neg_, n, s); }
  { OpScope op_("warp", s); k::warp(mode, tcur[1], 1, h, w, neg_, nullptr, 1, warped_, pad_, s); }
  { OpScope op_("average", s); k::average(tcur[0], warped_, aL, n, s); }
  { OpScope op_("warp", s); k::warp(mode, tcur[0], 1, h, w, filled_[1], nullptr, 1, warped_, pad_, s); }
  { OpScope op_("average", s); k::average(tcur[1], warped_, aR, n, s); }

  if (!dir.empty())
    for (int e = 0; e < 2; ++e) {
      std::string tag = t + "_" + std::to_string(e);
      dump_dev(dir, "dfill" + tag, "f4", {h, w}, filled_[e].p, 4, s);
      dump_dev(dir, "tema" + tag, "f4", {h, w}, tcur[e], 4, s);
      dump_dev(dir, "alpha" + tag, "f4", {h, w}, e ? aR : aL, 4, s);
    }
}

// ------------------------------------------------------------------ frame
const uint8_t* Pipeline::process(const Mat& frame) {
  const int W = W_, H = H_;
  size_t row = (size_t)2 * W * 3, bytes = row * H;
  Mat L = frame.colRange(0, W), R = frame.colRange(W, 2 * W);
  ++frames_;
  auto geo_future = worker_->submit([&] { return geometry(L, R); });
  {
    Timer::Scope t(timer_, "gpu.upload_frame", false);
    for (int y = 0; y < H; ++y) std::memcpy(frame_pin_.p + y * row, frame.ptr(y), row);
    CU(cudaMemcpyAsync(frame_.p, frame_pin_.p, bytes, cudaMemcpyHostToDevice, main_));
  }
  {
    Timer::Scope t(timer_, "seg.preprocess", false);
    for (int e = 0; e < 2; ++e)
      seg_.preprocess(*in_, e, frame_.p + (size_t)e * W * 3, 2 * W * 3, frame_.p, frame_.n, main_);
  }
  for (int e = 0; e < 2; ++e) seg_.person(*in_, e, *seg_out_, main_, timer_);
  ev_seg_done_.record(main_);

  Geometry g;
  {
    Timer::Scope t(timer_, "geo.wait", false);
    g = geo_future.get();
    double late_ms = 0;
    if (cfg_.geo_threads > 1 && !geo_par_ && cudaEventQuery(ev_seg_done_.e) == cudaSuccess) {
      // the stream is idle: an event recorded now completes at once, its distance to seg_done is
      // the GPU idle time = the lateness of the geometry
      ev_now_.record(main_);
      CU(cudaEventSynchronize(ev_now_.e));
      float ms = 0;
      CU(cudaEventElapsedTime(&ms, ev_seg_done_.e, ev_now_.e));
      late_ms = ms;
    }
    if (frames_ > 10) late_ema_ = 0.9 * late_ema_ + 0.1 * late_ms;
    if (cfg_.geo_threads > 1 && !geo_par_ && late_ema_ > cfg_.geo_late_ms) geo_par_ = true;
  }
  const std::string& dir = seg_.stage_dir();
  if (!dir.empty()) {
    std::string t = "_" + std::to_string(frames_ - 1);
    const char* eye[2] = {"_0", "_1"};
    const Mat* gm[2] = {&g.gL, &g.gR};
    const Mat* dm[2] = {&g.dL, &g.dR};
    for (int e = 0; e < 2; ++e) {
      write_bin(dir, "gray" + t + eye[e], "u1", {ws_.h, ws_.w}, gm[e]->data, n_);
      write_bin(dir, "disp" + t + eye[e], "f4", {ws_.h, ws_.w}, dm[e]->data, n_ * 4);
      if (g.has_flow)
        for (int i = 0; i < 2; ++i)
          write_bin(dir, std::string(i ? "ff" : "fb") + t + eye[e], "f4", {ws_.h, ws_.w, 2}, g.flow[2 * e + i].data,
                    n_ * 8);
    }
  }
  {
    Timer::Scope t(timer_, "gpu.post", true);
    ev_up_.wait_in(main_);  // uploads ran on up_ while the GPU was segmenting
    post(g);
  }
  {
    Timer::Scope t(timer_, "gpu.compose+d2h", true);
    if (pip_from_frame_) pip_.src = frame_.p;
    { OpScope op_("compose", main_); k::compose(seg_.mode, frame_, bg_, alpha_, apad_, apitch_, out_, H, W, ws_.h, ws_.w, pip_, main_); }
    if (pip_.on && !pip_from_frame_) pip_.rgba_ready = 1;
    CU(cudaMemcpyAsync(out_pin_.p, out_.p, bytes, cudaMemcpyDeviceToHost, main_));
    CU(cudaStreamSynchronize(main_));
  }
  if (!dir.empty())
    write_bin(dir, "out_" + std::to_string(frames_ - 1), "u1", {H, 2 * W, 3}, out_pin_.p, bytes);
  return out_pin_.p;
}

void Pipeline::dump_outputs(const std::string& dir, int t) {
  const int w = ws_.w, h = ws_.h;
  char tag[16];
  std::snprintf(tag, sizeof tag, "_%04d", t);
  const char* eye[2] = {"L", "R"};
  for (int e = 0; e < 2; ++e) {
    dump_dev(dir, std::string("a") + eye[e] + tag, "f4", {h, w}, alpha_.p + (size_t)e * n_, 4, main_);
    dump_dev(dir, std::string("p") + eye[e] + tag, "f4", {h, w}, seg_out_->prob[e].p, 4, main_);
    dump_dev(dir, std::string("h") + eye[e] + tag, "u1", {h, w}, seg_out_->hard[e].p, 1, main_);
  }
}
