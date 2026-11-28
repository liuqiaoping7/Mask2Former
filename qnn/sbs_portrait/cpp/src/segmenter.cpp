#include "segmenter.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

Assets Assets::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path + " (run tools/export_assets.py)");
  std::map<std::string, std::string> kv;
  std::string line;
  while (std::getline(f, line)) {
    auto sp = line.find(' ');
    if (sp != std::string::npos) kv[line.substr(0, sp)] = line.substr(sp + 1);
  }
  Assets a;
  a.backbone = kv.at("backbone");
  a.pixel_decoder = kv.at("pixel_decoder");
  a.decoder = kv.at("decoder");
  a.icon = kv.at("icon");
  std::istringstream(kv.at("mean")) >> a.mean[0] >> a.mean[1] >> a.mean[2];
  std::istringstream(kv.at("std")) >> a.std_[0] >> a.std_[1] >> a.std_[2];
  a.object_mask_threshold = std::stof(kv.at("object_mask_threshold"));
  a.overlap_threshold = std::stof(kv.at("overlap_threshold"));
  a.min_size = std::stoi(kv.at("min_size"));
  a.max_size = std::stoi(kv.at("max_size"));
  a.size_divisibility = std::stoi(kv.at("size_divisibility"));
  a.person_id = std::stoi(kv.at("person_id"));
  return a;
}

// ------------------------------------------------------------------ resize weights (seg.py)
RawTaps area_taps(int src, int dst) {
  double scale = (double)src / dst;
  RawTaps taps(dst);
  for (int d = 0; d < dst; ++d) {
    double f1 = d * scale, f2 = f1 + scale, cw = std::min(scale, src - f1);
    int s1 = (int)std::ceil(f1), s2 = (int)std::floor(f2);
    s2 = std::min(s2, src - 1);
    s1 = std::min(s1, s2);
    if (s1 - f1 > 1e-3) taps[d].push_back({s1 - 1, (float)((s1 - f1) / cw)});
    for (int s = s1; s < s2; ++s) taps[d].push_back({s, (float)(1.0 / cw)});
    if (f2 - s2 > 1e-3) taps[d].push_back({s2, (float)(std::min(std::min(f2 - s2, 1.0), cw) / cw)});
  }
  return taps;
}

RawTaps linear_area_taps(int src, int dst) {
  double inv = (double)dst / src, scale = 1.0 / inv;
  RawTaps taps(dst);
  for (int d = 0; d < dst; ++d) {
    int sx = (int)std::floor(d * scale);
    float fx = (float)((d + 1) - (sx + 1) * inv);
    fx = fx <= 0 ? 0.f : fx - std::floor(fx);
    if (sx < 0) { sx = 0; fx = 0.f; }
    if (sx >= src - 1) { sx = src - 1; fx = 0.f; }
    int c0 = (int)std::nearbyintf((1.0f - fx) * 2048), c1 = (int)std::nearbyintf(fx * 2048);  // 11-bit fixed point
    taps[d] = {{sx, c0 / 2048.0f}, {std::min(sx + 1, src - 1), c1 / 2048.0f}};
  }
  return taps;
}

namespace {
template <class T>
void upload(Dev<T>& d, const std::vector<T>& h) {
  d.alloc(h.size());
  CU(cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
int max_len(const RawTaps& t) {
  size_t k = 0;
  for (auto& v : t) k = std::max(k, v.size());
  return (int)k;
}
void pack(const RawTaps& t, int K, DevTaps& d) {  // padded with zero-weight taps at the last index
  std::vector<int> idx;
  std::vector<float> w;
  for (auto& v : t)
    for (int k = 0; k < K; ++k) {
      idx.push_back(k < (int)v.size() ? v[k].first : v.back().first);
      w.push_back(k < (int)v.size() ? v[k].second : 0.f);
    }
  upload(d.i, idx);
  upload(d.w, w);
  d.K = K;
}
// pair adjacent taps (i, a), (i + 1, b) into one linear fetch at i + 0.5 + b / (a + b), weight a + b
void pack_merged(const RawTaps& t, int M, DevTaps& d) {
  std::vector<float> pos, w;
  for (auto& v : t) {
    float last = 0.f;
    for (int m = 0; m < M; ++m) {
      size_t j = 2 * m;
      if (j < v.size()) {
        int i = v[j].first, i1 = j + 1 < v.size() ? v[j + 1].first : i;
        double a = v[j].second, b = j + 1 < v.size() ? v[j + 1].second : 0.0, s = a + b;
        last = (float)(i + 0.5 + (s > 0 && i1 != i ? b / s : 0.0));
        pos.push_back(last);
        w.push_back((float)s);
      } else {
        pos.push_back(last);
        w.push_back(0.f);
      }
    }
  }
  upload(d.pos, pos);
  upload(d.mw, w);
  d.M = M;
}
}  // namespace

void upload_taps(const RawTaps& x, const RawTaps& y, DevTaps& dx, DevTaps& dy, bool merged) {
  int K = std::max(max_len(x), max_len(y));
  pack(x, K, dx);
  pack(y, K, dy);
  if (merged) {
    int M = std::max((max_len(x) + 1) / 2, (max_len(y) + 1) / 2);
    pack_merged(x, M, dx);
    pack_merged(y, M, dy);
  }
}

// ------------------------------------------------------------------ segmentation
Segmenter::Segmenter(const Assets& a, k::Mode m) : assets(a), mode(m) {
  chain_ = std::make_unique<TrtChain>(
      std::vector<std::string>{a.backbone, a.pixel_decoder, a.decoder},
      std::vector<std::map<std::string, std::string>>{
          {},
          {{"res2", "res2"}, {"res3", "res3"}, {"res4", "res4"}, {"res5", "res5"}},
          {{"ms0", "ms0"}, {"ms1", "ms1"}, {"ms2", "ms2"}, {"mf", "mask_features"}}});
}

std::unique_ptr<Input> Segmenter::make_input(int h, int w) {
  auto in = std::make_unique<Input>();
  in->h = h;
  in->w = w;
  // detectron2 ResizeShortestEdge.get_output_shape
  double scale = assets.min_size * 1.0 / std::min(h, w), nh, nw;
  if (h < w) { nh = assets.min_size; nw = scale * w; }
  else { nh = scale * h; nw = assets.min_size; }
  if (std::max(nh, nw) > assets.max_size) {
    double s = assets.max_size * 1.0 / std::max(nh, nw);
    nh *= s;
    nw *= s;
  }
  in->nw = (int)(nw + 0.5);
  in->nh = (int)(nh + 0.5);
  int div = assets.size_divisibility;
  in->Hp = (in->nh + div - 1) / div * div;
  in->Wp = (in->nw + div - 1) / div * div;
  bool down = in->nh <= h && in->nw <= w;  // cv::resize's dispatch
  upload_taps(down ? area_taps(w, in->nw) : linear_area_taps(w, in->nw),
              down ? area_taps(h, in->nh) : linear_area_taps(h, in->nh), in->tx, in->ty, mode == k::TEX_LINEAR);
  for (auto& d : in->data) {
    d.alloc((size_t)3 * in->Hp * in->Wp);
    CU(cudaMemset(d.p, 0, d.bytes()));  // zero padding; the resize writes only the (nh, nw) region
  }
  if (mode == k::TEX_LINEAR) {
    in->rgba_wp = (w + 127) / 128 * 128;
    in->rgba.alloc((size_t)h * in->rgba_wp * 4);
  }
  return in;
}

std::unique_ptr<PersonOut> Segmenter::make_output(int w, int h) {
  auto o = std::make_unique<PersonOut>();
  o->w = w;
  o->h = h;
  for (int e = 0; e < 2; ++e) {
    o->prob[e].alloc((size_t)w * h);
    o->hard[e].alloc((size_t)w * h);
  }
  o->ids.alloc((size_t)w * h);
  o->own_at.alloc((size_t)w * h);
  return o;
}

void Segmenter::preprocess(Input& in, int e, const uint8_t* eye, int row, const uint8_t* frame, size_t frame_bytes,
                           cudaStream_t s) {
  const Assets& a = assets;
  OpScope op_("resize", s);
  if (mode == k::TEX_LINEAR)
    k::resize_linear(eye, row, in.h, in.w, in.rgba, in.rgba_wp, in.tx.lin(), in.ty.lin(), in.tx.M, in.nh, in.nw, in.Hp,
                     in.Wp, a.mean, a.std_, in.data[e], s);
  else
    k::resize_taps(mode, eye, row, frame, frame_bytes, in.tx.taps(), in.ty.taps(), in.tx.K, in.nh, in.nw, in.Hp, in.Wp,
                   a.mean, a.std_, in.data[e], s);
}

void Segmenter::person(const Input& in, int e, PersonOut& out, cudaStream_t caller, Timer* tm) {
  wait_stream(stream_, caller, ev_in_);
  cudaStream_t s = stream_;
  std::map<std::string, ChainOutput> o;
  {
    Timer::Scope t(tm, "seg.model", true);
    OpScope op_("trt", s);
    o = (*chain_)(in.data[e], in.Hp, in.Wp, s);
  }
  {
    Timer::Scope t(tm, "seg.person", true);
    const ChainOutput &logits = o.at("pred_logits"), &masks = o.at("pred_masks");
    if (logits.type != nvinfer1::DataType::kFLOAT || masks.type != nvinfer1::DataType::kHALF)
      throw std::runtime_error("unexpected output dtype of the decoder engine");
    int Q = (int)logits.shape[1], C = (int)logits.shape[2];
    int H4 = (int)masks.shape[2], W4 = (int)masks.shape[3];
    { OpScope op_("query_weights", s); k::query_weights((const float*)logits.data, Q, C, assets.person_id, assets.object_mask_threshold, wp_, wd_, sk_,
                     flag_, s); }
    float sy = (float)(((double)in.nh / out.h) * ((double)H4 / in.Hp));
    float sx = (float)(((double)in.nw / out.w) * ((double)W4 / in.Wp));
    // the official geometry maps through the resized size (h, w) of the model input
    { OpScope op_("person", s); k::person(mode, masks.data, wp_, wd_, sk_, out.prob[e], out.ids, out.own_at, own_cnt_, Q, H4, W4, out.h, out.w, sy,
              sx, s); }
    { OpScope op_("panoptic_person", s); k::panoptic_person(out.ids, out.own_at, own_cnt_, flag_, assets.overlap_threshold, Q, out.w * out.h, hist_, valid_,
                       out.hard[e], s); }
    if (!stage_dir_.empty()) {
      std::string tag = "_" + std::to_string(frame_) + "_" + std::to_string(e);
      dump_dev(stage_dir_, "pre" + tag, "f4", {3, in.Hp, in.Wp}, in.data[e].p, 4, s);
      dump_dev(stage_dir_, "logits" + tag, "f4", {Q, C}, logits.data, 4, s);
      dump_dev(stage_dir_, "masks" + tag, "f2", {Q, H4, W4}, masks.data, 2, s);
      dump_dev(stage_dir_, "prob" + tag, "f4", {out.h, out.w}, out.prob[e].p, 4, s);
      dump_dev(stage_dir_, "hard" + tag, "u1", {out.h, out.w}, out.hard[e].p, 1, s);
    }
  }
  wait_stream(caller, stream_, ev_out_);
}
