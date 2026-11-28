// CPU geometry: SGBM disparity, DIS optical flow and the background-depth helpers
// (temporal_stereo.py + the helpers of pipeline.py). Same OpenCV version as the Python wheel.
#pragma once
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/features.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/video/tracking.hpp>
#include <thread>
#include <vector>

namespace geo {

// numpy percentile (linear interpolation) of v; q in [0, 100]
inline double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  double pos = q / 100.0 * (v.size() - 1);
  size_t lo = (size_t)pos, hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (pos - lo) * (v[hi] - v[lo]);
}

// nanpercentile of a float map over mask (empty mask: everywhere); dflt when nothing is left
inline double nanpct(const cv::Mat& d, const cv::Mat& mask, double q, double dflt) {
  std::vector<double> v;
  for (int y = 0; y < d.rows; ++y)
    for (int x = 0; x < d.cols; ++x) {
      float f = d.at<float>(y, x);
      if (!std::isnan(f) && (mask.empty() || mask.at<uint8_t>(y, x))) v.push_back(f);
    }
  return v.empty() ? dflt : percentile(std::move(v), q);
}

struct Work {
  int w, h;
  double s;
};
inline Work work_size(int W, int H, int work_pixels = 960 * 540) {
  double s = std::min(1.0, std::sqrt(work_pixels / double(W) / H));
  return {(int)(W * s), (int)(H * s), s};
}
inline int odd(double x) { return std::max(1, (int)std::nearbyint((x - 1) / 2.0) * 2 + 1); }

struct Params {
  int ellipse, core, gf_r;
};
inline Params work_params(int h) {
  double k = h / 540.0;
  return {odd(31 * k), odd(9 * k), std::max(1, (int)std::nearbyint(4 * k))};
}

// (min_disp, num_disp) from ORB matches on a rectified pair
inline std::pair<int, int> disparity_range(const cv::Mat& l, const cv::Mat& r, int margin = 8) {
  auto orb = cv::ORB::create(4000);
  std::vector<cv::KeyPoint> kl, kr;
  cv::Mat dl, dr;
  orb->detectAndCompute(l, cv::noArray(), kl, dl);
  orb->detectAndCompute(r, cv::noArray(), kr, dr);
  if (dl.empty() || dr.empty()) return {-16, 64};
  std::vector<cv::DMatch> m;
  cv::BFMatcher(cv::NORM_HAMMING, true).match(dl, dr, m);
  std::vector<double> dx, dy;
  for (auto& x : m) {
    cv::Point2f a = kl[x.queryIdx].pt, b = kr[x.trainIdx].pt;
    dx.push_back((double)a.x - (double)b.x);
    dy.push_back((double)a.y - (double)b.y);
  }
  double med = percentile(dy, 50);
  std::vector<double> ok;
  for (size_t i = 0; i < dx.size(); ++i)
    if (std::abs(dy[i] - med) < 2) ok.push_back(dx[i]);
  double lo = percentile(ok, 1) - margin, hi = percentile(ok, 99) + margin;
  int min_d = (int)std::floor(lo);
  int num_d = (int)(std::ceil((hi - min_d) / 16.0) * 16);
  return {min_d, std::max(num_d, 16)};
}

// out(x) = img(x + dx(x)), border replicate
inline cv::Mat warp_h(const cv::Mat& img, const cv::Mat& dx, int interp = cv::INTER_LINEAR) {
  cv::Mat mx(dx.size(), CV_32F), my(dx.size(), CV_32F);
  for (int y = 0; y < dx.rows; ++y) {
    const float* d = dx.ptr<float>(y);
    float *px = mx.ptr<float>(y), *py = my.ptr<float>(y);
    for (int x = 0; x < dx.cols; ++x) {
      px[x] = (float)x + d[x];
      py[x] = (float)y;
    }
  }
  cv::Mat out;
  cv::remap(img, out, mx, my, interp, cv::BORDER_REPLICATE);
  return out;
}

class Stereo {
 public:
  Stereo(int min_disp, int num_disp, int block = 5) : min_disp_(min_disp) {
    sgbm_ = cv::StereoSGBM::create(min_disp, num_disp, block, 8 * block * block, 32 * block * block, 1, 0, 10, 100, 2,
                                   cv::StereoSGBM::MODE_SGBM_3WAY);
  }
  cv::Mat left(const cv::Mat& l, const cv::Mat& r) { return disp(l, r); }
  // matched on the mirrored pair
  cv::Mat right(const cv::Mat& l, const cv::Mat& r) {
    cv::Mat fl, fr, out;
    cv::flip(r, fr, 1);
    cv::flip(l, fl, 1);
    cv::flip(disp(fr, fl), out, 1);
    return out;
  }
  // left / right consistency check, in place
  static void check(cv::Mat& dl, cv::Mat& dr) {
    auto fill_nan = [](const cv::Mat& d, float v) {
      cv::Mat o = d.clone();
      o.forEach<float>([v](float& x, const int*) { if (std::isnan(x)) x = v; });
      return o;
    };
    cv::Mat neg_dl = -fill_nan(dl, 0.f);
    cv::Mat dr_at_l = warp_h(fill_nan(dr, -1e4f), neg_dl, cv::INTER_NEAREST);
    cv::Mat dl_at_r = warp_h(fill_nan(dl, -1e4f), fill_nan(dr, 0.f), cv::INTER_NEAREST);
    auto drop = [](cv::Mat& d, const cv::Mat& other) {
      for (int y = 0; y < d.rows; ++y) {
        float* p = d.ptr<float>(y);
        const float* o = other.ptr<float>(y);
        for (int x = 0; x < d.cols; ++x)
          if (!(std::abs(p[x] - o[x]) <= 1.5f)) p[x] = NAN;
      }
    };
    drop(dl, dr_at_l);
    drop(dr, dl_at_r);
  }
  std::pair<cv::Mat, cv::Mat> operator()(const cv::Mat& l, const cv::Mat& r) {
    cv::Mat dl = left(l, r), dr = right(l, r);
    check(dl, dr);
    return {dl, dr};
  }

 private:
  cv::Mat disp(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat d16, d;
    sgbm_->compute(a, b, d16);
    d16.convertTo(d, CV_32F, 1.0 / 16.0);
    float lo = (float)min_disp_;
    d.forEach<float>([lo](float& x, const int*) { if (x < lo) x = NAN; });
    return d;
  }
  int min_disp_;
  cv::Ptr<cv::StereoSGBM> sgbm_;
};

class Flow {
 public:
  Flow() : dis_(cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM)) {}
  cv::Mat operator()(const cv::Mat& a, const cv::Mat& b) {  // a(x) ~ b(x + flow(x))
    cv::Mat f;
    dis_->calc(a, b, f);
    return f;
  }

 private:
  cv::Ptr<cv::DISOpticalFlow> dis_;
};

// fixed-size thread pool (concurrent.futures.ThreadPoolExecutor)
class Pool {
 public:
  explicit Pool(int n) {
    for (int i = 0; i < n; ++i)
      threads_.emplace_back([this] {
        for (;;) {
          std::function<void()> job;
          {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
            if (q_.empty()) return;
            job = std::move(q_.front());
            q_.pop_front();
          }
          job();
        }
      });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }
  template <class F>
  auto submit(F f) -> std::future<decltype(f())> {
    auto task = std::make_shared<std::packaged_task<decltype(f())()>>(std::move(f));
    auto fut = task->get_future();
    {
      std::lock_guard<std::mutex> lk(m_);
      q_.emplace_back([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

 private:
  std::vector<std::thread> threads_;
  std::deque<std::function<void()>> q_;
  std::mutex m_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace geo
