// Runner (run_pipeline.py / run_multires.py): one or several SBS sequences in one process, decode
// prefetched, latency / module / memory statistics. No preview encoding (compare with the Python
// runners under --no-preview).
#include <dirent.h>
#include <nvml.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <cstring>
#include <future>
#include <opencv2/imgcodecs.hpp>
#include <thread>
#include "pipeline.hpp"

namespace {

struct Args {
  std::string frames, sets, sizes, bg, out, assets, kernel = "cuda", pip, key_frames = "0,75,149", stage_dir;
  int max_frames = 150, skip = 5, stage_frames = 0, geo_threads = 4;
  bool profile = false, dump = false, op_times = false;
};

class MemMonitor {  // samples this process's GPU memory (NVML) and RSS every 10 ms
 public:
  MemMonitor() {
    nvmlInit_v2();
    nvmlDeviceGetHandleByIndex_v2(0, &dev_);
    t_ = std::thread([this] {
      while (!stop_) {
        size_t g = gpu_used(), r = rss();
        if (g > gpu_peak) gpu_peak = g;
        if (r > rss_peak) rss_peak = r;
        usleep(10000);
      }
    });
  }
  ~MemMonitor() {
    stop_ = true;
    t_.join();
    nvmlShutdown();
  }
  void reset() {
    gpu_peak = gpu_used();
    rss_peak = rss();
  }
  std::atomic<size_t> gpu_peak{0}, rss_peak{0};

 private:
  size_t gpu_used() {
    nvmlProcessInfo_t info[64];
    unsigned n = 64;
    if (nvmlDeviceGetComputeRunningProcesses_v3(dev_, &n, info) != NVML_SUCCESS) return 0;
    for (unsigned i = 0; i < n; ++i)
      if (info[i].pid == (unsigned)getpid()) return info[i].usedGpuMemory;
    return 0;
  }
  static size_t rss() {
    std::ifstream f("/proc/self/statm");
    size_t total = 0, res = 0;
    f >> total >> res;
    return res * sysconf(_SC_PAGESIZE);
  }
  nvmlDevice_t dev_;
  std::atomic<bool> stop_{false};
  std::thread t_;
};

std::vector<std::string> jpgs(const std::string& dir) {
  std::vector<std::string> out;
  DIR* d = opendir(dir.c_str());
  if (!d) throw std::runtime_error("cannot open " + dir);
  while (auto* e = readdir(d)) {
    std::string n = e->d_name;
    if (n.size() > 4 && n.substr(n.size() - 4) == ".jpg") out.push_back(dir + "/" + n);
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, ','))
    if (!x.empty()) out.push_back(x);
  return out;
}

double unix_time() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

double pct(std::vector<double> v, double q) { return geo::percentile(std::move(v), q); }

// returns the stats JSON of one sequence
std::string run_sequence(Segmenter& seg, const Args& a, const std::vector<std::string>& files, const cv::Mat& bg,
                         const std::string& out, MemMonitor& mon, const PipelineConfig& cfg, double& first_unix) {
  mkdir(out.c_str(), 0755);
  if (a.dump) mkdir((out + "/dump").c_str(), 0755);
  Timer timer(a.profile);
  mon.reset();
  double t_init = now_ms();
  Pipeline pipe(seg, cv::imread(files[0]), bg, &timer, cfg);
  double init_s = (now_ms() - t_init) / 1000.0;
  if (!a.stage_dir.empty()) mkdir(a.stage_dir.c_str(), 0755);

  std::vector<std::string> keys = split(a.key_frames);
  auto decode = [](std::string f) { return cv::imread(f); };
  auto next = std::async(std::launch::async, decode, files[0]);
  std::vector<double> ms;
  g_ops.on = a.op_times;  // after the pipeline's init, which is not a frame
  g_ops.ms.clear();
  for (size_t t = 0; t < files.size(); ++t) {
    if (t) timer.new_frame();
    seg.set_dump((int)t < a.stage_frames ? a.stage_dir : "", (int)t);
    double t0 = now_ms();
    cv::Mat frame;
    {
      Timer::Scope s(&timer, "read", false);
      frame = next.get();
      if (t + 1 < files.size()) next = std::async(std::launch::async, decode, files[t + 1]);
    }
    const uint8_t* res = pipe.process(frame);
    CU(cudaDeviceSynchronize());
    ms.push_back(now_ms() - t0);
    if (a.op_times) g_ops.end_frame((int)t >= a.skip);
    if (first_unix == 0) first_unix = unix_time();
    if (a.dump) pipe.dump_outputs(out + "/dump", (int)t);
    for (auto& k : keys)
      if (std::stoi(k) == (int)t) {
        char name[32];
        std::snprintf(name, sizeof name, "/key_%04zu.png", t);
        cv::imwrite(out + name, cv::Mat(pipe.H(), 2 * pipe.W(), CV_8UC3, (void*)res));
      }
  }
  seg.set_dump("", 0);
  g_ops.on = false;
  std::vector<double> fm(ms.begin() + std::min<size_t>(a.skip, ms.size() - 1), ms.end());
  double mean = std::accumulate(fm.begin(), fm.end(), 0.0) / fm.size();
  std::ostringstream j;
  j << "{\"frames\": " << files.size() << ", \"kernel\": \"" << a.kernel << "\", \"pip\": \"" << cfg.pip
    << "\", \"profile_sync\": " << (a.profile ? "true" : "false") << ", \"eye_wh\": [" << pipe.W() << ", " << pipe.H()
    << "], \"work_wh\": [" << pipe.work().w << ", " << pipe.work().h << "], \"model_input_hw\": ["
    << pipe.model_input_hw().first << ", " << pipe.model_input_hw().second << "], \"padded_hw\": ["
    << pipe.padded_hw().first << ", " << pipe.padded_hw().second << "], \"geo_parallel\": "
    << (pipe.geo_parallel() ? "true" : "false") << ", \"person_disp\": " << pipe.init_person_disp()
    << ", \"bg_shift\": " << pipe.bg_shift() << ", \"e2e_ms\": {\"median\": " << median(fm) << ", \"mean\": " << mean
    << ", \"p95\": " << pct(fm, 95) << ", \"first\": " << ms[0] << "}, \"modules_ms\": {";
  bool comma = false;
  for (auto& [k, v] : timer.summary(a.skip)) {
    j << (comma ? ", " : "") << "\"" << k << "\": " << v;
    comma = true;
  }
  j << "}, \"op_ms\": {";
  comma = false;
  for (auto& [k, v] : g_ops.ms) {
    j << (comma ? ", " : "") << "\"" << k << "\": " << median(v);
    comma = true;
  }
  j << "}, \"init_s\": " << init_s << ", \"gpu_mem_peak_mb_nvml\": " << mon.gpu_peak / 1048576.0
    << ", \"rss_peak_mb\": " << mon.rss_peak / 1048576.0 << ", \"arena_mb\": " << seg.arena_bytes() / 1048576.0 << "}";
  std::ofstream(out + "/stats.json") << j.str() << "\n";
  return j.str();
}

}  // namespace

int main(int argc, char** argv) try {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto val = [&] {
      if (i + 1 >= argc) throw std::runtime_error("missing value of " + k);
      return std::string(argv[++i]);
    };
    if (k == "--frames") a.frames = val();
    else if (k == "--sets") a.sets = val();
    else if (k == "--sizes") a.sizes = val();
    else if (k == "--bg") a.bg = val();
    else if (k == "--out") a.out = val();
    else if (k == "--assets") a.assets = val();
    else if (k == "--kernel") a.kernel = val();
    else if (k == "--pip") a.pip = val();
    else if (k == "--max-frames") a.max_frames = std::stoi(val());
    else if (k == "--skip") a.skip = std::stoi(val());
    else if (k == "--key-frames") a.key_frames = val();
    else if (k == "--geo-threads") a.geo_threads = std::stoi(val());
    else if (k == "--stage-dir") a.stage_dir = val();
    else if (k == "--stage-frames") a.stage_frames = std::stoi(val());
    else if (k == "--profile") a.profile = true;
    else if (k == "--dump") a.dump = true;
    else if (k == "--op-times") a.op_times = true;
    else throw std::runtime_error("unknown argument " + k);
  }
  if ((a.frames.empty() == a.sets.empty()) || a.bg.empty() || a.out.empty() || a.assets.empty())
    throw std::runtime_error("need --assets, --bg, --out and one of --frames DIR | --sets DIR --sizes WxH,...");
  const std::map<std::string, k::Mode> modes{{"cuda", k::GLOBAL}, {"tex_point", k::TEX_POINT}, {"tex_linear", k::TEX_LINEAR}};

  MemMonitor mon;
  mkdir(a.out.c_str(), 0755);
  Assets assets = Assets::load(a.assets);
  double t0 = now_ms();
  Segmenter seg(assets, modes.at(a.kernel));
  double seg_init_s = (now_ms() - t0) / 1000.0;
  cv::Mat bg = cv::imread(a.bg);
  if (bg.empty()) throw std::runtime_error("cannot read " + a.bg);
  PipelineConfig cfg;
  cfg.geo_threads = a.geo_threads;
  cfg.pip = a.pip == "icon" ? assets.icon : a.pip;

  std::vector<std::string> seqs = a.sets.empty() ? std::vector<std::string>{""} : split(a.sizes);
  double first_unix = 0;
  std::ostringstream summary;
  summary << "{\"seg_init_s\": " << seg_init_s << ", \"sizes\": {";
  for (size_t i = 0; i < seqs.size(); ++i) {
    std::string name = seqs[i];
    auto files = jpgs(name.empty() ? a.frames : a.sets + "/" + name);
    if ((int)files.size() > a.max_frames) files.resize(a.max_frames);
    if (files.empty()) throw std::runtime_error("no frames");
    std::string st = run_sequence(seg, a, files, bg, name.empty() ? a.out : a.out + "/" + name, mon, cfg, first_unix);
    std::printf("%s %s\n", name.c_str(), st.c_str());
    std::fflush(stdout);
    summary << (i ? ", " : "") << "\"" << (name.empty() ? "seq" : name) << "\": " << st;
  }
  rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  summary.precision(15);
  summary << "}, \"first_frame_unix\": " << std::fixed << first_unix << ", \"ru_maxrss_mb\": " << ru.ru_maxrss / 1024.0
          << "}";
  std::ofstream(a.out + "/summary.json") << summary.str() << "\n";
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
