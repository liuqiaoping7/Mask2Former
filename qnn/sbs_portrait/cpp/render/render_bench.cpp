// The render path alone (no inference): a producer thread writes pre-made device frames as fast as
// the queue allows, the main thread renders them. Measures zero-copy against host copy and checks
// that what the graphics API reads is bit-identical to what CUDA wrote (--check N).
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include "gpu.hpp"
#include "renderer.hpp"

int main(int argc, char** argv) try {
  render::Config c;
  int eye_w = 1920, eye_h = 1080, frames = 600;
  double hold_s = 0;
  c.vsync = false;
  c.shader_dir = SHADER_DIR;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto val = [&] {
      if (i + 1 >= argc) throw std::runtime_error("missing value of " + k);
      return std::string(argv[++i]);
    };
    if (k == "--api") c.api = val();
    else if (k == "--present") c.zerocopy = val() != "copy";
    else if (k == "--size") std::sscanf(val().c_str(), "%dx%d", &eye_w, &eye_h);
    else if (k == "--frames") frames = std::stoi(val());
    else if (k == "--vsync") c.vsync = std::stoi(val()) != 0;
    else if (k == "--queue") c.drop = val() == "drop";
    else if (k == "--slots") c.slots = std::stoi(val());
    else if (k == "--check") c.check = std::stoi(val());
    else if (k == "--window") std::sscanf(val().c_str(), "%dx%d", &c.win_w, &c.win_h);
    else if (k == "--validate") c.validate = true;
    else if (k == "--hold") hold_s = std::stod(val());  // producer pause before the last frame (screenshots)
    else throw std::runtime_error("unknown argument " + k);
  }
  c.width = 2 * eye_w;
  c.height = eye_h;
  const size_t bytes = (size_t)c.width * c.height * 3;
  const int K = 8;
  std::vector<std::unique_ptr<Dev<uint8_t>>> pat;
  std::vector<uint8_t> host(bytes);
  for (int f = 0; f < K; ++f) {  // B: left->right, G: top->bottom, R: checkerboard that moves with f
    for (int y = 0; y < c.height; ++y)
      for (int x = 0; x < c.width; ++x) {
        uint8_t* p = &host[((size_t)y * c.width + x) * 3];
        p[0] = (uint8_t)(x * 255 / (c.width - 1));
        p[1] = (uint8_t)(y * 255 / (c.height - 1));
        p[2] = ((x + f * 8) / 64 + y / 64) % 2 ? 200 : (uint8_t)((x * 7 + y * 3 + f) & 63);
      }
    pat.push_back(std::make_unique<Dev<uint8_t>>(bytes));
    CU(cudaMemcpy(pat.back()->p, host.data(), bytes, cudaMemcpyHostToDevice));
  }
  Dev<uint8_t> own(bytes);

  render::Presenter pres(c);
  std::fprintf(stderr, "%s\n", pres.description().c_str());
  double ms = 0;
  std::string error;
  std::thread producer([&] {
    try {
      Stream s;
      auto t0 = std::chrono::steady_clock::now();
      int warm = std::min(30, frames / 4);
      for (int f = 0; f < frames; ++f) {
        if (f == warm) t0 = std::chrono::steady_clock::now();
        if (hold_s > 0 && f == frames - 1) std::this_thread::sleep_for(std::chrono::duration<double>(hold_s));
        uint8_t* dst = nullptr;
        if (!pres.begin(&dst, s)) break;
        if (!dst) dst = own;
        CU(cudaMemcpyAsync(dst, pat[f % K]->p, bytes, cudaMemcpyDeviceToDevice, s));  // stands in for compose
        pres.end(dst, s);
        CU(cudaStreamSynchronize(s));
      }
      ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / (frames - warm);
    } catch (const std::exception& e) {
      error = e.what();
    }
    pres.finish();
  });
  pres.run();
  producer.join();
  if (!error.empty()) throw std::runtime_error(error);
  std::printf("{\"size\": \"%dx%d\", \"frames\": %d, \"producer_ms_per_frame\": %.4f, \"render\": %s}\n", c.width,
              c.height, frames, ms, pres.stats_json().c_str());
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
