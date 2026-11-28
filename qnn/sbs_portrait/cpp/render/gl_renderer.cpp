// OpenGL 4.5 backend (render/gl_renderer.py).
//   per slot: a shader storage buffer (SSBO) registered with CUDA (cudaGraphicsGLRegisterBuffer).
// Ownership of a slot alternates between the two APIs:
//   mapped   -> CUDA may write (the compose kernel stores straight into the GL buffer)
//   unmapped -> OpenGL may read (the fragment shader indexes the same memory)
// Only this (consumer) thread has the GL context, so it maps / unmaps; the producer thread orders
// its stream against those operations with CUDA events, without any CPU wait.
#define GL_GLEXT_PROTOTYPES 0
#include <GL/glcorearb.h>
#include <cuda_gl_interop.h>
#include <cstring>
#include <vector>
#include "gpu.hpp"
#include "renderer.hpp"
#include "shader_file.hpp"

namespace render {
namespace {

#define GL_FUNCS(F)                                                                                                    \
  F(PFNGLGETSTRINGPROC, GetString) F(PFNGLCREATESHADERPROC, CreateShader) F(PFNGLSHADERSOURCEPROC, ShaderSource)       \
  F(PFNGLCOMPILESHADERPROC, CompileShader) F(PFNGLGETSHADERIVPROC, GetShaderiv)                                        \
  F(PFNGLGETSHADERINFOLOGPROC, GetShaderInfoLog) F(PFNGLDELETESHADERPROC, DeleteShader)                                \
  F(PFNGLCREATEPROGRAMPROC, CreateProgram) F(PFNGLATTACHSHADERPROC, AttachShader) F(PFNGLLINKPROGRAMPROC, LinkProgram) \
  F(PFNGLGETPROGRAMIVPROC, GetProgramiv) F(PFNGLGETPROGRAMINFOLOGPROC, GetProgramInfoLog)                              \
  F(PFNGLDELETEPROGRAMPROC, DeleteProgram) F(PFNGLUSEPROGRAMPROC, UseProgram) F(PFNGLUNIFORM2IPROC, Uniform2i)         \
  F(PFNGLGENVERTEXARRAYSPROC, GenVertexArrays) F(PFNGLBINDVERTEXARRAYPROC, BindVertexArray)                            \
  F(PFNGLDELETEVERTEXARRAYSPROC, DeleteVertexArrays) F(PFNGLGENBUFFERSPROC, GenBuffers)                                \
  F(PFNGLBINDBUFFERPROC, BindBuffer) F(PFNGLBUFFERDATAPROC, BufferData) F(PFNGLBUFFERSUBDATAPROC, BufferSubData)       \
  F(PFNGLBINDBUFFERBASEPROC, BindBufferBase) F(PFNGLDELETEBUFFERSPROC, DeleteBuffers)                                  \
  F(PFNGLVIEWPORTPROC, Viewport) F(PFNGLCLEARPROC, Clear) F(PFNGLCLEARCOLORPROC, ClearColor)                           \
  F(PFNGLDRAWARRAYSPROC, DrawArrays) F(PFNGLDISABLEPROC, Disable) F(PFNGLGENFRAMEBUFFERSPROC, GenFramebuffers)         \
  F(PFNGLBINDFRAMEBUFFERPROC, BindFramebuffer) F(PFNGLGENRENDERBUFFERSPROC, GenRenderbuffers)                          \
  F(PFNGLBINDRENDERBUFFERPROC, BindRenderbuffer) F(PFNGLRENDERBUFFERSTORAGEPROC, RenderbufferStorage)                  \
  F(PFNGLFRAMEBUFFERRENDERBUFFERPROC, FramebufferRenderbuffer)                                                         \
  F(PFNGLCHECKFRAMEBUFFERSTATUSPROC, CheckFramebufferStatus) F(PFNGLDELETEFRAMEBUFFERSPROC, DeleteFramebuffers)        \
  F(PFNGLDELETERENDERBUFFERSPROC, DeleteRenderbuffers) F(PFNGLREADPIXELSPROC, ReadPixels)                              \
  F(PFNGLPIXELSTOREIPROC, PixelStorei) F(PFNGLGENQUERIESPROC, GenQueries) F(PFNGLBEGINQUERYPROC, BeginQuery)           \
  F(PFNGLENDQUERYPROC, EndQuery) F(PFNGLGETQUERYOBJECTUI64VPROC, GetQueryObjectui64v)                                  \
  F(PFNGLDELETEQUERIESPROC, DeleteQueries)

struct Api {  // OpenGL entry points come from the driver at run time (what GLEW / GLAD generate)
#define DECL(T, N) T N = nullptr;
  GL_FUNCS(DECL)
#undef DECL
  void load() {
#define LOAD(T, N)                                                 \
  N = reinterpret_cast<T>(glfwGetProcAddress("gl" #N));            \
  if (!N) throw std::runtime_error("missing OpenGL function gl" #N);
    GL_FUNCS(LOAD)
#undef LOAD
  }
};

class GlBackend : public Backend {
 public:
  GlBackend(const Config& c, GLFWwindow* win) : c_(c), win_(win), bytes_((size_t)c.width * c.height * 3) {
    glfwMakeContextCurrent(win_);
    glfwSwapInterval(c.vsync ? 1 : 0);
    gl.load();
    unsigned n = 0;
    int dev = -1;
    // the context must live on the GPU CUDA computes on, otherwise "interop" copies through the host
    CU(cudaGLGetDevices(&n, &dev, 1, cudaGLDeviceListAll));
    if (!n) throw std::runtime_error("the OpenGL context is not on a CUDA device");

    prog_ = gl.CreateProgram();
    GLuint vs = compile(GL_VERTEX_SHADER, c.shader_dir + "/sbs.vert");
    GLuint fs = compile(GL_FRAGMENT_SHADER, c.shader_dir + "/sbs.frag");
    gl.AttachShader(prog_, vs);
    gl.AttachShader(prog_, fs);
    gl.LinkProgram(prog_);
    GLint ok = 0;
    gl.GetProgramiv(prog_, GL_LINK_STATUS, &ok);
    if (!ok) {
      char log[2048];
      gl.GetProgramInfoLog(prog_, sizeof log, nullptr, log);
      throw std::runtime_error(std::string("link: ") + log);
    }
    gl.DeleteShader(vs);
    gl.DeleteShader(fs);
    gl.GenVertexArrays(1, &vao_);  // empty: the core profile needs one bound to draw
    gl.Disable(GL_DITHER);         // keep 8-bit values exact

    CU(cudaStreamCreateWithFlags(&map_stream_, cudaStreamNonBlocking));
    slots_.resize(c.slots);
    for (auto& s : slots_) {
      gl.GenBuffers(1, &s.buf);
      gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, s.buf);
      gl.BufferData(GL_SHADER_STORAGE_BUFFER, bytes_, nullptr, GL_DYNAMIC_DRAW);
      gl.GenQueries(1, &s.query);
      if (c.zerocopy) {
        CU(cudaGraphicsGLRegisterBuffer(&s.res, s.buf, cudaGraphicsRegisterFlagsWriteDiscard));
        CU(cudaEventCreateWithFlags(&s.written, cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&s.mapped, cudaEventDisableTiming));
        map(s);  // start out owned by CUDA
      }
    }
    if (c.check) {  // offscreen target at source size for the read-back check
      gl.GenFramebuffers(1, &fbo_);
      gl.GenRenderbuffers(1, &rbo_);
      gl.BindRenderbuffer(GL_RENDERBUFFER, rbo_);
      gl.RenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, c.width, c.height);
      gl.BindFramebuffer(GL_FRAMEBUFFER, fbo_);
      gl.FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo_);
      if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("incomplete framebuffer");
      gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
    }
  }

  ~GlBackend() override {
    for (auto& s : slots_) {
      if (s.res) {
        cudaGraphicsUnmapResources(1, &s.res, map_stream_);
        cudaGraphicsUnregisterResource(s.res);
        cudaEventDestroy(s.written);
        cudaEventDestroy(s.mapped);
      }
      gl.DeleteBuffers(1, &s.buf);
      gl.DeleteQueries(1, &s.query);
    }
    if (fbo_) {
      gl.DeleteFramebuffers(1, &fbo_);
      gl.DeleteRenderbuffers(1, &rbo_);
    }
    gl.DeleteVertexArrays(1, &vao_);
    gl.DeleteProgram(prog_);
    cudaStreamDestroy(map_stream_);
  }

  // ---- producer thread
  uint8_t* cuda_begin(int i, cudaStream_t s) override {
    CU(cudaStreamWaitEvent(s, slots_[i].mapped, 0));  // after the map, hence after OpenGL's reads
    return slots_[i].ptr;
  }
  void cuda_end(int i, cudaStream_t s) override { CU(cudaEventRecord(slots_[i].written, s)); }

  // ---- consumer thread
  void draw(int i, const uint8_t* host, uint8_t* rgba) override {
    Slot& s = slots_[i];
    if (c_.zerocopy) {
      CU(cudaStreamWaitEvent(map_stream_, s.written, 0));
      CU(cudaGraphicsUnmapResources(1, &s.res, map_stream_));  // OpenGL now sees the CUDA writes
    } else {
      gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, s.buf);
      gl.BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes_, host);  // the upload zero-copy avoids
    }
    gl.UseProgram(prog_);
    gl.Uniform2i(0, c_.width, c_.height);
    gl.BindVertexArray(vao_);
    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s.buf);

    if (rgba) {
      gl.BindFramebuffer(GL_FRAMEBUFFER, fbo_);
      gl.Viewport(0, 0, c_.width, c_.height);
      gl.DrawArrays(GL_TRIANGLES, 0, 3);
      gl.PixelStorei(GL_PACK_ALIGNMENT, 1);
      std::vector<uint8_t> up((size_t)c_.width * c_.height * 4);
      gl.ReadPixels(0, 0, c_.width, c_.height, GL_RGBA, GL_UNSIGNED_BYTE, up.data());
      size_t row = (size_t)c_.width * 4;  // OpenGL returns the bottom row first
      for (int y = 0; y < c_.height; ++y) std::memcpy(rgba + y * row, up.data() + (c_.height - 1 - y) * row, row);
      gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    if (s.timed) {  // result of the previous use of this slot: long finished
      GLuint64 ns = 0;
      gl.GetQueryObjectui64v(s.query, GL_QUERY_RESULT, &ns);
      gpu_ms_ = ns * 1e-6;
    }
    int w, h;
    glfwGetFramebufferSize(win_, &w, &h);
    gl.Viewport(0, 0, w, h);
    gl.ClearColor(0, 0, 0, 1);
    gl.Clear(GL_COLOR_BUFFER_BIT);
    double k = std::min((double)w / c_.width, (double)h / c_.height);  // letterbox
    int vw = (int)(c_.width * k), vh = (int)(c_.height * k);
    gl.Viewport((w - vw) / 2, (h - vh) / 2, vw, vh);
    gl.BeginQuery(GL_TIME_ELAPSED, s.query);
    gl.DrawArrays(GL_TRIANGLES, 0, 3);
    gl.EndQuery(GL_TIME_ELAPSED);
    s.timed = true;
    glfwSwapBuffers(win_);
    if (c_.zerocopy) map(s);  // give the slot back to CUDA once OpenGL's draw has read it
  }

  double gpu_draw_ms() const override { return gpu_ms_; }
  std::string description() const override {
    return std::string("OpenGL ") + (const char*)gl.GetString(GL_VERSION) + " on " + (const char*)gl.GetString(GL_RENDERER);
  }

 private:
  struct Slot {
    GLuint buf = 0, query = 0;
    cudaGraphicsResource_t res = nullptr;
    uint8_t* ptr = nullptr;
    cudaEvent_t written = nullptr, mapped = nullptr;
    bool timed = false;
  };

  void map(Slot& s) {
    // guarantees: GL commands issued before this call finish before CUDA work queued after it on map_stream_
    CU(cudaGraphicsMapResources(1, &s.res, map_stream_));
    size_t n = 0;
    CU(cudaGraphicsResourceGetMappedPointer((void**)&s.ptr, &n, s.res));  // may move between maps
    CU(cudaEventRecord(s.mapped, map_stream_));
  }

  GLuint compile(GLenum type, const std::string& path) {
    std::string src = read_file(path);
    const char* p = src.c_str();
    GLuint sh = gl.CreateShader(type);
    gl.ShaderSource(sh, 1, &p, nullptr);
    gl.CompileShader(sh);
    GLint ok = 0;
    gl.GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
      char log[2048];
      gl.GetShaderInfoLog(sh, sizeof log, nullptr, log);
      throw std::runtime_error(path + ": " + log);
    }
    return sh;
  }

  Config c_;
  GLFWwindow* win_;
  size_t bytes_;
  Api gl;
  GLuint prog_ = 0, vao_ = 0, fbo_ = 0, rbo_ = 0;
  cudaStream_t map_stream_ = nullptr;
  std::vector<Slot> slots_;
  double gpu_ms_ = -1;
};

}  // namespace

void gl_window_hints() {
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

std::unique_ptr<Backend> make_gl(const Config& c, GLFWwindow* win) { return std::make_unique<GlBackend>(c, win); }

}  // namespace render
