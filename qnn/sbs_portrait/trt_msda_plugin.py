"""Build / load the TensorRT plugin that wraps the project's own MSDeformAttn CUDA kernel
(trt_plugin/msda_upstream_plugin.cu, plugin name "MSDeformAttnUpstream_TRT").

  load()   # compiles weights/libmsda_upstream_trt.so on first use, dlopen(RTLD_GLOBAL) registers it

Build needs nvcc (CUDA 12.8) and the TensorRT C++ headers of the installed version (TRT_INCLUDE,
default ~/tools/trt-oss-10.13.3/include: the include/ directory of NVIDIA/TensorRT tag v10.13.3; the
pip package ships only the libraries). Links against the pip package's libnvinfer.so.10.
"""
import ctypes
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(HERE, "trt_plugin", "msda_upstream_plugin.cu")
LIB = os.path.join(REPO, "weights", "libmsda_upstream_trt.so")
OPS = os.path.join(REPO, "mask2former", "modeling", "pixel_decoder", "ops", "src", "cuda")
PLUGIN_NAME = "MSDeformAttnUpstream_TRT"
_loaded = False


def build(arch="sm_89"):
    import tensorrt_libs

    libdir = os.path.dirname(tensorrt_libs.__file__)
    inc = os.environ.get("TRT_INCLUDE", os.path.expanduser("~/tools/trt-oss-10.13.3/include"))
    nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8"), "bin", "nvcc")
    cmd = [nvcc, "-O3", "-std=c++17", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC",
           "-I", inc, "-I", os.path.join(HERE, "trt_plugin", "stub"), "-I", OPS, SRC,
           "-L", libdir, "-l:libnvinfer.so.10", f"-Xlinker=-rpath={libdir}", "-o", LIB]
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    subprocess.run(cmd, check=True)
    return LIB


def load():
    """Register the plugin creator (idempotent). Import tensorrt first so the same libnvinfer is used."""
    global _loaded
    if _loaded:
        return
    import tensorrt  # noqa: F401  (loads libnvinfer.so.10 from the pip package)

    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        build()
    ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
    _loaded = True
