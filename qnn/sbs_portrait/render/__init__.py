"""CUDA -> OpenGL / Vulkan zero-copy presentation of the composited SBS frame (Python half; the C++
half is cpp/render/). The bindings live outside the conda environment (setup_deps.sh)."""
import os
import sys

DEPS = os.environ.get("SBS_RENDER", os.path.expanduser("~/data/sbs/render"))
if os.path.isdir(os.path.join(DEPS, "pylib")):
    sys.path.append(os.path.join(DEPS, "pylib"))
SHADERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
