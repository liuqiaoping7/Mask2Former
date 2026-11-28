#!/usr/bin/env bash
# User-level rendering dependencies in $RENDER (default ~/data/sbs/render), no root needed:
# headers, GLFW, glslang and the Vulkan validation layer extracted from the distribution's
# packages into sysroot/, the Python bindings into pylib/. The GPU driver (OpenGL, Vulkan ICD)
# and the Vulkan loader are the system's.
set -euo pipefail
RENDER=${RENDER:-$HOME/data/sbs/render}
PY=${PY:-$HOME/anaconda3/envs/m2f/bin/python}
mkdir -p "$RENDER/debs" "$RENDER/sysroot"
cd "$RENDER/debs"
apt-get download libvulkan-dev libglfw3 libglfw3-dev libgl-dev libglx-dev libegl-dev libopengl-dev \
  glslang-tools vulkan-validationlayers spirv-tools vulkan-tools
for f in *.deb; do dpkg-deb -x "$f" "$RENDER/sysroot"; done
"$PY" -m pip install -q --upgrade --target "$RENDER/pylib" PyOpenGL glfw vulkan
echo "render dependencies in $RENDER"
