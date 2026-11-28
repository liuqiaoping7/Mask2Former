#!/usr/bin/env bash
# User-level build dependencies in $DEPS (default ~/data/sbs/cpp): cmake + ninja in a private venv,
# OpenCV 5.1.0 (same version as the Python wheel, IPP + pthreads, same CPU dispatch), then the
# pipeline itself. Needs: g++, CUDA 12.8 (nvcc), TensorRT 10.13.3 headers (~/tools/trt-oss-10.13.3)
# and the libraries of the tensorrt wheel (see TRT_INCLUDE / TRT_LIBDIR in CMakeLists.txt).
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
DEPS=${DEPS:-$HOME/data/sbs/cpp}
CUDA=${CUDA_HOME:-/usr/local/cuda-12.8}
mkdir -p "$DEPS" && cd "$DEPS"
[ -x venv/bin/pip ] || python3 -m venv venv || "$HOME/anaconda3/envs/m2f/bin/python" -m venv venv
venv/bin/pip install -q cmake ninja
export PATH=$DEPS/venv/bin:$CUDA/bin:/usr/bin:/bin
[ -d opencv ] || git clone --depth 1 -b 5.1.0 git@github.com:opencv/opencv.git </dev/null
if [ ! -d opencv-install ]; then
  cmake -S opencv -B opencv-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$DEPS/opencv-install" \
    -DBUILD_LIST=core,imgproc,imgcodecs,geometry,features,flann,stereo,video \
    -DBUILD_SHARED_LIBS=ON -DWITH_IPP=ON -DWITH_PTHREADS_PF=ON -DWITH_OPENMP=OFF -DWITH_TBB=OFF \
    -DCPU_BASELINE=SSE3 -DCPU_DISPATCH=SSE4_1,SSE4_2,AVX,FP16,AVX2,AVX_VNNI,AVX512_SKX,AVX512_CLX \
    -DBUILD_JPEG=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF
  cmake --build opencv-build -j 8
  cmake --install opencv-build > install.log
fi
cmake -S "$SRC" -B build -G Ninja -DCMAKE_PREFIX_PATH="$DEPS/opencv-install" \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA"
cmake --build build -j 8
echo "built $DEPS/build/sbs_portrait"
