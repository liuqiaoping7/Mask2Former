cmake_minimum_required(VERSION 3.24)
project(sbs_portrait_cpp LANGUAGES CXX CUDA)

set(TRT_INCLUDE "$ENV{HOME}/tools/trt-oss-10.13.3/include" CACHE PATH "TensorRT headers")
set(TRT_LIBDIR "$ENV{HOME}/anaconda3/envs/m2f/lib/python3.12/site-packages/tensorrt_libs" CACHE PATH "TensorRT libraries")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(OpenCV 5.1 REQUIRED COMPONENTS core imgproc imgcodecs geometry features stereo video)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_executable(sbs_portrait
  src/main.cpp src/pipeline.cpp src/segmenter.cpp src/trt_chain.cpp
  kernels/resize.cu kernels/person.cu kernels/post.cu kernels/compose.cu)
target_include_directories(sbs_portrait PRIVATE src kernels ${TRT_INCLUDE} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sbs_portrait PRIVATE ${OpenCV_LIBS} CUDA::cudart CUDA::nvml Threads::Threads
  ${TRT_LIBDIR}/libnvinfer.so.10 ${TRT_LIBDIR}/libnvinfer_plugin.so.10)
set_target_properties(sbs_portrait PROPERTIES BUILD_RPATH ${TRT_LIBDIR})
