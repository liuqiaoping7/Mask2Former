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

# OpenGL / Vulkan presentation (headers and GLFW extracted by ../render/setup_deps.sh); optional
set(RENDER_SYSROOT "$ENV{HOME}/data/sbs/render/sysroot" CACHE PATH "GL / Vulkan / GLFW headers and libraries")
if(EXISTS ${RENDER_SYSROOT}/usr/include/GLFW/glfw3.h)
  set(RLIB ${RENDER_SYSROOT}/usr/lib/x86_64-linux-gnu)
  set(SHADER_SRC ${CMAKE_SOURCE_DIR}/../render/shaders)
  set(SHADER_OUT ${CMAKE_BINARY_DIR}/shaders)
  set(SHADERS)
  foreach(sh sbs.vert sbs.frag)  # GLSL for OpenGL (compiled by the driver), SPIR-V for Vulkan
    add_custom_command(OUTPUT ${SHADER_OUT}/${sh}.spv ${SHADER_OUT}/${sh}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUT}
      COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_SRC}/${sh} ${SHADER_OUT}/${sh}
      COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${RLIB} ${RENDER_SYSROOT}/usr/bin/glslangValidator
              -V --target-env vulkan1.3 -o ${SHADER_OUT}/${sh}.spv ${SHADER_SRC}/${sh}
      DEPENDS ${SHADER_SRC}/${sh} VERBATIM)
    list(APPEND SHADERS ${SHADER_OUT}/${sh}.spv)
  endforeach()
  add_custom_target(shaders DEPENDS ${SHADERS})
  add_library(sbs_render STATIC render/presenter.cpp render/gl_renderer.cpp render/vk_renderer.cpp)
  add_dependencies(sbs_render shaders)
  target_include_directories(sbs_render PUBLIC render src ${RENDER_SYSROOT}/usr/include)
  target_compile_definitions(sbs_render PUBLIC SBS_RENDER SHADER_DIR="${SHADER_OUT}")
  target_link_libraries(sbs_render PUBLIC ${RLIB}/libglfw.so /lib/x86_64-linux-gnu/libvulkan.so.1 CUDA::cudart
    Threads::Threads)
  target_link_libraries(sbs_portrait PRIVATE sbs_render)
  set_target_properties(sbs_portrait PROPERTIES BUILD_RPATH "${TRT_LIBDIR};${RLIB}")
  add_executable(render_bench render/render_bench.cpp)
  target_link_libraries(render_bench PRIVATE sbs_render)
  set_target_properties(render_bench PROPERTIES BUILD_RPATH ${RLIB})
else()
  message(STATUS "rendering disabled: ${RENDER_SYSROOT} has no GLFW headers (run ../render/setup_deps.sh)")
endif()
