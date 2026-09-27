cmake_minimum_required(VERSION 3.24)
project(flashrt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FLASHRT_CUDA   "Build the CUDA parts" ON)
option(FLASHRT_NATIVE "Tune CPU code for the build machine (-march=native)" ON)

if(FLASHRT_NATIVE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

find_package(Threads REQUIRED)

# ---- core: platform services shared by the engine and the tools
add_library(flashrt_core STATIC
  core/platform.cpp)
target_include_directories(flashrt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(flashrt_core PUBLIC Threads::Threads)

# ---- engine process (protocol skeleton; generation arrives in P1)
add_executable(flashrt-engine engine/main.cpp)
target_link_libraries(flashrt-engine PRIVATE flashrt_core)

# ---- phase 0 probes
add_executable(membw tools/membw.cpp)
target_link_libraries(membw PRIVATE flashrt_core)

if(FLASHRT_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
  endif()
  set(CMAKE_CUDA_STANDARD 20)
  add_executable(h2dbw tools/h2dbw.cu)
  target_link_libraries(h2dbw PRIVATE flashrt_core CUDA::cudart)
endif()
