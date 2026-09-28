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
enable_testing()

# ---- core: platform services shared by the engine and the tools
add_library(flashrt_core STATIC
  core/platform.cpp
  core/gguf.cpp
  core/expert_arena.cpp
  core/row_reader.cpp
  core/cpu_pool.cpp
  core/json.cpp)
target_include_directories(flashrt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(flashrt_core PUBLIC Threads::Threads)

# ---- quant packs
add_library(flashrt_q2_0 STATIC
  quant/q2_0/q2_0.cpp
  quant/q2_0/q2_0_avx512.cpp
  quant/q2_0/moe_cpu.cpp)
target_link_libraries(flashrt_q2_0 PUBLIC flashrt_core)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set_source_files_properties(quant/q2_0/q2_0_avx512.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mf16c;-mfma")
endif()

# ---- architectures
add_library(flashrt_qwen4exp STATIC
  arch/qwen4exp/spec.cpp
  arch/qwen4exp/experts.cpp
  arch/qwen4exp/ple.cpp)
target_link_libraries(flashrt_qwen4exp PUBLIC flashrt_core flashrt_q2_0)


# ---- phase 0 probes
add_executable(membw tools/membw.cpp)
target_link_libraries(membw PRIVATE flashrt_core)
add_executable(ssdrand tools/ssdrand.cpp)
target_link_libraries(ssdrand PRIVATE Threads::Threads)
add_executable(bench_q2_0 tools/bench_q2_0.cpp)
target_link_libraries(bench_q2_0 PRIVATE flashrt_q2_0)
add_executable(fr_load tools/fr_load.cpp)
target_link_libraries(fr_load PRIVATE flashrt_qwen4exp)
add_executable(fr_ple tools/fr_ple.cpp)
target_link_libraries(fr_ple PRIVATE flashrt_qwen4exp)
add_executable(fr_inspect tools/fr_inspect.cpp)
target_link_libraries(fr_inspect PRIVATE flashrt_qwen4exp flashrt_q2_0)
add_executable(bench_moe_cpu tools/bench_moe_cpu.cpp)
target_link_libraries(bench_moe_cpu PRIVATE flashrt_q2_0)
add_executable(gguf_dump tools/gguf_dump.cpp)
target_link_libraries(gguf_dump PRIVATE flashrt_core)

# route_trace links a llama.cpp build (source tree with build/ inside):
#   -DFLASHRT_LLAMA_DIR=$LLAMA_CPP
set(FLASHRT_LLAMA_DIR "" CACHE PATH "llama.cpp source tree with a build/ directory, for route_trace")
if(FLASHRT_LLAMA_DIR)
  find_library(LLAMA_LIB llama HINTS ${FLASHRT_LLAMA_DIR}/build/bin ${FLASHRT_LLAMA_DIR}/build/src REQUIRED NO_DEFAULT_PATH)
  find_library(GGML_LIB ggml HINTS ${FLASHRT_LLAMA_DIR}/build/bin ${FLASHRT_LLAMA_DIR}/build/ggml/src REQUIRED NO_DEFAULT_PATH)
  find_library(GGML_BASE_LIB ggml-base HINTS ${FLASHRT_LLAMA_DIR}/build/bin ${FLASHRT_LLAMA_DIR}/build/ggml/src REQUIRED NO_DEFAULT_PATH)
  add_executable(route_trace tools/route_trace.cpp)
  target_include_directories(route_trace PRIVATE ${FLASHRT_LLAMA_DIR}/include ${FLASHRT_LLAMA_DIR}/ggml/include)
  target_link_libraries(route_trace PRIVATE ${LLAMA_LIB} ${GGML_LIB} ${GGML_BASE_LIB})
  get_filename_component(LLAMA_LIB_DIR ${LLAMA_LIB} DIRECTORY)
  set_target_properties(route_trace PROPERTIES BUILD_RPATH ${LLAMA_LIB_DIR})
  add_executable(ref_dump tools/ref_dump.cpp)
  target_include_directories(ref_dump PRIVATE ${FLASHRT_LLAMA_DIR}/include ${FLASHRT_LLAMA_DIR}/ggml/include)
  target_link_libraries(ref_dump PRIVATE ${LLAMA_LIB} ${GGML_LIB} ${GGML_BASE_LIB})
  set_target_properties(ref_dump PROPERTIES BUILD_RPATH ${LLAMA_LIB_DIR})
endif()

if(FLASHRT_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
  endif()
  set(CMAKE_CUDA_STANDARD 20)
  add_executable(h2dbw tools/h2dbw.cu)
  target_link_libraries(h2dbw PRIVATE flashrt_core CUDA::cudart)

  # dense mat-vec over ggml's vendored CUDA kernels (MIT, third_party/ggml)
  add_library(flashrt_gemv STATIC kernels/cuda/ggml_gemv.cu kernels/cuda/ggml_shim.cu kernels/cuda/q3r.cu)
  target_include_directories(flashrt_gemv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cuda)
  target_include_directories(flashrt_gemv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(flashrt_gemv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --extended-lambda --expt-relaxed-constexpr -Xcudafe=--diag_suppress=177>)
  target_link_libraries(flashrt_gemv PUBLIC CUDA::cudart)

  if(FLASHRT_LLAMA_DIR)
    add_executable(test_gemv tests/test_gemv.cpp)
    target_include_directories(test_gemv PRIVATE ${FLASHRT_LLAMA_DIR}/ggml/include)
    target_link_libraries(test_gemv PRIVATE flashrt_gemv ${GGML_BASE_LIB} ${GGML_LIB})
    set_target_properties(test_gemv PROPERTIES BUILD_RPATH ${LLAMA_LIB_DIR})
    add_test(NAME gemv COMMAND test_gemv)
  endif()
  # stubs for binaries that do not link libggml (never next to libggml: see the file)
  add_library(flashrt_ggml_stubs STATIC kernels/cuda/ggml_stubs.cpp)
  target_include_directories(flashrt_ggml_stubs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include)
  add_library(flashrt_qwen4exp_gpu STATIC arch/qwen4exp/gpu_weights.cu arch/qwen4exp/blocks.cu arch/qwen4exp/forward_ref.cu
    arch/qwen4exp/moe_fast.cu arch/qwen4exp/mtp.cu kernels/cuda/sample.cu)
  target_link_libraries(flashrt_qwen4exp_gpu PUBLIC flashrt_qwen4exp flashrt_gemv)
  add_executable(fr_gpu_load tools/fr_gpu_load.cpp)
  target_link_libraries(fr_gpu_load PRIVATE flashrt_qwen4exp_gpu flashrt_gemv flashrt_ggml_stubs)   # stubs last

  # the engine process: the JSON-lines protocol over a Session
  add_executable(flashrt-engine engine/main.cpp engine/session.cpp)
  target_link_libraries(flashrt-engine PRIVATE flashrt_qwen4exp_gpu flashrt_gemv flashrt_ggml_stubs)   # stubs last

  add_executable(fr_bench tools/fr_bench.cpp)
  target_link_libraries(fr_bench PRIVATE flashrt_qwen4exp_gpu flashrt_gemv flashrt_ggml_stubs)   # stubs last
  add_executable(fr_kld tools/fr_kld.cpp)
  target_link_libraries(fr_kld PRIVATE flashrt_qwen4exp_gpu flashrt_gemv flashrt_ggml_stubs)   # stubs last
  add_executable(fr_parity tools/fr_parity.cpp)
  target_link_libraries(fr_parity PRIVATE flashrt_qwen4exp_gpu flashrt_gemv flashrt_ggml_stubs)   # stubs last

  add_executable(bench_gemv tools/bench_gemv.cpp)
  target_link_libraries(bench_gemv PRIVATE flashrt_gemv flashrt_ggml_stubs)
  add_executable(test_moe_hits tests/test_moe_hits.cpp)
  target_link_libraries(test_moe_hits PRIVATE flashrt_qwen4exp_gpu flashrt_gemv flashrt_ggml_stubs)   # stubs last
  add_test(NAME moe_hits COMMAND test_moe_hits)
  add_executable(test_moe_q tests/test_moe_q.cpp)
  target_link_libraries(test_moe_q PRIVATE flashrt_gemv flashrt_ggml_stubs)
  add_test(NAME moe_q COMMAND test_moe_q)
  add_executable(test_sample tests/test_sample.cpp)
  target_link_libraries(test_sample PRIVATE flashrt_qwen4exp_gpu flashrt_gemv flashrt_ggml_stubs)   # stubs last
  add_test(NAME sample COMMAND test_sample)
endif()

# ---- tests
add_executable(test_cpu_pool tests/test_cpu_pool.cpp)
target_link_libraries(test_cpu_pool PRIVATE flashrt_core)
add_test(NAME cpu_pool COMMAND test_cpu_pool)
add_executable(test_q2_0 tests/test_q2_0.cpp)
target_link_libraries(test_q2_0 PRIVATE flashrt_q2_0)
add_test(NAME q2_0 COMMAND test_q2_0)
add_executable(test_moe_cpu tests/test_moe_cpu.cpp)
target_link_libraries(test_moe_cpu PRIVATE flashrt_q2_0)
add_test(NAME moe_cpu COMMAND test_moe_cpu)
