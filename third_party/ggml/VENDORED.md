# Vendored: ggml CUDA mat-vec kernels (MIT)

Source: llama.cpp (https://github.com/ggml-org/llama.cpp), commit 187664b537aabee60f10ca2e3e791010433b3876, directory ggml/.
License: MIT, see LICENSE in this directory (copied from the llama.cpp repository root).

Files are copied **unmodified**, keeping ggml's own paths (include/, src/, src/ggml-cuda/).
They are the include closure of five kernel sources:
- src/ggml-cuda/mmvq.cu      quantized-weight x Q8_1-activation mat-vec (all ggml quant types)
- src/ggml-cuda/mmvf.cu      F32 / F16 / BF16-weight mat-vec
- src/ggml-cuda/quantize.cu  F32 -> Q8_1 activation quantization
- src/ggml-cuda/convert.cu   dequantization to F32 (embedding and PLE rows); adds dequantize.cuh
- src/ggml-cuda/mmq.cuh      quantized matrix-matrix products (int8 tensor cores), for prefill; with
                             its configs, mma.cuh, mmq-load-tiles.cuh and mmq-vec-dot.cuh
(src/ggml-cuda/mmid.cu, ggml's expert grouping, was vendored on 2026-09-28 and removed the same
day: kernels/cuda/ggml_gemm.cu groups the tokens itself.)

These .cu files are not compiled on their own. flashrt's wrapper, kernels/cuda/ggml_gemv.cu,
includes them into one translation unit and calls their raw-pointer dispatchers; the MMQ
kernels are launched by kernels/cuda/mmq_launch.cuh (one translation unit per weight type)
(mul_mat_vec_q_switch_type, mul_mat_vec_f_cuda, quantize_row_q8_1_cuda, ggml_get_to_fp32_cuda). The ggml runtime
symbols they reference but flashrt does not use are defined in kernels/cuda/ggml_shim.cu (the CUDA
backend's) and kernels/cuda/ggml_stubs.cpp (ggml core's).

To update: copy the same closure from a newer llama.cpp, record the commit here, and re-run
tests/test_gemv.
