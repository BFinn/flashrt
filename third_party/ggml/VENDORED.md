# Vendored: ggml CUDA mat-vec kernels (MIT)

Source: llama.cpp (https://github.com/ggml-org/llama.cpp), commit 187664b537aabee60f10ca2e3e791010433b3876, directory ggml/.
License: MIT, see LICENSE in this directory (copied from the llama.cpp repository root).

Files are copied **unmodified**, keeping ggml's own paths (include/, src/, src/ggml-cuda/).
They are the include closure of three kernel files:
- src/ggml-cuda/mmvq.cu      quantized-weight x Q8_1-activation mat-vec (all ggml quant types)
- src/ggml-cuda/mmvf.cu      F32 / F16 / BF16-weight mat-vec
- src/ggml-cuda/quantize.cu  F32 -> Q8_1 activation quantization

These .cu files are not compiled on their own. flashrt's wrapper, kernels/cuda/ggml_gemv.cu,
includes them into one translation unit and calls their raw-pointer dispatchers
(mul_mat_vec_q_switch_type, mul_mat_vec_f_cuda, quantize_row_q8_1_cuda). The ggml runtime
symbols they reference but flashrt does not use are stubbed in kernels/cuda/ggml_shim.cu.

To update: copy the same closure from a newer llama.cpp, record the commit here, and re-run
tests/test_gemv.
