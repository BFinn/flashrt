// SPDX-License-Identifier: Apache-2.0
// fr_gpu_load: upload qwen4exp's dense weights to VRAM and run one real mat-vec as a check.
//
//   fr_gpu_load MODEL.gguf
#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "core/gguf.hpp"
#include "kernels/cuda/ggml_gemv.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace flashrt;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: fr_gpu_load MODEL.gguf\n"); return 2; }
    const Gguf g = Gguf::open(argv[1]);
    const qwen4exp::Spec s = qwen4exp::parse(g);
    const qwen4exp::WeightPlan plan = qwen4exp::plan(g, s);
    qwen4exp::GpuWeights w;
    w.load(g, plan, false);   // ggml types as stored (this tool times ggml's mat-vec on them)
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::printf("dense weights: %.1f MiB in VRAM (with padding) in %.1f s; VRAM free %.0f of %.0f MiB\n",
                w.device_bytes() / 1048576.0, w.load_seconds(), free_b / 1048576.0, total_b / 1048576.0);

    // the router of layer 0 (BF16 [2560 -> 512]) on a constant input: finite, not all zero
    const qwen4exp::GpuTensor& r = w.layer(0, "ffn_gate_inp.weight");
    std::vector<float> x(r.cols(), 0.01f), y(r.rows());
    float *dx, *dy;
    void* scratch;
    cudaMalloc(&dx, x.size() * 4);
    cudaMalloc(&dy, y.size() * 4);
    cudaMalloc(&scratch, gemv::q8_1_bytes(r.cols(), 1));
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    gemv::matvec(r.type, r.dev, dx, dy, r.cols(), r.rows(), 1, scratch, nullptr);
    cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
    double ss = 0;
    bool finite = true;
    for (float v : y) { ss += double(v) * v; finite &= std::isfinite(v); }
    std::printf("router(layer 0) on 0.01s: rms %.4g, finite %s, cuda %s\n", std::sqrt(ss / y.size()), finite ? "yes" : "NO",
                cudaGetErrorString(cudaGetLastError()));
    return finite ? 0 : 1;
}
