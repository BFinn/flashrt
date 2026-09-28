// SPDX-License-Identifier: Apache-2.0
// bench_mma: int8 mma.m16n8k32 throughput on this GPU, register-only, alone and with the scale
// arithmetic moe_q2 does per MMA (the ceiling its kernels can reach).
//
//   bench_mma
#include <cuda_runtime.h>

#include <cstdio>

namespace {

__device__ __forceinline__ void mma(int (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ void mma_acc(int (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_tf32(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// MODE 0: MMAs only, 8 independent accumulators (int32 across iterations)
// MODE 3: bf16 m16n8k16 with fp32 accumulation; MODE 4: tf32 m16n8k8
// MODE 1: per pair of MMAs (two k-steps), moe_q2's per-32 arithmetic: 2 conversions, 3 FMA-type ops per element
// MODE 2: per pair (int32 across both k-steps), the per-64 arithmetic: 1 conversion, FMUL + FFMA per element
template <int MODE>
__global__ void k_bench(int iters, unsigned seed, float* out) {
    unsigned a[4] = {seed, seed * 3u, seed * 5u, seed * 7u};
    unsigned b0 = seed * 11u, b1 = seed * 13u;
    float acc[8][4] = {};
    int ci[8][4] = {};
    const float s0 = 1e-3f * (threadIdx.x + 1), s1 = 2e-3f;
    const int m0 = 0x4B400000 - int(seed & 15);
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const unsigned at[4] = {a[0] + t, a[1], a[2], a[3]};   // distinct per t: no merged MMAs
            if constexpr (MODE == 3) {
                mma_f16(acc[t], at, b0, b1);
            } else if constexpr (MODE == 4) {
                mma_tf32(acc[t], at, b0, b1);
            } else if constexpr (MODE == 0) {
                mma_acc(ci[t], at, b0, b1);
            } else if constexpr (MODE == 1) {
                int c0[4], c1[4];
                mma(c0, at, b0, b1);
                mma(c1, at, b1, b0);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float f0 = __int_as_float(c0[q] + m0) - 12582912.0f, f1 = __int_as_float(c1[q] + m0) - 12582912.0f;
                    acc[t][q] = fmaf(fmaf(f1, s1, f0 * s0), s1, acc[t][q]);
                }
            } else {
                int c[4];
                mma(c, at, b0, b1);
                mma_acc(c, at, b1, b0);
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[t][q] = fmaf(__int_as_float(c[q] + m0) - 12582912.0f, s0 * s1, acc[t][q]);
            }
        }
        b0 += 0x01010101u;
    }
    float r = 0.0f;
    for (int t = 0; t < 8; ++t)
        for (int q = 0; q < 4; ++q) r += acc[t][q] + float(ci[t][q]);
    if (r == 12345.678f) out[0] = r;
}

template <int MODE>
void run(const char* name, int mmas_per_iter) {
    int dev = 0, nsm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    float* out;
    cudaMalloc(&out, 4);
    const int iters = 4096;
    for (int warps : {4, 8, 16}) {
      for (int per_sm : {1, 4}) {   // CTAs per SM
        const dim3 grid(nsm * per_sm), block(32 * warps);
        k_bench<MODE><<<grid, block>>>(16, 1, out);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        k_bench<MODE><<<grid, block>>>(iters, 1, out);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double k = MODE == 3 ? 16.0 : MODE == 4 ? 8.0 : 32.0;   // the MMA's k
        const double ops = double(grid.x) * warps * iters * mmas_per_iter * 16.0 * 8 * k * 2;
        std::printf("%-44s %2d warps/CTA, %d CTA(s)/SM: %6.1f TOPS\n", name, warps, per_sm, ops / (ms * 1e-3) / 1e12);
      }
    }
    cudaFree(out);
}

}  // namespace

int main() {
    run<0>("mma only", 8);
    run<1>("mma + per-32 scale arithmetic", 16);
    run<2>("mma + per-64 scale arithmetic", 16);
    run<3>("bf16 m16n8k16, fp32 accumulate (TFLOPS)", 8);
    run<4>("tf32 m16n8k8, fp32 accumulate (TFLOPS)", 8);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
