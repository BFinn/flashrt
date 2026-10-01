// SPDX-License-Identifier: Apache-2.0
// For the CUDA tests: a failed CUDA call ends the test there, with its location and error,
// instead of showing up later as a mismatch against the reference.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                                                               \
    do {                                                                                            \
        const cudaError_t e_ = (x);                                                                 \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(2);                                                                           \
        }                                                                                           \
    } while (0)
