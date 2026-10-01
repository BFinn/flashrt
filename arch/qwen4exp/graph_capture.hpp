// SPDX-License-Identifier: Apache-2.0
// Graph capture that cannot leave a stream capturing.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace flashrt::qwen4exp {

// The graph of what body() enqueues on st. If body throws, the capture is ended and discarded
// before the exception goes on: a stream left capturing fails every later call, the session's
// recovery included.
template <class F>
cudaGraph_t capture_graph(cudaStream_t st, F&& body, const char* what) {
    auto check = [what](cudaError_t e) {
        if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    };
    check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    try {
        body();
    } catch (...) {
        cudaGraph_t g = nullptr;
        if (cudaStreamEndCapture(st, &g) == cudaSuccess && g) cudaGraphDestroy(g);
        (void)cudaGetLastError();
        throw;
    }
    cudaGraph_t g = nullptr;
    check(cudaStreamEndCapture(st, &g));
    return g;
}

}  // namespace flashrt::qwen4exp
