# SPDX-License-Identifier: Apache-2.0
# flashrt in a container (phase 6, X-2): the engine, the tools and the server, built in NVIDIA's
# CUDA 12.9 image, run on its runtime image. The GPU is passed through by the NVIDIA container
# toolkit; the model files are mounted.
#
#   docker build -t flashrt .                                 # sm_120; --build-arg CUDA_ARCH=90 for Hopper (untested)
#   docker build -t flashrt --build-arg NATIVE=ON .           # CPU code for the build machine, as measured
#   docker run --gpus all -p 8090:8090 -v $MODELS:/models flashrt \
#       --model /models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --host 0.0.0.0 --port 8090 \
#       --engine /opt/flashrt/bin/flashrt-engine --engine-arg /models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
#       --engine-arg --mtp --engine-arg /models/mtp-Flash-Next-Q8_0-noembd.gguf --engine-arg --spec --engine-arg 2 \
#       --engine-arg --draft-vocab --engine-arg /opt/flashrt/reference/mtp-vocab-ranks.txt
#
# The host needs 64 GB of RAM and an NVMe drive for the model (README: "Will it run on my
# machine"). The development box's measurements ran outside containers.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build g++ curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.93
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
COPY . .
ARG CUDA_ARCH=120
ARG NATIVE=OFF
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DFLASHRT_NATIVE=${NATIVE} \
    && cmake --build build --target flashrt-engine fr_bench fr_kld
RUN cargo build --release --locked --manifest-path server/Cargo.toml

FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 curl && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/flashrt-engine /src/build/fr_bench /src/build/fr_kld /src/server/target/release/flashrt-server /opt/flashrt/bin/
COPY bench/reference /opt/flashrt/reference
COPY bench/flashrt_depthbench.py bench/depthsum.py bench/server_smoke.py /opt/flashrt/bench/
ENV PATH=/opt/flashrt/bin:$PATH
EXPOSE 8090
ENTRYPOINT ["flashrt-server"]
