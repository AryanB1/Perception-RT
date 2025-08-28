FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopencv-dev libyaml-cpp-dev cmake g++ git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
EXPOSE 9090
CMD ["./build/framekeeper-rt", "--config", "configs/config.yaml"]
