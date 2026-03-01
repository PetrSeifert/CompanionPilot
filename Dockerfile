FROM rust:1.88-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Required by audiopus_sys CMake invocation used by Songbird.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

COPY . .

RUN cargo build --release -p companionpilot

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/companionpilot /usr/local/bin/companionpilot

ENV RUST_LOG=info
CMD ["companionpilot"]
