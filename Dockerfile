FROM golang:1.24-bookworm AS spogo-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*
ARG SPOGO_REPO=https://github.com/PetrSeifert/spogo.git
ARG SPOGO_REF=main
RUN git clone --depth 1 --branch ${SPOGO_REF} ${SPOGO_REPO} /tmp/spogo \
    && cd /tmp/spogo \
    && CGO_ENABLED=0 go build -o /go/bin/spogo ./cmd/spogo

FROM rust:1.91.0-slim-bookworm AS builder
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
COPY --from=spogo-builder /go/bin/spogo /usr/local/bin/spogo
COPY --from=builder /app/skills /app/skills

ENV RUST_LOG=info
CMD ["companionpilot"]
