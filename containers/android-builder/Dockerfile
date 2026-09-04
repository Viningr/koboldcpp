# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ghcr.io/snapdragon-toolchain/arm64-android:v0.7@sha256:91714433626f0d94a926538a1e46ec43756c5b8e3262b91b95df1e812940aed1
FROM ${BASE_IMAGE}

ARG MAKE_DEBIAN_VERSION=4.4.1-2

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends "make=${MAKE_DEBIAN_VERSION}" \
 && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="KoboldCpp Snapdragon Android builder" \
      org.opencontainers.image.description="Pinned NDK r29 and Qualcomm Hexagon SDK builder with GNU Make" \
      org.opencontainers.image.source="https://github.com/Viningr/koboldcpp"
