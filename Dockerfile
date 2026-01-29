FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0,1
ENV POLARS_SKIP_CPU_CHECK=1
ENV RAY_DISABLE_METRICS_EXPORT=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN pip3 install -U "flwr[simulation]"
