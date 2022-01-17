FROM ubuntu:20.04

ARG CONDA_VERSION=4.9.2

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    PATH="$PATH:/root/.local/bin" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin \
    # CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MKL_THREADING_LAYER=GNU

RUN apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        unzip \
        ca-certificates

# Install conda and python.
# NOTE new Conda does not forward the exit status... https://github.com/conda/conda/issues/8385
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_${CONDA_VERSION}-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh

# Cleaning
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

# Make Directories for config, caches and workdir
# and set permissions to also use them as non-root
RUN mkdir /workdir && \
    chmod -R 777 /workdir && \
    mkdir /.cache && \
    chmod -R 777 /.cache && \
    mkdir /.config && \
    chmod -R 777 /.config

ENV \
    PATH="/root/miniconda3/bin:$PATH" \
    LD_LIBRARY_PATH="/root/miniconda3/lib:$LD_LIBRARY_PATH" \
    CONDA_ENV=base

COPY . /workdir/medical-dl-utils
RUN pip install /workdir/medical-dl-utils -f https://download.pytorch.org/whl/cpu/torch_stable.html

WORKDIR /workdir