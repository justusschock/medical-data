# supported are cuda and cpu
ARG HARDWARE=cuda

FROM nvcr.io/nvidia/pytorch:21.10-py3 as base_cuda-latest
ARG PYPI_OPTIONS=""

FROM nvcr.io/nvidia/pytorch:21.10-py3 as base_latest
ARG PYPI_OPTIONS=""

FROM continuumio/miniconda3:4.10.3p1 as base_cpu-latest
# look for cpu torch releases
ARG PYPI_OPTIONS="-f https://download.pytorch.org/whl/cpu/torch_stable.html"

FROM base_${HARDWARE}

# Make Directories for config, caches and workdir
# and set permissions to also use them as non-root
RUN mkdir /workdir && \
    chmod -R 777 /workdir && \
    mkdir /.cache && \
    chmod -R 777 /.cache && \
    mkdir /.config && \
    chmod -R 777 /.config

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y bash

COPY . /workdir/medical-dl-utils
RUN pip install /workdir/medical-dl-utils ${PYPI_OPTIONS}

WORKDIR /workdir
