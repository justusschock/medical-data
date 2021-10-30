FROM nvcr.io/nvidia/pytorch:21.10-py3
# opencv dependencies + pip packages
RUN \
    apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y bash ffmpeg libsm6 libxext6 && \
    mkdir /workdir && \
    chmod -R 777 /workdir && \
    mkdir /.cache && \
    chmod -R 777 /.cache && \
    mkdir /.config && \
    chmod -R 777 /.config

COPY . /workdir/medical-dl-utils
RUN pip install /workdir/medical-dl-utils

WORKDIR /workdir
