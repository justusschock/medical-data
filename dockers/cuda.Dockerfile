FROM nvcr.io/nvidia/pytorch:21.10-py3

# Make Directories for config, caches and workdir 
# and set permissions to also use them as non-root
RUN mkdir /workdir && \
    chmod -R 777 /workdir && \
    mkdir /.cache && \
    chmod -R 777 /.cache && \
    mkdir /.config && \
    chmod -R 777 /.config

# opencv dependencies
RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y bash ffmpeg libsm6 libxext6

COPY . /workdir/medical-dl-utils
RUN pip install /workdir/medical-dl-utils

WORKDIR /workdir
