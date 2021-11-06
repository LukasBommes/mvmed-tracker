FROM ubuntu:18.04 AS builder

RUN apt-get update && apt-get install -y \
  git && \
  rm -rf /var/lib/apt/lists/*

ENV HOME "/home"

# Download and build sfmt-videocap from source
RUN cd $HOME && \
  git clone -b "v1.0.0" https://github.com/LukasBommes/mv-extractor.git video_cap && \
  cd video_cap && \
  chmod +x install.sh && \
  ./install.sh


FROM nvidia/cuda:10.0-base-ubuntu18.04 AS base
ARG ARCH=
ARG CUDA=10.0
ARG CUDNN=7.6.2.24-1

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        cuda-cublas-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

RUN [ ${ARCH} = ppc64le ] || (apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*)

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

ARG PYTHON=python3
ARG PIP=pip3

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    pkg-config \
    ${PYTHON}-dev \
    ${PYTHON}-pip \
    ${PYTHON}-numpy \
    ${PYTHON}-pkgconfig && \
    rm -rf /var/lib/apt/lists/*

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow-gpu
ARG TF_PACKAGE_VERSION=1.13.1
RUN ${PIP} install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

RUN apt-get update && \
  apt-get -y install \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libmp3lame-dev \
    zlib1g-dev \
    libx264-dev \
    libsdl2-dev \
    libvpx-dev \
    libvdpau-dev \
    libvorbis-dev \
    libopus-dev \
    libdc1394-22-dev \
    liblzma-dev && \
  rm -rf /var/lib/apt/lists/*

# copy libraries
WORKDIR /usr/local/lib
COPY --from=builder /usr/local/lib .
WORKDIR /usr/local/include
COPY --from=builder /home/ffmpeg_build/include .
WORKDIR /home/ffmpeg_build/lib
COPY --from=builder /home/ffmpeg_build/lib .
WORKDIR /usr/local/include/opencv4/
COPY --from=builder /usr/local/include/opencv4/ .
WORKDIR /home/opencv/build/lib
COPY --from=builder /home/opencv/build/lib .
WORKDIR /home/video_cap
COPY --from=builder /home/video_cap .

# Set environment variables
ENV PATH="$PATH:/home/bin"
ENV PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/home/ffmpeg_build/lib/pkgconfig"

# Install video_cap Python module
WORKDIR /home/video_cap
RUN python3 setup.py install

# Install Python packages
COPY requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt

WORKDIR /mvmed_tracker

CMD ["sh", "-c", "tail -f /dev/null"]

