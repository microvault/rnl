FROM ubuntu:22.04

SHELL [ "/bin/bash" , "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    python3-pip \
    python3-dev \
    apt-utils \
    g++ \
    git \
    swig \
    zlib1g-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --upgrade "setuptools>=44.0.0"
RUN pip install --upgrade "wheel>=0.37.1"
RUN pip install --upgrade Cython
RUN pip install --no-binary=h5py h5py
RUN pip install gymnasium[box2d]

WORKDIR /rnl

RUN pip install agilerl

RUN pip install matplotlib scikit-image numba shapely

COPY ./rnl /rnl

RUN export PYTHONPATH=$(pwd)
