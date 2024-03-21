FROM ubuntu:22.04

SHELL [ "/bin/bash" , "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-dev \
    gcc \
    apt-utils && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /microvault

COPY . /microvault

RUN pip install -r requirements.txt --no-cache-dir
