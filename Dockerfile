FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PROGRESS_BAR=1

# Instala GNUPG e baixa a chave pública da NVIDIA
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates gnupg2 \
    && rm -rf /var/lib/apt/lists/*
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# Agora o update + instalação das libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workdir
ENV PYTHONPATH=/workdir

COPY rnl ./rnl
COPY main.py ./
COPY requirements.txt ./
COPY pyproject.toml poetry.lock ./

RUN pip install agilerl --progress-bar off
RUN pip install -r requirements.txt --progress-bar off

CMD ["bash", "-c", "python main.py"]
