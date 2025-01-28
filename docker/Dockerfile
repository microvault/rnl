FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PROGRESS_BAR=1 \
    PIP_PROGRESS_BAR=off \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workdir

ENV PYTHONPATH=/workdir

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/requirements.txt ./
COPY rnl ./rnl
COPY data ./data
COPY main.py ./

RUN pip install -r requirements.txt --no-cache-dir -qq

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
