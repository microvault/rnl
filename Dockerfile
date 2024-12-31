FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PROGRESS_BAR=1

WORKDIR /workdir

ENV PYTHONPATH=/workdir

COPY rnl ./rnl
COPY requirements.txt ./
COPY pyproject.toml poetry.lock ./

RUN pip install agilerl --progress-bar off
RUN pip install -r requirements.txt --progress-bar off

CMD ["bash", "-c", "apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0"]
