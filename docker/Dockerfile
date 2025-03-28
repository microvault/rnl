FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PROGRESS_BAR=1 \
    PIP_PROGRESS_BAR=off \
    DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /workdir
ENV PYTHONPATH=/workdir

COPY rnl ./rnl
COPY data ./data
COPY main.py ./
COPY pyproject.toml ./

RUN pip install uv && uv pip install .

ENTRYPOINT ["bash", "-c", "apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 && python -m main $@"]
CMD ["learn"]
