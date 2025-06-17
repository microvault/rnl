# This docker is only adapted for the server, it is not the official version.
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PROGRESS_BAR=1 \
    PIP_PROGRESS_BAR=off \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workdir

ENV PYTHONPATH=/workdir

COPY rnl ./rnl
COPY data ./data
COPY main.py ./

RUN pip install \
    "gymnasium >=0.28.1, <0.30.0" \
    "matplotlib >=3.3" \
    "torch >=2.0.1, <3.0.0" \
    "pymunk ==6.11.0" \
    "tqdm >=4.65.0, <5.0.0" \
    "numba >=0.59.1, <0.61.0" \
    "shapely >=1.8, <3.0" \
    "stable-baselines3 >=2.4.1, <3.0.0" \
    "wandb ==0.17.6" \
    "google-genai>=1.3.0" \
    "numpy>=1.26.4"


ENTRYPOINT ["bash", "-c", "apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 && python -m main $@"]

CMD ["learn"]
