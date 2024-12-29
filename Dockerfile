FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_MAX_PARALLEL=1 \
    PIP_DISABLE_PROGRESS_BAR=1 \
    RICH_DISABLE=1

WORKDIR /workdir

ENV PYTHONPATH=/workdir

COPY rnl ./rnl
COPY requirements.txt ./
COPY pyproject.toml poetry.lock ./

RUN pip install -r requirements.txt --progress-bar off
# RUN pip install --no-cache-dir --progress-bar off poetry && \
#     poetry config virtualenvs.create false && \
#     poetry install --only main --no-ansi --no-interaction && \
#     poetry install --without dev && \
#     rm -rf $POETRY_CACHE_DIR

#CMD ["poetry", "run", "python", "rnl/benchmarks/train_multi_env.py"]
