FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_MAX_PARALLEL=1 \
    PIP_DISABLE_PROGRESS_BAR=1

WORKDIR /workdir

ENV PYTHONPATH=/workdir

COPY rnl ./rnl
COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir --progress-bar off poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-ansi --no-interaction && \
    poetry install --without dev && \
    rm -rf $POETRY_CACHE_DIR

CMD ["poetry", "run", "python", "rnl/benchmarks/train_multi_env.py"]
