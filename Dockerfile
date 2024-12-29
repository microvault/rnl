FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

ENV PYTHONDONTWRITEBYTECODE 1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /workdir

ENV PYTHONPATH=/workdir

COPY rnl ./rnl
COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-ansi --no-interaction && \
    poetry install --without dev && \
    rm -rf $POETRY_CACHE_DIR

CMD ["poetry", "run", "python", "rnl/benchmarks/train_multi_env.py"]
