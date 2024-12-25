FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends locales && \
    locale-gen en_US.UTF-8 && \
    locale-gen C.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workdir

COPY rnl ./rnl
COPY pyproject.toml poetry.lock ./
COPY train.py ./

ENV PYTHONPATH=/workdir

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

RUN poetry install --without dev && rm -rf $POETRY_CACHE_DIR

CMD ["poetry", "run", "python", "train.py", "train"]
