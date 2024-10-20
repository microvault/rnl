FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workdir

COPY rnl ./rnl
COPY tests ./tests
COPY pyproject.toml poetry.lock ./
COPY train.py ./

RUN poetry install --without dev && rm -rf $POETRY_CACHE_DIR

CMD ["poetry", "run", "python", "train.py", "train"]
