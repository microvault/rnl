FROM python:3.12-slim

RUN apt-get update && apt-get install -y build-essential

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /workdir

COPY rnl ./rnl
COPY tests ./tests
COPY Makefile ./
COPY pyproject.toml poetry.lock ./
COPY train.py ./

VOLUME . /workdir

RUN poetry install --without dev && rm -rf $POETRY_CACHE_DIR
