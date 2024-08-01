FROM --platform=linux/amd64 python:3.10-slim as builder

WORKDIR /app

ENV POETRY_VERSION=1.7.1

# Install libraries for necessary python package builds
RUN apt-get update && apt-get install build-essential python3-dev libpq-dev -y

RUN pip install --upgrade pip
RUN pip install --upgrade poetry==${POETRY_VERSION}

# Configure Poetry
ENV POETRY_CACHE_DIR=/tmp/poetry_cache
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_VIRTUALENVS_CREATE=true

# Install dependencies
COPY contributor-3/poetry.lock contributor-3/pyproject.toml ./

RUN poetry install --no-cache --no-root

FROM --platform=linux/amd64 python:3.10-slim as runtime

RUN apt-get update && apt-get install libpq5 -y && rm -rf /var/lib/apt/lists/*  # Install libpq for psycopg2

RUN groupadd -r appuser && useradd --no-create-home -g appuser -r appuser
USER appuser

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Copy source code
COPY ./logging.ini ./logging.ini
COPY ./contributor-3/contributor_3 ./contributor_3
