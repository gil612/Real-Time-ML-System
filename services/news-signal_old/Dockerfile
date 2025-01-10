# An example using multi-stage image builds to create a final image without uv.

# First stage: build dependencies
FROM ghcr.io/astral-sh/uv:python3.10-bookworm AS builder

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install --system \
        pydantic \
        pydantic-settings \
        loguru \
        llama-index \
        llama-index-llms-anthropic \
        llama-index-llms-ollama \
        anthropic \
        quixstreams \
        httpx && \
    uv sync --frozen --no-install-project --no-dev

# Copy application code
ADD . /app

# Install application
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Second stage: final image
FROM python:3.10-slim

WORKDIR /app

# Install networking tools
RUN apt-get update && \
    apt-get install -y curl iputils-ping net-tools dnsutils && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 app

# Copy installed dependencies and application code
COPY --from=builder --chown=app:app /app /app
COPY --from=builder --chown=app:app /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Add app directory to Python path
ENV PYTHONPATH=/app

# Switch to non-root user
USER app

# Command to run the application
ENTRYPOINT ["python", "-m", "run"]