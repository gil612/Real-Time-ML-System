# An example using multi-stage image builds to create a final image without uv.

# First stage: build dependencies
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Second stage: final image
FROM python:3.12-slim

WORKDIR /app

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app
COPY --from=builder --chown=app:app /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["python", "/app/run.py"]
