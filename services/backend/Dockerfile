FROM python:3.14

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Pin to the base image's own Python rather than letting uv manage one.
ENV UV_PYTHON_PREFERENCE=only-system

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# .dockerignore excludes .venv - without it this COPY overwrites the venv
# uv just built above with whatever local dev .venv happens to exist on the
# host (broken: it symlinks to host-only interpreter paths).
COPY . .
