# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies with cache mounts
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user 'appuser'
RUN groupadd -r appuser && useradd -u 1000 -r -g appuser -m -d /home/appuser appuser

# Set up the working directory
WORKDIR /ui

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT="/home/appuser/venv"
ENV PYTHONUNBUFFERED=1

# Setup permissions
RUN mkdir -p /home/appuser/venv /ui && \
    chown -R appuser:appuser /ui /home/appuser

# Switch to non-root user
USER appuser

# Place executables in the path
ENV PATH="/home/appuser/venv/bin:$PATH"

# Install dependencies (Cached Layer)
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy source code with ownership
COPY --chown=appuser:appuser . /ui

# Install the project itself
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-dev

# Set PYTHONPATH to prioritize the src directory
ENV PYTHONPATH="/ui/src:$PYTHONPATH"

# Run the application.
CMD ["uv", "run", "streamlit", "run", "src/atrium_csd_ner/ui.py"]