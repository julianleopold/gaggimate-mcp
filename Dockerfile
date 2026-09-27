FROM python:3.13-slim

# Install uv
RUN pip install uv --no-cache-dir

WORKDIR /app

# Copy project definition and lockfile first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen from lockfile, no dev extras)
RUN uv sync --frozen --no-dev

# Copy source and runtime files
COPY run_server.py ./
COPY src/ ./src/
COPY knowledge/ ./knowledge/
COPY agent-instructions/ ./agent-instructions/
COPY agent-skills/ ./agent-skills/

# Persistent data directories (override with volume mounts in compose)
RUN mkdir -p coffees data profiles user

EXPOSE 3000

CMD ["uv", "run", "python", "run_server.py"]
