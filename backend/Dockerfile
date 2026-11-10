FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY fmv ./fmv
RUN pip install --no-cache-dir .
ENV FMV_ARTIFACTS_DIR=/artifacts
EXPOSE 8000
# First start: load synthetic data, train and value (about a minute). Later starts: serve.
CMD ["sh", "-c", "python -m fmv.cli bootstrap && uvicorn fmv.api:app --host 0.0.0.0 --port 8000"]
