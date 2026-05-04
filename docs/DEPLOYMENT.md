# Deployment Guide

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
make run-api
make run-ui
```

## Docker

```bash
docker compose up --build
```

## Cloud Targets

### Render

- Deploy FastAPI as a web service.
- Deploy Streamlit as a second service or static client replacement.
- Persist Chroma data using a disk mount or migrate to a managed vector database.

### Hugging Face Spaces

- Use Streamlit for the UI layer.
- Host the backend separately or embed a lightweight local inference workflow for demos.

### AWS

- Containerize backend with ECS or App Runner.
- Move storage to S3 + RDS/pgvector or Pinecone for managed retrieval.
- Put secrets in AWS Secrets Manager.

