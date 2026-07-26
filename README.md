# LoreTrace Backend

RAG API for LoreTrace, a chatbot that answers mythology questions by retrieving from
public-domain source texts and flagging where those sources disagree instead of
silently picking one as correct.

FastAPI + PostgreSQL/pgvector. Frontend lives in a separate `loretrace-frontend` repo.

## Status

Early development. Data layer and admin ingestion pipeline in progress.

## Stack

- FastAPI (async)
- PostgreSQL + pgvector
- SQLAlchemy + Alembic
- JWT auth (admin-only for now)
