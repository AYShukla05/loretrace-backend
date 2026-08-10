import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin_admins, admin_sources, auth, chat
from app.core.config import settings
from app.worker.runner import run_worker


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Runs the scrape-job poller as a background task inside this same
    # process rather than a separate service, since the free tier of the
    # target deployment platform (Render) has no free background-worker
    # service type.
    worker_task = asyncio.create_task(run_worker())
    try:
        yield
    finally:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_origin_regex=r"http://localhost:\d+" if settings.environment == "development" else None,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(admin_admins.router)
app.include_router(admin_sources.router)
app.include_router(chat.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
