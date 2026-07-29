from fastapi import FastAPI

from app.api.routes import admin_sources, auth
from app.core.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(auth.router)
app.include_router(admin_sources.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
