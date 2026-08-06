from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin_sources, auth, chat
from app.core.config import settings

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_origin_regex=r"http://localhost:\d+" if settings.environment == "development" else None,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(admin_sources.router)
app.include_router(chat.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
