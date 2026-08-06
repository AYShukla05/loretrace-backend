from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "LoreTrace API"
    environment: str = "development"
    database_url: str = "postgresql+asyncpg://loretrace:loretrace@localhost:5432/loretrace"

    secret_key: str = "dev-secret-change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"

    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_model: str = "@cf/meta/llama-3.1-8b-instruct"

    self_hosted_url: str | None = None
    self_hosted_api_token: str | None = None
    self_hosted_model: str = "llama3.1:8b"

    # Comma-separated, e.g. "https://loretrace.pages.dev". Local dev origins
    # (any http://localhost:<port>) are allowed separately in app.main, since
    # the Vite dev server's port isn't fixed.
    cors_allowed_origins: str = ""

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


settings = Settings()
