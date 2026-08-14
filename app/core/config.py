from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "LoreTrace API"
    environment: str = "development"
    database_url: str = "postgresql+asyncpg://loretrace:loretrace@localhost:5432/loretrace"

    secret_key: str = "dev-secret-change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Consumed only by scripts/seed_super_admin.py, never read at request time.
    super_admin_email: str | None = None
    super_admin_password: str | None = None

    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"

    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_model: str = "@cf/meta/llama-3.1-8b-instruct"

    self_hosted_url: str | None = None
    self_hosted_api_token: str | None = None
    self_hosted_model: str = "llama3.1:8b"

    # A claim+scrape+chunk+embed+commit cycle should never legitimately take
    # longer than this; past it, app.worker.runner assumes the connection is
    # wedged (see its module comment) rather than that the job is just slow.
    # Kept tunable via env rather than a code constant since the right value
    # depends on real embedding throughput for the largest source ingested,
    # which will only be known empirically.
    worker_job_watchdog_seconds: float = 300.0

    # Consumed only by the admin credibility-suggestion lookup
    # (LoreTrace_Credibility_Suggestion_Design.md), never by /chat.
    tavily_api_key: str | None = None

    # Comma-separated, e.g. "https://loretrace.pages.dev". Local dev origins
    # (any http://localhost:<port>) are allowed separately in app.main, since
    # the Vite dev server's port isn't fixed.
    cors_allowed_origins: str = ""

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


settings = Settings()

if settings.environment == "production" and settings.secret_key == "dev-secret-change-me":
    raise RuntimeError("SECRET_KEY must be set via environment variable in production")
