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

    self_hosted_space: str | None = None
    self_hosted_api_name: str = "/predict"
    self_hosted_hf_token: str | None = None


settings = Settings()
