from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "LoreTrace API"
    environment: str = "development"
    database_url: str = "postgresql+asyncpg://loretrace:loretrace@localhost:5432/loretrace"


settings = Settings()
