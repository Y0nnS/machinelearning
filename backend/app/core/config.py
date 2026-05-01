from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AI_LAB_", extra="ignore")

    APP_NAME: str = "AI Lab API"
    APP_VERSION: str = "2.0.0"
    APP_DEBUG: bool = True

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:4321",
        "http://localhost:3000",
        "http://127.0.0.1:4321",
    ]

    # Model directory
    MODELS_DIR: str = "models"

    # Optional OpenAI-compatible external provider
    EXTERNAL_AI_BASE_URL: str = "https://api.openai.com/v1"
    EXTERNAL_AI_API_KEY: str = ""
    EXTERNAL_AI_MODEL: str = "gpt-4o-mini"


settings = Settings()
