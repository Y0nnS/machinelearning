from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "ML Playground API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:4321",  
        "http://localhost:3000",
        "http://127.0.0.1:4321",
    ]

    # Model directory
    MODELS_DIR: str = "models"

    class Config:
        env_file = ".env"


settings = Settings()
