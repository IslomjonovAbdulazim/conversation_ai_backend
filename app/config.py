from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Database
    database_url: str

    # API Keys
    google_vision_api_key: str
    elevenlabs_api_key: str
    openai_api_key: str

    # Apple Sign In
    apple_team_id: str
    apple_key_id: str
    apple_private_key_path: str = "./apple_private_key.p8"

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30 * 24 * 60  # 30 days

    # App Settings
    debug: bool = False
    port: int = 8000

    # Quiz Settings
    min_words_for_quiz: int = 5
    min_words_for_reading: int = 8
    quiz_time_limits: dict = {
        "anagram": 45,
        "translation_blitz": 30,
        "word_blitz": 30
    }

    class Config:
        env_file = ".env"
        case_sensitive = False


# Load settings
settings = Settings()

# Validate Apple private key exists
if not os.path.exists(settings.apple_private_key_path):
    print(f"Warning: Apple private key not found at {settings.apple_private_key_path}")
