from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # Database
    database_url: str

    # API Keys
    openai_api_key: str
    google_vision_api_key: str
    elevenlabs_api_key: str

    # Apple Sign In
    apple_team_id: str
    apple_key_id: str
    apple_private_key_path: str = "./apple_private_key.p8"

    # JWT & Security
    jwt_secret_key: str
    secret_key: str  # Additional secret key
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30 * 24 * 60  # 30 days

    # App Settings
    debug: bool = False
    port: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = False


# Load settings
settings = Settings()

# Validate Apple private key exists
if not os.path.exists(settings.apple_private_key_path):
    print(f"Warning: Apple private key not found at {settings.apple_private_key_path}")

# Set Google Vision API key as environment variable (required by Google client)
if settings.google_vision_api_key:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'] = f'{{"type": "service_account", "private_key": "dummy", "client_email": "dummy@example.com", "project_id": "dummy"}}'
    os.environ['GOOGLE_VISION_API_KEY'] = settings.google_vision_api_key