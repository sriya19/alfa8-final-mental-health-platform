# backend/app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from typing import Optional


class Settings(BaseSettings):
    # Core
    PROJECT_NAME: str = "mental_health_platform"
    ENVIRONMENT: str = "dev"
    TZ: str = "America/New_York"

    # Primary PG_* (preferred) – leave as optional so we can fall back to POSTGRES_*
    PG_HOST: Optional[str] = None
    PG_PORT: Optional[int] = None
    PG_USER: Optional[str] = None
    PG_PASSWORD: Optional[str] = None
    PG_DB: Optional[str] = None

    # S3 / MinIO
    S3_ENDPOINT: str = "http://minio:9000"
    S3_BUCKET: str = "mh-raw"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    AWS_DEFAULT_REGION: str = "us-east-1"

    # --- Socrata (MISSING BEFORE) ---
    _SOCRATA_APP_TOKEN: str | None = None
    SOCRATA_TIMEOUT: int = 20

    # Optional LLM
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None

    # Allow unknown envs (so POSTGRES_* exists on Settings too)
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
        case_sensitive=False,
    )

    # Clean getters that fall back from PG_* → POSTGRES_*
    @property
    def pg_host(self) -> str:
        return self.PG_HOST or os.getenv("POSTGRES_HOST") or "pg"

    @property
    def pg_port(self) -> int:
        v = self.PG_PORT or os.getenv("POSTGRES_PORT") or 5432
        return int(v)

    @property
    def pg_user(self) -> str:
        return self.PG_USER or os.getenv("POSTGRES_USER") or "app_user"

    @property
    def pg_password(self) -> str:
        return self.PG_PASSWORD or os.getenv("POSTGRES_PASSWORD") or "please-change"

    @property
    def pg_db(self) -> str:
        return self.PG_DB or os.getenv("POSTGRES_DB") or "mh_catalog"

    @property
    def pg_db(self) -> str:
        return self.PG_DB or os.getenv("POSTGRES_DB") or "mh_catalog"

    # Add this new property:
    @property
    def SOCRATA_APP_TOKEN(self) -> str | None:
        """Strip whitespace from Socrata token"""
        token = self._SOCRATA_APP_TOKEN or os.getenv("SOCRATA_APP_TOKEN")
        if token:
            return token.strip()
        return None

settings = Settings()
