# backend/app/db.py
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from .config import settings

dsn = (
    f"postgresql+psycopg://{settings.pg_user}:{settings.pg_password}"
    f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"
)

engine: Engine = create_engine(dsn, pool_pre_ping=True)
