"""
db_connection.py
----------------
Connects to a local PostgreSQL database using SQLAlchemy and
credentials loaded from a .env file. Never hard-codes secrets.

Usage
-----
    python src/db_connection.py          # runs the connection test
    from db_connection import get_engine  # import the engine elsewhere
"""

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# Resolve .env relative to this file's project root (XAI/.env)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")


def get_engine():
    """Build and return a SQLAlchemy engine from .env credentials."""
    missing = [v for v in ("DB_USER", "DB_PASSWORD", "DB_NAME") if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required .env variable(s): {', '.join(missing)}"
        )

    url = (
        f"postgresql+psycopg2://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(
        url,
        connect_args={"options": f"-csearch_path={DB_SCHEMA}"},
        pool_pre_ping=True,
    )
    return engine


def test_connection() -> bool:
    """
    Run SELECT version(); to verify the database is reachable.

    Returns True on success, False on failure.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).scalar()
        print("Connected successfully.")
        print(f"PostgreSQL version: {version}")
        return True

    except EnvironmentError as exc:
        print(f"Configuration error: {exc}")
    except OperationalError as exc:
        print(
            f"Connection failed: could not reach '{DB_HOST}:{DB_PORT}/{DB_NAME}'.\n"
            f"Check that PostgreSQL is running and the credentials in .env are correct.\n"
            f"Detail: {exc.orig}"
        )
    except SQLAlchemyError as exc:
        print(f"Database error: {exc}")

    return False


if __name__ == "__main__":
    test_connection()
