import os
from logging.config import fileConfig
from typing import Any

from sqlalchemy import engine_from_config, pool

from alembic import context

# ---------------------------------------------------------------------------
# Alembic config
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Wire up the ORM metadata so that `alembic revision --autogenerate` can
# detect schema changes automatically.
from qe.models import Base  # noqa: E402

target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------


def _get_url() -> str:
    """
    Resolve the database URL in priority order:
      1. DATABASE_URL environment variable (e.g. set in CI or .env)
      2. Individual PG* environment variables (same defaults as qe/db.py)
      3. sqlalchemy.url from alembic.ini (fallback for local dev)
    """
    if url := os.getenv("DATABASE_URL"):
        return url

    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5433")
    user = os.getenv("PGUSER", "qe")
    password = os.getenv("PGPASSWORD", "qe")
    dbname = os.getenv("PGDATABASE", "qe")
    if all([host, port, user, password, dbname]):
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"

    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise ValueError(
            "No database URL found. Set DATABASE_URL, the PG* env vars, "
            "or sqlalchemy.url in alembic.ini."
        )
    return url


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    section: dict[str, Any] = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _get_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Emit `COMMENT ON` DDL for server_default comparisons during autogenerate
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
