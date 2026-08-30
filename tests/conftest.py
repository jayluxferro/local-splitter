"""Shared pytest helpers for local-splitter (Postgres backend).

The suite runs against a dedicated ``local_splitter_test`` database on
the local Postgres instance (override via ``LOCAL_SPLITTER_TEST_DB_URL``
for CI / other machines).  Tests that need the cache drop the known
tables before constructing their store; the v1 migration recreates
them on the next open.

The schema itself is NEVER dropped — ``DROP SCHEMA public CASCADE`` on
Postgres 15+ destroys the schema-bound ``vector`` extension and strips
grants on the recreated schema.  Drops are scoped to exactly the tables
the migrations can create, nothing else.
"""

from __future__ import annotations

import os

import psycopg

TEST_DB_URL = os.environ.get(
    "LOCAL_SPLITTER_TEST_DB_URL",
    "postgresql://local_splitter@localhost:5432/local_splitter_test",
)

# Every table the v1 migration can create.  Drops are scoped to exactly
# this set — keep in sync with sem_cache.py's migrations.
KNOWN_TABLES = ("cache_entry", "schema_migrations")


def drop_cache_tables(dsn: str = TEST_DB_URL) -> None:
    """Drop the known cache tables via a dedicated connection.

    Uses a fresh short-lived connection because the store's own
    connection is opened only after this drop (and closed by its
    caller before the next test).  Autocommit keeps each DROP outside
    any transaction (no locking surprises).
    """
    conn = psycopg.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for table in KNOWN_TABLES:
                cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
    finally:
        conn.close()
