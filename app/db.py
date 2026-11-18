"""SQLite access. Parameterised queries only. We never alter existing tables, only add ours."""
from __future__ import annotations

import logging
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("veltrix.db")

EMPLOYEE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")

_ADDED_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_employee ON chat_sessions(employee_id);

-- Mirror of manifest.json. Eligibility is decided from THIS metadata, never from document text.
CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    file           TEXT NOT NULL,
    title          TEXT NOT NULL,
    doc_type       TEXT NOT NULL,
    regions        TEXT NOT NULL,            -- JSON array
    effective_date TEXT NOT NULL,
    access_level   TEXT NOT NULL,
    content_hash   TEXT,
    extraction     TEXT NOT NULL DEFAULT 'text',   -- text | ocr | mixed
    ocr_confidence REAL,
    status         TEXT NOT NULL DEFAULT 'ok'
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id   TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL,
    locator  TEXT NOT NULL,
    kind     TEXT NOT NULL,                  -- text | table
    text     TEXT NOT NULL,
    tokens   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- Public holidays extracted automatically from calendar documents during ingestion.
CREATE TABLE IF NOT EXISTS holidays (
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    holiday_date TEXT NOT NULL,
    name         TEXT NOT NULL,
    PRIMARY KEY (doc_id, holiday_date)
);

-- At most one proposal waiting per session (spec 8.2).
CREATE TABLE IF NOT EXISTS pending_proposals (
    session_id  TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL,
    action      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- Append-only audit trail of write attempts. Employee IDs only, no personal data.
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    employee_id TEXT NOT NULL,
    action      TEXT NOT NULL,
    request_id  TEXT,
    outcome     TEXT NOT NULL,
    detail      TEXT
);
CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""


@dataclass(frozen=True)
class Identity:
    """Who is asking. Deliberately contains NO personal data (name/email/phone),
    so it can be passed anywhere (prompts, logs) without leaking anything."""

    employee_id: str
    region: str
    role: str
    level: str
    manager_id: Optional[str]
    join_date: str


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    # isolation_level=None: autocommit; write paths open explicit BEGIN IMMEDIATE transactions.
    conn = sqlite3.connect(str(db_path), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        yield conn
    finally:
        conn.close()


def init_schema(db_path: Path) -> None:
    """Add our own tables. Existing tables/columns are never touched."""
    with connect(db_path) as conn:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            log.warning("could not enable WAL; continuing with default journal mode")
        conn.executescript(_ADDED_SCHEMA)


def get_identity(conn: sqlite3.Connection, employee_id: str) -> Optional[Identity]:
    """The database row is the ONLY source of identity, role and region."""
    if not EMPLOYEE_ID_RE.match(employee_id or ""):
        return None
    row = conn.execute(
        "SELECT employee_id, region, role, level, manager_id, join_date FROM employees WHERE employee_id = ?",
        (employee_id,),
    ).fetchone()
    return Identity(**dict(row)) if row else None


def load_sensitive_literals(conn: sqlite3.Connection) -> list[str]:
    """Names/emails/phones, used ONLY to seed the log scrubber."""
    out: list[str] = []
    for r in conn.execute("SELECT full_name, email, phone FROM employees"):
        out.extend([r["full_name"], r["email"], r["phone"]])
    return out


def claim_session(conn: sqlite3.Connection, session_id: str, employee_id: str) -> bool:
    """Bind a session to its first sender. Atomic via the primary key, so two racing
    first messages from different employees cannot both win. Returns False if the
    session belongs to someone else (caller returns HTTP 403)."""
    conn.execute(
        "INSERT OR IGNORE INTO chat_sessions(session_id, employee_id, created_at) VALUES (?, ?, ?)",
        (session_id, employee_id, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    row = conn.execute("SELECT employee_id FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
    return row is not None and row["employee_id"] == employee_id
