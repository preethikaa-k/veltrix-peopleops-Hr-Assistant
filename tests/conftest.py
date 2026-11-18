import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings

# Schema copied exactly from the assignment spec (section 2.3).
SPEC_SCHEMA = """
CREATE TABLE employees (
  employee_id TEXT PRIMARY KEY, full_name TEXT NOT NULL, email TEXT NOT NULL, phone TEXT NOT NULL,
  region TEXT NOT NULL, join_date TEXT NOT NULL, level TEXT NOT NULL, role TEXT NOT NULL,
  manager_id TEXT REFERENCES employees(employee_id));
CREATE TABLE leave_balances (
  employee_id TEXT NOT NULL REFERENCES employees(employee_id), leave_type TEXT NOT NULL,
  balance_days REAL NOT NULL, PRIMARY KEY (employee_id, leave_type));
CREATE TABLE leave_requests (
  request_id TEXT PRIMARY KEY, employee_id TEXT NOT NULL REFERENCES employees(employee_id),
  leave_type TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL,
  working_days INTEGER NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
"""

EMPLOYEES = [
    ("E1001", "Riya Sharma", "riya.sharma@veltrix.example", "+91 98765 43210", "IN", "2022-01-10", "L3", "employee", "E1002"),
    ("E1002", "Marcus Webb", "marcus.webb@veltrix.example", "+1 415 555 0134", "US", "2019-03-04", "L5", "manager", None),
    ("E1003", "Hanna Vogel", "hanna.vogel@veltrix.example", "+49 30 1234 5678", "DE", "2020-06-01", "L4", "hr", None),
    ("E1004", "Arjun Rao", "arjun.rao@veltrix.example", "+91 91234 56789", "IN", "2023-02-20", "L2", "employee", "E1002"),
    ("E1005", "Dana Cole", "dana.cole@veltrix.example", "+1 512 555 0188", "US", "2024-05-13", "L3", "employee", "E1002"),
    ("E1006", "Lukas Brandt", "lukas.brandt@veltrix.example", "+49 30 9876 5432", "DE", "2021-09-01", "L3", "employee", None),
]


@pytest.fixture()
def settings(tmp_path):
    db_path = tmp_path / "veltrix.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(SPEC_SCHEMA)
    conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?)", EMPLOYEES)
    conn.commit()
    conn.close()
    return Settings.from_env(
        {"DB_PATH": str(db_path), "CORPUS_DIR": str(tmp_path), "MANIFEST_PATH": str(tmp_path / "m.json"), "APP_TODAY": "2026-09-15"}
    )


def wait_ready(client, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if client.get("/health").status_code == 200:
            return
        time.sleep(0.02)
    raise AssertionError("service never became ready")


@pytest.fixture()
def client(settings):
    app = create_app(settings, ingest_fn=lambda s: None)
    with TestClient(app) as c:
        wait_ready(c)
        yield c


def post(client, employee, session="s1", message="hello", **kw):
    headers = {"X-Employee-Id": employee} if employee is not None else {}
    return client.post("/chat", json={"session_id": session, "message": message}, headers=headers, **kw)


# ---- helpers for tools/ingestion tests ---------------------------------------------------------
import json
from datetime import date

from app import db as _db
from app.tools import ToolContext


@pytest.fixture()
def conn(settings):
    """Real SQLite connection with our added tables, balances and two calendars seeded."""
    _db.init_schema(settings.db_path)
    with _db.connect(settings.db_path) as c:
        for e in EMPLOYEES:
            c.execute("INSERT INTO leave_balances VALUES (?, 'annual', 10)", (e[0],))
            c.execute("INSERT INTO leave_balances VALUES (?, 'sick', 12)", (e[0],))
        seed_calendar(c, "CAL-IN", ["IN"], "2026-01-01", {"2026-10-02": "Gandhi Jayanti", "2026-10-20": "Vijayadashami"})
        seed_calendar(c, "CAL-US", ["US"], "2026-01-01", {"2026-11-26": "Thanksgiving", "2026-11-27": "Day after"})
        yield c


def seed_calendar(c, doc_id, regions, eff, hols, access="all"):
    c.execute("INSERT INTO documents(doc_id,file,title,doc_type,regions,effective_date,access_level) VALUES (?,?,?,?,?,?,?)",
              (doc_id, doc_id + ".pdf", doc_id, "calendar", json.dumps(regions), eff, access))
    for d, n in hols.items():
        c.execute("INSERT INTO holidays VALUES (?,?,?)", (doc_id, d, n))


def make_ctx(c, employee_id, today="2026-09-15"):
    ident = _db.get_identity(c, employee_id)
    return ToolContext(c, ident, date.fromisoformat(today), "req-test")


def add_request(c, rid, emp, start, end, days, status="approved", leave_type="annual"):
    c.execute("INSERT INTO leave_requests VALUES (?,?,?,?,?,?,?,?)", (rid, emp, leave_type, start, end, days, status, "2026-01-01T00:00:00"))
