"""Targeted tests for side-by-side tables (e.g. POL-LEAVE-GLOBAL-V2 Section 3).

The v2 policy document contains a summary table displaying both Version 1 and Version 2 values side-by-side.
These tests verify that the system always quotes the currently active Version 2 value and NEVER quotes
the superseded Version 1 number for current policy inquiries.
"""
import pytest
from app.api import create_app
from fastapi.testclient import TestClient
from tests.conftest import wait_ready, post, EMPLOYEES, seed_calendar
import sqlite3
from app.config import Settings
from app import db


@pytest.fixture()
def client_with_corpus(tmp_path):
    dbp = tmp_path / "veltrix.db"
    conn = sqlite3.connect(dbp)
    from tests.conftest import SPEC_SCHEMA
    conn.executescript(SPEC_SCHEMA)
    conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?)", EMPLOYEES)
    conn.commit()
    conn.close()

    s = Settings.from_env({
        "DB_PATH": str(dbp),
        "CORPUS_DIR": "data/corpus",
        "MANIFEST_PATH": "data/manifest.json",
        "APP_TODAY": "2026-09-15",
        "LLM_PROVIDER": "mock",
    })
    db.init_schema(dbp)

    # Ingest corpus
    from app.ingestion import run_ingestion
    run_ingestion(s)

    app = create_app(s, ingest_fn=lambda _: None)
    with TestClient(app) as c:
        wait_ready(c)
        yield c


def test_sick_leave_quotes_v2_not_v1(client_with_corpus):
    r = post(client_with_corpus, "E1001", message="How many sick days do I get per year?")
    assert r.status_code == 200
    data = r.json()
    assert data["route"] == "rag"
    ans = data["answer"]
    assert "12" in ans, f"Expected 12 sick days in answer, got: {ans}"
    assert "10" not in ans or "version 1" in ans.lower(), f"Unexpected quote of old 10 days without qualification: {ans}"
    assert any(c["doc_id"] == "POL-LEAVE-GLOBAL-V2" for c in data["citations"])


def test_medical_certificate_quotes_v2_not_v1(client_with_corpus):
    r = post(client_with_corpus, "E1001", message="When is a medical certificate or doctor's note required for sick leave?")
    assert r.status_code == 200
    data = r.json()
    assert data["route"] == "rag"
    ans = data["answer"]
    assert "3" in ans, f"Expected 3 consecutive working days in answer, got: {ans}"
    assert any(c["doc_id"] == "POL-LEAVE-GLOBAL-V2" for c in data["citations"])


def test_parental_leave_primary_quotes_v2_not_v1(client_with_corpus):
    r = post(client_with_corpus, "E1001", message="How long is paid parental leave for a primary caregiver?")
    assert r.status_code == 200
    data = r.json()
    assert data["route"] == "rag"
    ans = data["answer"]
    assert "20" in ans, f"Expected 20 weeks in answer, got: {ans}"
    assert any(c["doc_id"] == "POL-LEAVE-GLOBAL-V2" for c in data["citations"])


def test_parental_leave_secondary_quotes_v2_not_v1(client_with_corpus):
    r = post(client_with_corpus, "E1001", message="How much paid parental leave does a secondary caregiver get?")
    assert r.status_code == 200
    data = r.json()
    assert data["route"] == "rag"
    ans = data["answer"]
    assert "4" in ans, f"Expected 4 weeks in answer, got: {ans}"
    assert any(c["doc_id"] == "POL-LEAVE-GLOBAL-V2" for c in data["citations"])


def test_bereavement_leave_quotes_v2_not_v1(client_with_corpus):
    r = post(client_with_corpus, "E1001", message="How many days of bereavement leave can I take?")
    assert r.status_code == 200
    data = r.json()
    assert data["route"] == "rag"
    ans = data["answer"]
    assert "5" in ans, f"Expected 5 days in answer, got: {ans}"
    assert any(c["doc_id"] == "POL-LEAVE-GLOBAL-V2" for c in data["citations"])


def test_carried_forward_deadline_quotes_v2_not_v1(client_with_corpus):
    # For a general/global or DE employee, carry forward deadline is 31 March
    r = post(client_with_corpus, "E1006", message="What is the deadline to use carried forward annual leave?")
    assert r.status_code == 200
    data = r.json()
    assert data["route"] == "rag"
    ans = data["answer"]
    assert "31 March" in ans or "March 31" in ans, f"Expected 31 March in answer, got: {ans}"
    assert "30 June" not in ans, f"Should not quote superseded 30 June deadline: {ans}"
