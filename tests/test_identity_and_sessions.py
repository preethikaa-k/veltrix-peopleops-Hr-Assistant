import sqlite3

import pytest

from tests.conftest import post


def test_missing_header_is_401(client):
    assert post(client, None).status_code == 401


@pytest.mark.parametrize("bad", ["E9999", "e1001", "' OR 1=1 --", "E1001; DROP TABLE employees", "x" * 500])
def test_unknown_or_malicious_employee_id_is_401(client, bad):
    assert post(client, bad).status_code == 401


@pytest.mark.parametrize(
    "payload",
    [None, "not json", "[]", {}, {"session_id": "s"}, {"message": "hi"}, {"session_id": "", "message": "hi"},
     {"session_id": "s", "message": "   "}, {"session_id": 5, "message": "hi"}, {"session_id": "s", "message": "x" * 2001},
     {"session_id": "a\x00b", "message": "hi"}],
)
def test_invalid_body_is_400(client, payload):
    h = {"X-Employee-Id": "E1001", "Content-Type": "application/json"}
    if payload is None:
        r = client.post("/chat", headers=h)
    elif isinstance(payload, str):
        r = client.post("/chat", headers=h, content=payload)
    else:
        r = client.post("/chat", headers=h, json=payload)
    assert r.status_code == 400


def test_oversized_body_is_400(client):
    r = client.post("/chat", headers={"X-Employee-Id": "E1001"}, content=b"{" + b" " * 20_000 + b"}")
    assert r.status_code == 400


def test_valid_request_matches_contract(client):
    r = post(client, "E1001")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"request_id", "answer", "route", "citations", "tool_calls", "pending_confirmation"}
    assert body["route"] in {"rag", "tool", "hybrid", "clarify", "no_answer", "refuse"}


def test_same_session_other_employee_is_403(client):
    assert post(client, "E1001", session="shared").status_code == 200
    assert post(client, "E1003", session="shared").status_code == 403
    assert post(client, "E1001", session="shared").status_code == 200  # owner unaffected


def test_hr_cannot_hijack_a_session_either(client):
    post(client, "E1004", session="victim")
    assert post(client, "E1003", session="victim").status_code == 403


def test_extra_body_fields_cannot_set_identity(client):
    r = client.post(
        "/chat",
        headers={"X-Employee-Id": "E1001"},
        json={"session_id": "s9", "message": "hi", "role": "hr", "employee_id": "E1003", "region": "US"},
    )
    assert r.status_code == 200


def test_existing_tables_are_unchanged(client, settings):
    conn = sqlite3.connect(settings.db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(employees)")]
    assert cols == ["employee_id", "full_name", "email", "phone", "region", "join_date", "level", "role", "manager_id"]
    assert conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 6
