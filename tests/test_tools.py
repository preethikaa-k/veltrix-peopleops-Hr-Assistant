import uuid

import pytest

from app import tools
from tests.conftest import add_request, make_ctx, seed_calendar

PII_KEYS = {"full_name", "email", "phone", "name"}


def bal(conn, emp, t="annual"):
    return conn.execute("SELECT balance_days FROM leave_balances WHERE employee_id=? AND leave_type=?", (emp, t)).fetchone()[0]


# ------------------------------------------------------------------ apply_leave rules (6.2)
def test_apply_success_is_atomic_and_correct(conn):
    ctx = make_ctx(conn, "E1001")                                   # IN: 2026-10-02 is a holiday
    r = tools.apply_leave(ctx, "annual", "2026-09-28", "2026-10-02")
    assert r.status == "success" and r.data["working_days"] == 4
    row = conn.execute("SELECT * FROM leave_requests WHERE request_id=?", (r.data["leave_request_id"],)).fetchone()
    assert row["status"] == "pending" and row["working_days"] == 4 and row["employee_id"] == "E1001"
    assert uuid.UUID(row["request_id"]).version == 4
    assert bal(conn, "E1001") == 6


def test_region_changes_working_days(conn):
    assert tools.check_apply_leave(make_ctx(conn, "E1001"), "annual", "2026-09-28", "2026-10-02").data["working_days"] == 4   # IN holiday
    assert tools.check_apply_leave(make_ctx(conn, "E1005"), "annual", "2026-09-28", "2026-10-02").data["working_days"] == 5   # US, none


def test_range_spanning_weekend_and_holiday(conn):
    r = tools.check_apply_leave(make_ctx(conn, "E1001"), "annual", "2026-10-02", "2026-10-06")
    assert r.data["working_days"] == 2


@pytest.mark.parametrize("args,code", [
    (("annual", "2026-09-19", "2026-09-20"), "zero_working_days"),      # weekend
    (("annual", "2026-10-02", "2026-10-02"), "zero_working_days"),      # public holiday
    (("annual", "2026-09-14", "2026-09-16"), "start_in_past"),
    (("annual", "2026-12-30", "2027-01-05"), "outside_leave_year"),
    (("annual", "2027-02-01", "2027-02-02"), "outside_leave_year"),
    (("annual", "2026-09-22", "2026-09-21"), "end_before_start"),
    (("annual", "2026-09-16", "2026-10-01"), "insufficient_balance"),   # 12 working days > 10
    (("unpaid", "2026-09-22", "2026-09-23"), "invalid_leave_type"),
    ((None, "2026-09-22", "2026-09-23"), "invalid_leave_type"),
    (("annual", "22/09/2026", "2026-09-23"), "invalid_date"),
    (("annual", "2026-02-30", "2026-03-01"), "invalid_date"),
])
def test_rejections_have_specific_reason_and_error_status(conn, args, code):
    before = bal(conn, "E1001")
    r = tools.apply_leave(make_ctx(conn, "E1001"), *args)
    assert r.status == "error" and r.code == code and r.message
    assert bal(conn, "E1001") == before
    assert conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0] == 0


def test_today_is_allowed_as_start(conn):
    assert tools.apply_leave(make_ctx(conn, "E1001"), "annual", "2026-09-15", "2026-09-15").status == "success"


def test_overlap_with_pending_or_approved_rejected_cancelled_ignored(conn):
    add_request(conn, "r-pend", "E1001", "2026-09-22", "2026-09-23", 2, "pending")
    add_request(conn, "r-appr", "E1001", "2026-10-06", "2026-10-07", 2, "approved")
    add_request(conn, "r-canc", "E1001", "2026-11-02", "2026-11-03", 2, "cancelled")
    ctx = make_ctx(conn, "E1001")
    assert tools.apply_leave(ctx, "annual", "2026-09-23", "2026-09-24").code == "overlapping_request"   # shares one date
    assert tools.apply_leave(ctx, "annual", "2026-10-05", "2026-10-06").code == "overlapping_request"
    assert tools.apply_leave(ctx, "sick", "2026-09-22", "2026-09-22").code == "overlapping_request"     # any type
    assert tools.apply_leave(ctx, "annual", "2026-11-02", "2026-11-03").status == "success"            # cancelled doesn't block


def test_repeated_apply_cannot_double_spend(conn):
    ctx = make_ctx(conn, "E1005")
    assert tools.apply_leave(ctx, "annual", "2026-09-28", "2026-09-30").status == "success"
    assert tools.apply_leave(ctx, "annual", "2026-09-28", "2026-09-30").code == "overlapping_request"
    assert bal(conn, "E1005") == 7


def test_failure_mid_transaction_rolls_everything_back(conn, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(tools, "_audit", boom)
    with pytest.raises(RuntimeError):
        tools.apply_leave(make_ctx(conn, "E1005"), "annual", "2026-09-28", "2026-09-30")
    assert bal(conn, "E1005") == 10
    assert conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0] == 0
    assert not conn.in_transaction


def test_check_apply_changes_nothing(conn):
    r = tools.check_apply_leave(make_ctx(conn, "E1005"), "annual", "2026-09-28", "2026-09-30")
    assert r.status == "pending_confirmation"
    assert bal(conn, "E1005") == 10 and conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0] == 0


def test_holiday_calendar_follows_precedence_newer_calendar_wins(conn):
    seed_calendar(conn, "CAL-IN-REV", ["IN"], "2026-06-01", {"2026-09-29": "Revised holiday"})   # newer, same scope/authority
    r = tools.check_apply_leave(make_ctx(conn, "E1001"), "annual", "2026-09-28", "2026-10-02")
    assert r.data["working_days"] == 4          # 09-29 is now the holiday instead of 10-02
    assert tools.check_apply_leave(make_ctx(conn, "E1001"), "annual", "2026-10-02", "2026-10-02").status == "pending_confirmation"


def test_future_calendar_is_ignored(conn):
    seed_calendar(conn, "CAL-IN-FUT", ["IN"], "2026-12-01", {"2026-09-29": "Not yet effective"})
    assert tools.check_apply_leave(make_ctx(conn, "E1001"), "annual", "2026-09-29", "2026-09-29").status == "pending_confirmation"


def test_no_calendar_fails_closed(conn):
    r = tools.check_apply_leave(make_ctx(conn, "E1006"), "annual", "2026-09-28", "2026-09-29")   # DE: none seeded
    assert r.status == "error" and r.code == "holiday_calendar_unavailable"


# ------------------------------------------------------------------ cancel_leave (6.3)
def test_cancel_returns_days_and_marks_cancelled(conn):
    ctx = make_ctx(conn, "E1005")
    rid = tools.apply_leave(ctx, "annual", "2026-09-28", "2026-09-30").data["leave_request_id"]
    r = tools.cancel_leave(ctx, rid)
    assert r.status == "success" and bal(conn, "E1005") == 10
    assert conn.execute("SELECT status FROM leave_requests WHERE request_id=?", (rid,)).fetchone()[0] == "cancelled"
    again = tools.cancel_leave(ctx, rid)
    assert again.status == "error" and again.code == "not_cancellable" and bal(conn, "E1005") == 10   # no second refund


def test_cancel_only_future_pending_or_approved(conn):
    add_request(conn, "r-today", "E1005", "2026-09-15", "2026-09-16", 2, "approved")
    add_request(conn, "r-past", "E1005", "2026-09-01", "2026-09-02", 2, "approved")
    add_request(conn, "r-tmrw", "E1005", "2026-09-16", "2026-09-17", 2, "pending")
    add_request(conn, "r-canc", "E1005", "2026-10-01", "2026-10-02", 2, "cancelled")
    ctx = make_ctx(conn, "E1005")
    assert tools.cancel_leave(ctx, "r-today").code == "already_started"      # start must be AFTER today
    assert tools.cancel_leave(ctx, "r-past").code == "already_started"
    assert tools.cancel_leave(ctx, "r-canc").code == "not_cancellable"
    assert tools.cancel_leave(ctx, "r-tmrw").status == "success" and bal(conn, "E1005") == 12


def test_cannot_cancel_someone_elses_or_unknown_request_same_message(conn):
    add_request(conn, "r-other", "E1004", "2026-10-05", "2026-10-06", 2, "pending")
    ctx = make_ctx(conn, "E1005")
    a, b = tools.cancel_leave(ctx, "r-other"), tools.cancel_leave(ctx, "does-not-exist")
    assert a.status == b.status == "error" and a.code == b.code and a.message == b.message
    assert conn.execute("SELECT status FROM leave_requests WHERE request_id='r-other'").fetchone()[0] == "pending"
    assert bal(conn, "E1004") == 10


def test_hr_and_manager_cannot_cancel_or_apply_for_others(conn):
    add_request(conn, "r-x", "E1005", "2026-10-05", "2026-10-06", 2, "pending")
    assert tools.cancel_leave(make_ctx(conn, "E1002"), "r-x").code == "request_not_found"    # manager
    assert tools.cancel_leave(make_ctx(conn, "E1003"), "r-x").code == "request_not_found"    # hr
    import inspect
    assert list(inspect.signature(tools.apply_leave).parameters)[1:] == ["leave_type", "start_date", "end_date"]   # no way to name another employee


# ------------------------------------------------------------------ read access control (3.1, 7.3)
def test_employee_sees_only_self(conn):
    ctx = make_ctx(conn, "E1001")
    assert tools.get_leave_balance(ctx).status == "success"
    assert tools.get_leave_balance(ctx, "E1001").data["balances"] == {"annual": 10, "sick": 12}
    for other in ("E1004", "E1002", "E1003"):
        assert tools.get_leave_balance(ctx, other).message == tools.REFUSAL
        assert tools.list_leave_requests(ctx, other).message == tools.REFUSAL


def test_refusal_identical_for_existing_and_nonexistent(conn):
    ctx = make_ctx(conn, "E1001")
    a, b = tools.get_leave_balance(ctx, "E1004"), tools.get_leave_balance(ctx, "E9999")
    assert (a.status, a.code, a.message) == (b.status, b.code, b.message) == ("error", "forbidden", tools.REFUSAL)
    c, d = tools.list_leave_requests(ctx, "E1004"), tools.list_leave_requests(ctx, "E9999")
    assert (c.code, c.message) == (d.code, d.message)


def test_manager_sees_self_and_direct_reports_only(conn):
    ctx = make_ctx(conn, "E1002")                                 # reports: E1001, E1004, E1005
    for ok in ("E1002", "E1001", "E1004", "E1005"):
        assert tools.get_leave_balance(ctx, ok).status == "success"
    for no in ("E1003", "E1006", "E9999"):                        # HR, non-report, nonexistent
        r = tools.get_leave_balance(ctx, no)
        assert r.status == "error" and r.message == tools.REFUSAL


def test_hr_sees_any_one_employee_but_not_phantoms(conn):
    ctx = make_ctx(conn, "E1003")
    assert tools.get_leave_balance(ctx, "E1006").status == "success"
    assert tools.list_leave_requests(ctx, "E1001").status == "success"
    assert tools.get_leave_balance(ctx, "E9999").code == "not_found"


@pytest.mark.parametrize("bad", ["E1001; DROP TABLE employees", "*", "all", "E1001,E1004", ["E1001", "E1004"], "", "x" * 99])
def test_bulk_or_malformed_employee_ids_never_widen_access(conn, bad):
    r = tools.get_leave_balance(make_ctx(conn, "E1003"), bad)     # even HR: one employee at a time
    assert r.status == "error" or r.data.get("employee_id") == "E1003"
    if r.status == "error":
        assert r.code in ("forbidden", "not_found")


def test_team_leave_is_managers_only_and_scoped_to_direct_reports(conn):
    add_request(conn, "t1", "E1001", "2026-10-05", "2026-10-07", 3, "approved")
    add_request(conn, "t2", "E1004", "2026-10-07", "2026-10-09", 3, "pending")
    add_request(conn, "t3", "E1004", "2026-10-12", "2026-10-13", 2, "cancelled")
    add_request(conn, "t4", "E1006", "2026-10-06", "2026-10-06", 1, "approved")      # not a report
    add_request(conn, "t5", "E1002", "2026-10-06", "2026-10-06", 1, "approved")      # the manager's own
    add_request(conn, "t6", "E1005", "2026-11-02", "2026-11-03", 2, "approved")      # outside range
    r = tools.get_team_leave(make_ctx(conn, "E1002"), "2026-10-07", "2026-10-07")
    assert r.status == "success" and {x["request_id"] for x in r.data["requests"]} == {"t1", "t2"}   # inclusive overlap
    for role_emp in ("E1001", "E1003"):                                                  # employee, hr
        assert tools.get_team_leave(make_ctx(conn, role_emp), "2026-10-01", "2026-10-31").code == "forbidden"


def test_team_leave_validates_dates(conn):
    ctx = make_ctx(conn, "E1002")
    assert tools.get_team_leave(ctx, "bad", "2026-10-01").code == "invalid_date"
    assert tools.get_team_leave(ctx, "2026-10-05", "2026-10-01").code == "end_before_start"


def test_tool_outputs_contain_no_personal_data(conn):
    add_request(conn, "p1", "E1001", "2026-10-05", "2026-10-06", 2, "pending")
    ctx = make_ctx(conn, "E1002")
    for r in (tools.get_leave_balance(ctx, "E1001"), tools.list_leave_requests(ctx, "E1001"), tools.get_team_leave(ctx, "2026-10-01", "2026-10-31")):
        blob = str(r.data) + str(r.args) + r.message
        assert not (PII_KEYS & set(blob.replace("'", " ").split())) and "@" not in blob and "+91" not in blob


def test_audit_log_is_append_only(conn):
    tools.apply_leave(make_ctx(conn, "E1005"), "annual", "2026-09-28", "2026-09-29")
    assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 1
    import sqlite3
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE audit_log SET outcome='x'")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM audit_log")
