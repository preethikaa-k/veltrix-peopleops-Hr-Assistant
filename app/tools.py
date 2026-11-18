"""The five leave tools (spec section 6).

Security model: every permission check and every business rule runs HERE, using the identity that
came from the X-Employee-Id header (ToolContext.identity). Nothing depends on the prompt or on
arguments chosen by the LLM. Write tools run in a single BEGIN IMMEDIATE transaction and re-validate
inside it, so concurrent or repeated calls cannot double-spend a balance.
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

from app.dates import parse_iso, working_days
from app.db import EMPLOYEE_ID_RE, Identity
from app.holidays import HolidayError, holidays_for_range

log = logging.getLogger("veltrix.tools")

LEAVE_TYPES = ("annual", "sick")
REQUEST_STATUSES = ("pending", "approved", "cancelled")
LEAVE_YEAR = 2026
# One fixed message for "not allowed" AND "does not exist": no existence oracle (spec 7.3).
REFUSAL = "I can't share information about that employee."


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    identity: Identity
    today: date
    request_id: Optional[str] = None


@dataclass
class ToolResult:
    name: str
    args: dict[str, Any]
    status: str                      # success | error | pending_confirmation
    message: str = ""                # specific reason on error, human-readable summary otherwise
    code: Optional[str] = None       # machine-readable error code
    data: dict[str, Any] = field(default_factory=dict)

    def call_record(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args, "status": self.status}


class RuleViolation(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _err(name: str, args: dict, code: str, message: str) -> ToolResult:
    return ToolResult(name, args, "error", message, code)


def _audit(conn: sqlite3.Connection, ctx: ToolContext, action: str, outcome: str, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO audit_log(ts, employee_id, action, request_id, outcome, detail) VALUES (?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), ctx.identity.employee_id, action,
         ctx.request_id, outcome, detail[:200]),
    )


# --------------------------------------------------------------------------- read permissions
def _resolve_target(ctx: ToolContext, employee_id: Any) -> tuple[Optional[str], bool, dict]:
    """Returns (target_id, denied, args). Identical refusal whether the employee exists or not."""
    target = ctx.identity.employee_id if employee_id in (None, "") else employee_id
    args = {"employee_id": target if isinstance(target, str) else str(target)}
    if not isinstance(target, str) or not EMPLOYEE_ID_RE.match(target):
        return None, True, args
    ident = ctx.identity
    if target == ident.employee_id:
        return target, False, args
    if ident.role == "hr":
        return target, False, args                      # HR: any ONE employee at a time
    if ident.role == "manager":
        row = ctx.conn.execute(
            "SELECT 1 FROM employees WHERE employee_id = ? AND manager_id = ?", (target, ident.employee_id)
        ).fetchone()
        if row:
            return target, False, args                  # direct report
    return None, True, args


def _exists(conn: sqlite3.Connection, employee_id: str) -> bool:
    return conn.execute("SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)).fetchone() is not None


def get_leave_balance(ctx: ToolContext, employee_id: Any = None, leave_type: Any = None) -> ToolResult:
    name = "get_leave_balance"
    target, deny, args = _resolve_target(ctx, employee_id)
    if leave_type not in (None, ""):
        args["leave_type"] = leave_type if isinstance(leave_type, str) else str(leave_type)
    if deny:
        return _err(name, args, "forbidden", REFUSAL)
    if leave_type not in (None, "") and leave_type not in LEAVE_TYPES:
        return _err(name, args, "invalid_leave_type", "Leave type must be 'annual' or 'sick'.")
    if not _exists(ctx.conn, target):
        return _err(name, args, "not_found", "No employee with that ID was found.")  # only reachable by HR
    sql, params = "SELECT leave_type, balance_days FROM leave_balances WHERE employee_id = ?", [target]
    if leave_type:
        sql, params = sql + " AND leave_type = ?", params + [leave_type]
    rows = ctx.conn.execute(sql + " ORDER BY leave_type", params).fetchall()
    balances = {r["leave_type"]: r["balance_days"] for r in rows}
    return ToolResult(name, args, "success", "Current balance (future accrual not included).",
                      data={"employee_id": target, "balances": balances})


def list_leave_requests(ctx: ToolContext, employee_id: Any = None, status: Any = None) -> ToolResult:
    name = "list_leave_requests"
    target, deny, args = _resolve_target(ctx, employee_id)
    if status not in (None, ""):
        args["status"] = status if isinstance(status, str) else str(status)
    if deny:
        return _err(name, args, "forbidden", REFUSAL)
    if status not in (None, "") and status not in REQUEST_STATUSES:
        return _err(name, args, "invalid_status", "Status must be 'pending', 'approved' or 'cancelled'.")
    if not _exists(ctx.conn, target):
        return _err(name, args, "not_found", "No employee with that ID was found.")
    sql = ("SELECT request_id, leave_type, start_date, end_date, working_days, status "
           "FROM leave_requests WHERE employee_id = ?")
    params: list[Any] = [target]
    if status:
        sql, params = sql + " AND status = ?", params + [status]
    rows = ctx.conn.execute(sql + " ORDER BY start_date, request_id", params).fetchall()
    return ToolResult(name, args, "success", f"{len(rows)} request(s).",
                      data={"employee_id": target, "requests": [dict(r) for r in rows]})


def get_team_leave(ctx: ToolContext, start_date: Any, end_date: Any) -> ToolResult:
    name = "get_team_leave"
    args = {"start_date": str(start_date), "end_date": str(end_date)}
    if ctx.identity.role != "manager":
        return _err(name, args, "forbidden", "Team leave is available to managers only.")
    try:
        s, e = parse_iso(start_date), parse_iso(end_date)
    except ValueError:
        return _err(name, args, "invalid_date", "Dates must be valid calendar dates in YYYY-MM-DD format.")
    if e < s:
        return _err(name, args, "end_before_start", "The end date must be on or after the start date.")
    rows = ctx.conn.execute(
        """SELECT r.employee_id, r.request_id, r.leave_type, r.start_date, r.end_date, r.working_days, r.status
           FROM leave_requests r JOIN employees e ON e.employee_id = r.employee_id
           WHERE e.manager_id = ? AND r.status IN ('pending','approved')
             AND r.start_date <= ? AND r.end_date >= ?
           ORDER BY r.start_date, r.employee_id""",
        (ctx.identity.employee_id, e.isoformat(), s.isoformat()),
    ).fetchall()
    return ToolResult(name, args, "success", f"{len(rows)} request(s) from your direct reports overlap that range.",
                      data={"requests": [dict(r) for r in rows]})


# --------------------------------------------------------------------------- apply_leave
def _validate_apply(ctx: ToolContext, leave_type: Any, start_date: Any, end_date: Any) -> tuple[int, date, date]:
    """All rules from spec 6.2. Reads current data; call inside the write transaction to be race-safe."""
    if leave_type not in LEAVE_TYPES:
        raise RuleViolation("invalid_leave_type", "Leave type must be 'annual' or 'sick'.")
    try:
        s, e = parse_iso(start_date), parse_iso(end_date)
    except ValueError:
        raise RuleViolation("invalid_date", "Dates must be valid calendar dates in YYYY-MM-DD format.")
    if s.year != LEAVE_YEAR or e.year != LEAVE_YEAR:
        raise RuleViolation("outside_leave_year", f"Leave can only be requested within {LEAVE_YEAR}; both dates must fall in {LEAVE_YEAR}.")
    if s < ctx.today:
        raise RuleViolation("start_in_past", f"The start date {s} is before today ({ctx.today}); leave can't start in the past.")
    if e < s:
        raise RuleViolation("end_before_start", f"The end date {e} is before the start date {s}.")
    try:
        hol = holidays_for_range(ctx.conn, ctx.identity, ctx.today, s, e)
    except HolidayError as exc:
        raise RuleViolation("holiday_calendar_unavailable", str(exc))
    days = working_days(s, e, hol)
    if days == 0:
        raise RuleViolation("zero_working_days",
                            f"{s} to {e} contains no working days (weekends and public holidays are not counted).")
    row = ctx.conn.execute("SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = ?",
                           (ctx.identity.employee_id, leave_type)).fetchone()
    balance = row["balance_days"] if row else 0.0
    if days > balance:
        raise RuleViolation("insufficient_balance",
                            f"This request needs {days} working day(s) but your {leave_type} leave balance is {balance:g}.")
    clash = ctx.conn.execute(
        """SELECT request_id, status, start_date, end_date FROM leave_requests
           WHERE employee_id = ? AND status IN ('pending','approved') AND start_date <= ? AND end_date >= ?
           ORDER BY start_date LIMIT 1""",
        (ctx.identity.employee_id, e.isoformat(), s.isoformat())).fetchone()
    if clash:
        raise RuleViolation("overlapping_request",
                            f"These dates overlap your {clash['status']} request {clash['request_id']} "
                            f"({clash['start_date']} to {clash['end_date']}).")
    return days, s, e


def _apply_args(leave_type: Any, start_date: Any, end_date: Any) -> dict:
    return {"leave_type": str(leave_type) if leave_type is not None else "",
            "start_date": str(start_date), "end_date": str(end_date)}


def check_apply_leave(ctx: ToolContext, leave_type: Any, start_date: Any, end_date: Any) -> ToolResult:
    """Step 1 of spec 8.2: validate and propose. Changes nothing."""
    args = _apply_args(leave_type, start_date, end_date)
    try:
        days, s, e = _validate_apply(ctx, leave_type, start_date, end_date)
    except RuleViolation as v:
        return _err("apply_leave", args, v.code, v.message)
    return ToolResult("apply_leave", args, "pending_confirmation",
                      f"{leave_type} leave from {s} to {e}: {days} working day(s).",
                      data={"working_days": days})


def apply_leave(ctx: ToolContext, leave_type: Any, start_date: Any, end_date: Any) -> ToolResult:
    """Step 2 of spec 8.2: execute. Validation is repeated INSIDE the transaction because data may have
    changed since the proposal. Insert + working_days + balance deduction commit together or not at all."""
    args = _apply_args(leave_type, start_date, end_date)
    conn = ctx.conn
    conn.execute("BEGIN IMMEDIATE")
    try:
        days, s, e = _validate_apply(ctx, leave_type, start_date, end_date)
        request_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO leave_requests(request_id, employee_id, leave_type, start_date, end_date, working_days, status, created_at)"
            " VALUES (?,?,?,?,?,?, 'pending', ?)",
            (request_id, ctx.identity.employee_id, leave_type, s.isoformat(), e.isoformat(), days,
             datetime.now().isoformat(timespec="seconds")))
        cur = conn.execute(
            "UPDATE leave_balances SET balance_days = balance_days - ? WHERE employee_id = ? AND leave_type = ?",
            (days, ctx.identity.employee_id, leave_type))
        if cur.rowcount != 1:
            raise RuleViolation("insufficient_balance", f"You have no {leave_type} leave balance.")
        _audit(conn, ctx, "apply_leave", "success", f"{request_id} {days}d")
        conn.execute("COMMIT")
    except RuleViolation as v:
        conn.execute("ROLLBACK")
        _audit(conn, ctx, "apply_leave", f"rejected:{v.code}")
        return _err("apply_leave", args, v.code, v.message)
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    log.info("leave applied", extra={"request_id": ctx.request_id, "employee_id": ctx.identity.employee_id, "days": days})
    return ToolResult("apply_leave", args, "success",
                      f"Your {leave_type} leave request {request_id} for {s} to {e} ({days} working day(s)) was created as pending.",
                      data={"leave_request_id": request_id, "working_days": days})


# --------------------------------------------------------------------------- cancel_leave
def _cancel_checks(ctx: ToolContext, request_id: Any) -> sqlite3.Row:
    """Only the requester's OWN pending/approved requests that start after today. Unknown id and someone
    else's id give the same message, so request IDs can't be probed."""
    not_found = RuleViolation("request_not_found", "I couldn't find one of your leave requests with that ID.")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 64:
        raise not_found
    row = ctx.conn.execute("SELECT * FROM leave_requests WHERE request_id = ? AND employee_id = ?",
                           (request_id, ctx.identity.employee_id)).fetchone()
    if row is None:
        raise not_found
    if row["status"] not in ("pending", "approved"):
        raise RuleViolation("not_cancellable", f"That request is already {row['status']}, so it can't be cancelled.")
    if row["start_date"] <= ctx.today.isoformat():
        raise RuleViolation("already_started",
                            f"That leave starts on {row['start_date']}, which is not after today ({ctx.today}); only future leave can be cancelled.")
    return row


def check_cancel_leave(ctx: ToolContext, request_id: Any) -> ToolResult:
    args = {"request_id": str(request_id)}
    try:
        row = _cancel_checks(ctx, request_id)
    except RuleViolation as v:
        return _err("cancel_leave", args, v.code, v.message)
    return ToolResult("cancel_leave", args, "pending_confirmation",
                      f"Cancel your {row['status']} {row['leave_type']} leave {row['start_date']} to {row['end_date']} "
                      f"({row['working_days']} working day(s) returned to your balance).",
                      data={"working_days": row["working_days"]})


def cancel_leave(ctx: ToolContext, request_id: Any) -> ToolResult:
    args = {"request_id": str(request_id)}
    conn = ctx.conn
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = _cancel_checks(ctx, request_id)
        # The status guard makes a repeated "yes" a no-op error instead of a second refund.
        cur = conn.execute("UPDATE leave_requests SET status = 'cancelled' WHERE request_id = ? AND status IN ('pending','approved')",
                           (request_id,))
        if cur.rowcount != 1:
            raise RuleViolation("not_cancellable", "That request was already cancelled.")
        cur = conn.execute("UPDATE leave_balances SET balance_days = balance_days + ? WHERE employee_id = ? AND leave_type = ?",
                           (row["working_days"], ctx.identity.employee_id, row["leave_type"]))
        if cur.rowcount == 0:
            conn.execute("INSERT INTO leave_balances(employee_id, leave_type, balance_days) VALUES (?,?,?)",
                         (ctx.identity.employee_id, row["leave_type"], row["working_days"]))
        _audit(conn, ctx, "cancel_leave", "success", f"{request_id} {row['working_days']}d")
        conn.execute("COMMIT")
    except RuleViolation as v:
        conn.execute("ROLLBACK")
        _audit(conn, ctx, "cancel_leave", f"rejected:{v.code}")
        return _err("cancel_leave", args, v.code, v.message)
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return ToolResult("cancel_leave", args, "success",
                      f"Cancelled your {row['leave_type']} leave {row['start_date']} to {row['end_date']}; "
                      f"{row['working_days']} working day(s) were returned to your balance.",
                      data={"working_days": row["working_days"]})


# Registry used by the orchestrator. Names and argument names are fixed by the spec.
TOOLS = {
    "get_leave_balance": get_leave_balance,
    "list_leave_requests": list_leave_requests,
    "get_team_leave": get_team_leave,
    "apply_leave": apply_leave,
    "cancel_leave": cancel_leave,
}
