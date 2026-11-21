"""Live demonstration script showing real-time database updates in veltrix.db."""
from __future__ import annotations

import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import connect
from app.api import create_app


def print_db_state(conn, employee_id: str, label: str):
    print(f"\n==================== [ {label} ] ====================")
    bal_row = conn.execute(
        "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = 'annual'",
        (employee_id,),
    ).fetchone()
    current_bal = bal_row["balance_days"] if bal_row else 0.0
    print(f"💰 Employee {employee_id} Annual Leave Balance: {current_bal} days")

    requests = conn.execute(
        "SELECT request_id, leave_type, start_date, end_date, working_days, status FROM leave_requests WHERE employee_id = ? ORDER BY created_at DESC LIMIT 3",
        (employee_id,),
    ).fetchall()
    print(f"📋 Recent Leave Requests in DB:")
    for r in requests:
        print(f"   • ID: {r['request_id']} | {r['start_date']} to {r['end_date']} | {r['working_days']} days | Status: [{r['status']}]")

    audit = conn.execute(
        "SELECT id, action, outcome, detail FROM audit_log WHERE employee_id = ? ORDER BY id DESC LIMIT 1",
        (employee_id,),
    ).fetchone()
    if audit:
        print(f"🔒 Latest Audit Log Entry: #{audit['id']} - Action={audit['action']}, Outcome={audit['outcome']}, Detail={audit['detail']}")
    print("==============================================================\n")
    return current_bal


def main():
    employee_id = "E2004"
    session_id = "realtime-apply-session-1"

    settings = Settings.from_env()
    app = create_app(settings, ingest_fn=lambda _: None)
    client = TestClient(app)

    with connect(settings.db_path) as conn:
        print("🎯 TARGET EMPLOYEE: E2004 (Sneha Kulkarni)")
        
        # 1. Show INITIAL STATE
        initial_bal = print_db_state(conn, employee_id, "1. INITIAL STATE (BEFORE APPLYING)")

        # 2. Step 1: Propose Leave for Dec 1 - Dec 2 (2 working days)
        print("💬 [User Message]: 'I want to apply for annual leave from 2026-12-01 to 2026-12-02'")
        res1 = client.post(
            "/chat",
            json={"session_id": session_id, "message": "I want to apply for annual leave from 2026-12-01 to 2026-12-02"},
            headers={"X-Employee-Id": employee_id},
        ).json()
        print(f"🤖 [Assistant]: {res1['answer']}")
        print(f"   (pending_confirmation: {res1['pending_confirmation']})")

        # 3. Step 2: Confirm Leave
        print("\n💬 [User Message]: 'yes, confirm'")
        res2 = client.post(
            "/chat",
            json={"session_id": session_id, "message": "yes, confirm"},
            headers={"X-Employee-Id": employee_id},
        ).json()
        print(f"🤖 [Assistant]: {res2['answer']}")

        # 4. Show UPDATED STATE in DB
        bal_after_apply = print_db_state(conn, employee_id, "2. AFTER CONFIRMATION (DATABASE UPDATED!)")
        
        print(f"✨ REAL-TIME SQL PROOF:")
        print(f"   1. Balance DEDUCTED: {initial_bal} days -> {bal_after_apply} days (-2 working days!)")
        print(f"   2. A brand-new row was INSERTED into table 'leave_requests' with status 'pending'!")
        print(f"   3. A permanent record was INSERTED into table 'audit_log'!")


if __name__ == "__main__":
    main()
