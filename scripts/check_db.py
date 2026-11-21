"""Database connection and health verification script."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import connect


def main() -> None:
    settings = Settings.from_env()
    print("====================================================")
    print("           VELTRIX DATABASE HEALTH CHECK            ")
    print("====================================================")
    print(f"📁 Target Database Path: {settings.db_path}")

    if not settings.db_path.is_file():
        print(f"❌ Error: Database file not found at {settings.db_path}!")
        print("   Run: python scripts/setup_data.py <zip_path>")
        sys.exit(1)

    try:
        with connect(settings.db_path) as conn:
            emp_count = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
            bal_count = conn.execute("SELECT COUNT(*) FROM leave_balances").fetchone()[0]
            req_count = conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0]
            
            # Check helper tables if initialized
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            hol_count = conn.execute("SELECT COUNT(*) FROM holidays").fetchone()[0] if "holidays" in tables else 0
            chk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] if "chunks" in tables else 0

            print("✅ Status: Connection Successful!")
            print("----------------------------------------------------")
            print("📊 Database Tables Summary:")
            print(f"  • employees:        {emp_count} employees")
            print(f"  • leave_balances:   {bal_count} balance records")
            print(f"  • leave_requests:   {req_count} requests")
            if "holidays" in tables:
                print(f"  • holidays:         {hol_count} public holidays extracted")
            if "chunks" in tables:
                print(f"  • chunks:           {chk_count} policy chunks indexed")
            print("----------------------------------------------------")
            print("👤 Verified Sample Accounts:")
            for r in conn.execute("SELECT employee_id, full_name, role, region FROM employees LIMIT 3"):
                print(f"  • {r['employee_id']}: {r['full_name']} ({r['role']} | {r['region']})")
            print("====================================================")
            print("🎉 Database is 100% connected, healthy, and ready!")
            print("====================================================")
    except Exception as exc:
        print(f"❌ Database connection failed with error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
