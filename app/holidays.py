"""Public holidays for a requester, from calendar documents extracted at ingestion (never hardcoded).

The calendar is chosen with the SAME eligibility + precedence engine as every other answer, per year,
so a modified corpus (e.g. a corrected calendar with a later effective date) changes the result.
"""
from __future__ import annotations

import sqlite3
from datetime import date

from app.db import Identity
from app.policy import Claim, DocMeta, is_eligible, resolve


class HolidayError(Exception):
    """Holiday data is unavailable or contradictory; callers must fail closed with this message."""


def holidays_for_range(conn: sqlite3.Connection, ident: Identity, ref: date, start: date, end: date) -> dict[date, str]:
    docs = {
        r["doc_id"]: DocMeta.from_row(r)
        for r in conn.execute("SELECT * FROM documents WHERE doc_type = 'calendar' AND status = 'ok'")
    }
    eligible = {i: d for i, d in docs.items() if is_eligible(d, ident.role, ident.region, ref)}
    by_doc: dict[str, dict[date, str]] = {}
    for r in conn.execute("SELECT doc_id, holiday_date, name FROM holidays"):
        if r["doc_id"] in eligible:
            by_doc.setdefault(r["doc_id"], {})[date.fromisoformat(r["holiday_date"])] = r["name"]

    result: dict[date, str] = {}
    for year in range(start.year, end.year + 1):
        claims = []
        for doc_id, rows in by_doc.items():
            in_year = {d: n for d, n in rows.items() if d.year == year}
            if in_year:  # a calendar "covers" a year if it lists holidays in it
                claims.append(Claim(eligible[doc_id], in_year))
        if not claims:
            raise HolidayError(f"No holiday calendar applies to the {ident.region} office for {year}.")
        res = resolve(claims)
        if res.conflict:
            raise HolidayError(f"Holiday calendars for {year} conflict and cannot be resolved; please contact the HR team.")
        result.update(res.winners[0].value)
    return {d: n for d, n in result.items() if start <= d <= end}
