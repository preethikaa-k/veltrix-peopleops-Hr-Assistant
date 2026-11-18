"""Automatic extraction of public holidays from calendar documents (spec 4.1, 13).

Nothing is hardcoded: a table in a calendar document is read as holidays if its rows contain dates.
Tables under a heading about *restricted/optional* holidays are NOT office closures and are excluded,
so they never reduce a requester's working-day count.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Optional

from app.ingestion.parsers import Block

log = logging.getLogger("veltrix.holidays")

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")
_MDY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_RESTRICTED = re.compile(r"restrict|optional|floating|elective", re.I)


def _month(name: str) -> Optional[int]:
    n = name.lower()
    for full, i in _MONTHS.items():
        if full == n or (len(n) >= 3 and full.startswith(n[:3]) and len(n) <= len(full)):
            return i
    return None


def parse_date(text: str) -> Optional[date]:
    m = _ISO.search(text)
    try:
        if m:
            return date(int(m[1]), int(m[2]), int(m[3]))
        m = _DMY.search(text)
        if m and _month(m[2]):
            return date(int(m[3]), _month(m[2]), int(m[1]))
        m = _MDY.search(text)
        if m and _month(m[1]):
            return date(int(m[3]), _month(m[1]), int(m[2]))
    except ValueError:
        return None
    return None


def extract_holidays(blocks: list[Block]) -> list[tuple[date, str]]:
    """Walk blocks in order, tracking the nearest heading so each table is classified by its section."""
    out: dict[date, str] = {}
    heading = ""
    for b in blocks:
        if b.kind == "heading":
            heading = b.text
            continue
        if b.kind != "table" or len(b.rows) < 2:
            continue
        if _RESTRICTED.search(heading) or _RESTRICTED.search(" ".join(b.rows[0])):
            continue
        parsed = [(parse_date(" ".join(r)), r) for r in b.rows[1:]]
        if not parsed or sum(1 for d, _ in parsed if d) < max(1, len(parsed) // 2):
            continue                                        # not a date table
        for d, r in parsed:
            if d is None:
                log.warning("unparseable holiday row skipped", extra={"row_cells": len(r)})
                continue
            cells = [c for c in r if c and parse_date(c) is None and c.lower() not in _WEEKDAYS]
            name = cells[-1] if cells else "Public holiday"
            wd = next((c.lower() for c in r if c.lower() in _WEEKDAYS), None)
            if wd and _WEEKDAYS[d.weekday()] != wd:
                log.warning("holiday weekday mismatch", extra={"date": d.isoformat(), "stated": wd})
            out.setdefault(d, name)
    return sorted(out.items())
