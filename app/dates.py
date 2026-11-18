"""Calendar-date helpers. Dates are plain `date` objects: no times, no time zones (spec 8.4)."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Iterator, Mapping, Optional

WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def parse_iso(value: Any) -> date:
    """Strict YYYY-MM-DD. Anything else (wrong type, wrong shape, impossible date) raises ValueError."""
    if not isinstance(value, str) or len(value) != 10 or value[4] != "-" or value[7] != "-":
        raise ValueError(f"not an ISO date: {value!r}")
    return date.fromisoformat(value)


def iter_days(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def is_weekday(d: date) -> bool:
    return d.weekday() < 5  # Monday=0 .. Friday=4


def working_days(start: date, end: date, holidays: Mapping[date, str] | set[date]) -> int:
    """Dates in [start, end] (both inclusive) that fall Monday-Friday and are not public holidays."""
    return sum(1 for d in iter_days(start, end) if is_weekday(d) and d not in holidays)


def _next_weekday(today: date, target_wd: int) -> date:
    """The first occurrence of target_wd strictly after today (spec 8.4)."""
    days = (target_wd - today.weekday()) % 7
    if days == 0:
        days = 7
    return today + timedelta(days=days)


def _this_weekday(today: date, target_wd: int) -> date:
    """The target_wd in the current week (or coming up)."""
    days = target_wd - today.weekday()
    if days < 0:
        days += 7
    return today + timedelta(days=days)


def _parse_explicit_single_date(text: str, default_year: int = 2026) -> Optional[date]:
    """Parse phrases like 2026-10-02, October 2, 2nd October, Oct 2 2026."""
    text = text.strip().lower()
    m_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m_iso:
        try:
            return date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
        except ValueError:
            return None

    # Month day e.g. "October 2" or "Oct 2nd, 2026" or "October 2 2026"
    month_pat = r"|".join(MONTHS.keys())
    m = re.search(rf"\b({month_pat})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{{4}}))?\b", text)
    if m:
        mo = MONTHS[m.group(1)]
        day = int(m.group(2))
        yr = int(m.group(3)) if m.group(3) else default_year
        try:
            return date(yr, mo, day)
        except ValueError:
            return None

    # Day month e.g. "2nd October", "2 Oct 2026"
    m2 = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pat})(?:\s*,?\s*(\d{{4}}))?\b", text)
    if m2:
        day = int(m2.group(1))
        mo = MONTHS[m2.group(2)]
        yr = int(m2.group(3)) if m2.group(3) else default_year
        try:
            return date(yr, mo, day)
        except ValueError:
            return None

    return None


def extract_reference_date(text: str, today: date) -> date:
    """Extract reference date for past queries (e.g. 'in 2025' or 'as of 2025-08-01') (spec 5)."""
    text_lower = text.lower()
    # Check for specific ISO date in past query
    m_iso = re.search(r"\b(20\d\d)-(\d{2})-(\d{2})\b", text_lower)
    if m_iso:
        try:
            d = date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
            if d < today:
                return d
        except ValueError:
            pass

    # Check for year in past e.g. "in 2025", "for 2025", "during 2024"
    m_yr = re.search(r"\b(?:in|during|for|back in|as of)\s+(20\d\d)\b", text_lower)
    if m_yr:
        yr = int(m_yr.group(1))
        if yr < today.year:
            return date(yr, 12, 31)
        if yr > today.year:
            return date(yr, 1, 1)

    return today


def parse_date_phrase(text: str, today: date) -> tuple[Optional[date], Optional[date], bool, bool]:
    """Parse relative and natural language date phrases (spec 8.4).
    Returns (start_date, end_date, is_ambiguous, is_past).
    """
    text_lower = text.lower().strip()

    # Ambiguous phrases check
    ambiguous_patterns = [
        r"\bsometime\b",
        r"\bcouple of days\b",
        r"\ba few days\b",
        r"\blater this month\b",
        r"\bnext month\b",
        r"\bsome days\b",
    ]
    for pat in ambiguous_patterns:
        if re.search(pat, text_lower):
            return None, None, True, False

    # 1. "next week": Monday to Friday of the following week (spec 8.4)
    if "next week" in text_lower:
        cur_monday = today - timedelta(days=today.weekday())
        nxt_monday = cur_monday + timedelta(days=7)
        nxt_friday = nxt_monday + timedelta(days=4)
        return nxt_monday, nxt_friday, False, False

    # 2. "this week"
    if "this week" in text_lower:
        cur_monday = today - timedelta(days=today.weekday())
        cur_friday = cur_monday + timedelta(days=4)
        is_past = cur_monday < today
        return cur_monday, cur_friday, False, is_past

    # 3. "tomorrow"
    if "tomorrow" in text_lower:
        tom = today + timedelta(days=1)
        return tom, tom, False, False

    # 4. "today"
    if re.search(r"\btoday\b", text_lower):
        return today, today, False, False

    # 5. "yesterday"
    if "yesterday" in text_lower:
        yest = today - timedelta(days=1)
        return yest, yest, False, True

    # 6. "next <weekday>" range or single, e.g. "next Friday", "next monday to next wednesday"
    wd_pat = r"|".join(WEEKDAYS.keys())
    m_next_wd_range = re.search(rf"\bnext\s+({wd_pat})\s+(?:to|until|through|-)\s+next\s+({wd_pat})\b", text_lower)
    if m_next_wd_range:
        s = _next_weekday(today, WEEKDAYS[m_next_wd_range.group(1)])
        e = _next_weekday(today, WEEKDAYS[m_next_wd_range.group(2)])
        if e < s:
            e += timedelta(days=7)
        return s, e, False, False

    # 7. Explicit range between two dates: "from <d1> to <d2>" or "<d1> to <d2>" or "<d1> - <d2>"
    # Split on range delimiters
    range_match = re.search(r"(?:from\s+)?(.+?)\s+(?:to|until|through|--?)\s+(.+)", text_lower)
    if range_match:
        part1 = range_match.group(1).strip()
        part2 = range_match.group(2).strip()

        # Handle "next Friday" in part1/part2
        d1 = None
        d2 = None
        m_wd1 = re.search(rf"\bnext\s+({wd_pat})\b", part1)
        if m_wd1:
            d1 = _next_weekday(today, WEEKDAYS[m_wd1.group(1)])
        else:
            d1 = _parse_explicit_single_date(part1, default_year=2026)

        m_wd2 = re.search(rf"\bnext\s+({wd_pat})\b", part2)
        if m_wd2:
            d2 = _next_weekday(today, WEEKDAYS[m_wd2.group(1)])
        else:
            d2 = _parse_explicit_single_date(part2, default_year=2026)

        if d1 and d2:
            is_past = d1 < today or d2 < today
            return d1, d2, False, is_past

    # 8. Single "next <weekday>" e.g. "next Friday"
    m_single_next_wd = re.search(rf"\bnext\s+({wd_pat})\b", text_lower)
    if m_single_next_wd:
        d = _next_weekday(today, WEEKDAYS[m_single_next_wd.group(1)])
        return d, d, False, False

    # 9. Single "this <weekday>"
    m_this_wd = re.search(rf"\bthis\s+({wd_pat})\b", text_lower)
    if m_this_wd:
        d = _this_weekday(today, WEEKDAYS[m_this_wd.group(1)])
        return d, d, False, d < today

    # 10. Single explicit date: e.g. "2026-10-02" or "October 2"
    d_single = _parse_explicit_single_date(text_lower, default_year=2026)
    if d_single:
        is_past = d_single < today
        return d_single, d_single, False, is_past

    # Could not parse any date
    return None, None, False, False
