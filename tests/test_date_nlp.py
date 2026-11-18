from datetime import date
import pytest
from app.dates import parse_date_phrase, extract_reference_date

TODAY = date(2026, 9, 15)  # Tuesday


def test_next_friday():
    s, e, amb, past = parse_date_phrase("I want to take off next Friday", TODAY)
    assert s == date(2026, 9, 18)
    assert e == date(2026, 9, 18)
    assert not amb and not past


def test_next_friday_when_today_is_friday():
    friday_today = date(2026, 9, 18)
    s, e, amb, past = parse_date_phrase("leave on next Friday", friday_today)
    assert s == date(2026, 9, 25)
    assert e == date(2026, 9, 25)


def test_next_week():
    s, e, amb, past = parse_date_phrase("book annual leave for next week", TODAY)
    assert s == date(2026, 9, 21)
    assert e == date(2026, 9, 25)
    assert not amb and not past


def test_tomorrow():
    s, e, amb, past = parse_date_phrase("sick leave tomorrow", TODAY)
    assert s == date(2026, 9, 16)
    assert e == date(2026, 9, 16)


def test_explicit_dates_range():
    s, e, amb, past = parse_date_phrase("from 2026-10-02 to 2026-10-06", TODAY)
    assert s == date(2026, 10, 2)
    assert e == date(2026, 10, 6)
    assert not amb and not past


def test_month_name_range():
    s, e, amb, past = parse_date_phrase("apply leave October 2 to October 6", TODAY)
    assert s == date(2026, 10, 2)
    assert e == date(2026, 10, 6)
    assert not amb and not past


def test_ambiguous_phrase():
    s, e, amb, past = parse_date_phrase("I want to take a couple of days off sometime next month", TODAY)
    assert amb is True


def test_past_date():
    s, e, amb, past = parse_date_phrase("apply leave from 2026-09-01 to 2026-09-05", TODAY)
    assert past is True


def test_extract_reference_date_past_year():
    ref = extract_reference_date("What was the sick leave allowance in 2025?", TODAY)
    assert ref == date(2025, 12, 31)


def test_extract_reference_date_default_today():
    ref = extract_reference_date("What is the sick leave policy?", TODAY)
    assert ref == TODAY
