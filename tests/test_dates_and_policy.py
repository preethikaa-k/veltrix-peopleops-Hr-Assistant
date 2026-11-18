from datetime import date

import pytest

from app.dates import parse_iso, working_days
from app.policy import Claim, DocMeta, is_eligible, resolve, upcoming_changes


# ---------------- working days (spec 6.2) ----------------
def D(s): return date.fromisoformat(s)


def test_working_days_skips_weekends():
    assert working_days(D("2026-09-14"), D("2026-09-20"), set()) == 5          # Mon-Sun
    assert working_days(D("2026-09-19"), D("2026-09-20"), set()) == 0          # Sat-Sun only
    assert working_days(D("2026-09-18"), D("2026-09-21"), set()) == 2          # Fri..Mon spans a weekend


def test_working_days_skips_holidays_and_both_ends_inclusive():
    hol = {D("2026-10-02"): "x"}
    assert working_days(D("2026-09-28"), D("2026-10-02"), hol) == 4
    assert working_days(D("2026-10-02"), D("2026-10-02"), hol) == 0            # single holiday
    assert working_days(D("2026-10-02"), D("2026-10-06"), hol) == 2            # holiday + weekend + Mon,Tue
    assert working_days(D("2026-09-15"), D("2026-09-15"), set()) == 1          # one-day range


def test_holiday_on_weekend_is_not_double_subtracted():
    assert working_days(D("2026-08-14"), D("2026-08-17"), {D("2026-08-15"): "Sat holiday"}) == 2


@pytest.mark.parametrize("bad", ["2026-13-01", "2026-02-30", "15/09/2026", "2026-9-1", 20260915, None, ""])
def test_parse_iso_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_iso(bad)


# ---------------- precedence (spec 5) ----------------
def doc(i, typ, regions, eff, access="all"):
    return DocMeta(i, i, typ, tuple(regions), D(eff), access)


V1 = doc("V1", "policy", ["GLOBAL"], "2025-01-01")
V2 = doc("V2", "policy", ["GLOBAL"], "2026-04-01")
FAQ = doc("FAQ", "faq", ["GLOBAL"], "2026-05-01")
DE = doc("DE", "addendum", ["DE"], "2025-06-01")
V3 = doc("V3", "policy", ["GLOBAL"], "2027-01-01")
SPEC = {"V1": 2, "V2": 3, "FAQ": 2, "DE": 5, "V3": 4}      # the worked example in spec 5.2


def claims_for(region, ref, role="employee"):
    docs = [d for d in (V1, V2, FAQ, DE, V3) if is_eligible(d, role, region, ref)]
    return [Claim(d, SPEC[d.doc_id]) for d in docs]


def all_claims(region, role="employee"):
    docs = [d for d in (V1, V2, FAQ, DE, V3) if is_eligible(d, role, region, date(2099, 1, 1))]
    return [Claim(d, SPEC[d.doc_id]) for d in docs]


def test_worked_example_india_today():
    today = D("2026-09-15")
    r = resolve(claims_for("IN", today))
    assert r.value == 3 and r.winners[0].doc.doc_id == "V2" and r.decided_by == "recency"
    assert "FAQ eliminated by authority" in r.trace
    up = upcoming_changes(all_claims("IN"), today)
    assert [(c.doc.doc_id, c.value) for c in up] == [("V3", 4)]


def test_worked_example_germany_today_no_upcoming():
    today = D("2026-09-15")
    r = resolve(claims_for("DE", today))
    assert r.value == 5 and r.decided_by == "scope"
    assert upcoming_changes(all_claims("DE"), today) == []     # addendum would still win in 2027


def test_worked_example_india_asking_about_past_date():
    r = resolve(claims_for("IN", D("2025-08-01")))
    assert r.value == 2 and [c.doc.doc_id for c in r.winners] == ["V1"]


def test_authority_beats_recency():
    r = resolve([Claim(V2, 12), Claim(FAQ, 10)])               # FAQ is newer but loses on authority
    assert r.value == 12 and r.decided_by == "authority"


def test_scope_beats_recency():
    r = resolve([Claim(doc("G", "policy", ["GLOBAL"], "2026-06-01"), 12), Claim(doc("R", "addendum", ["DE"], "2025-01-01"), 30)])
    assert r.value == 30 and r.decided_by == "scope"


def test_unresolved_conflict_reports_both():
    a, b = doc("A", "policy", ["GLOBAL"], "2026-01-01"), doc("B", "policy", ["GLOBAL"], "2026-01-01")
    r = resolve([Claim(a, 1), Claim(b, 2)])
    assert r.conflict and {c.doc.doc_id for c in r.winners} == {"A", "B"} and r.value is None


def test_full_tie_with_same_value_is_not_a_conflict():
    a, b = doc("A", "policy", ["GLOBAL"], "2026-04-01"), doc("B", "policy", ["GLOBAL"], "2026-04-01")
    r = resolve([Claim(a, 12), Claim(b, 12)])                  # e.g. policy v2 and its summary page
    assert not r.conflict and r.value == 12


def test_multi_region_memo_outranks_global_policy():
    memo = doc("M", "memo", ["IN", "US"], "2026-07-01")
    r = resolve([Claim(doc("P", "policy", ["GLOBAL"], "2025-01-01"), 2), Claim(memo, 3)])
    assert r.value == 3 and r.decided_by == "scope"


def test_upcoming_not_reported_when_value_unchanged():
    nxt = doc("N", "policy", ["GLOBAL"], "2027-01-01")
    assert upcoming_changes([Claim(V2, 3), Claim(nxt, 3)], D("2026-09-15")) == []


@pytest.mark.parametrize("role,level,ok", [("employee", "all", True), ("employee", "managers", False), ("employee", "hr", False),
                                          ("manager", "managers", True), ("manager", "hr", False), ("hr", "hr", True), ("hr", "managers", True),
                                          ("intern", "all", False)])
def test_access_levels(role, level, ok):
    assert is_eligible(doc("X", "policy", ["GLOBAL"], "2025-01-01", level), role, "IN", D("2026-09-15")) is ok


def test_eligibility_requires_region_and_effective_date():
    assert not is_eligible(doc("X", "policy", ["DE"], "2025-01-01"), "hr", "IN", D("2026-09-15"))
    assert not is_eligible(doc("X", "policy", ["GLOBAL"], "2026-09-16"), "hr", "IN", D("2026-09-15"))
    assert is_eligible(doc("X", "policy", ["GLOBAL"], "2026-09-15"), "hr", "IN", D("2026-09-15"))   # effective day counts
