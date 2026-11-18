"""Section 5 of the spec as pure, deterministic code: eligibility and precedence.

Nothing here touches an LLM or a database. The LLM may *tag* which rule a passage states, but WHICH
document wins is decided only by these functions, so it is testable and auditable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable, Sequence

ROLE_ACCESS: dict[str, frozenset[str]] = {
    "employee": frozenset({"all"}),
    "manager": frozenset({"all", "managers"}),
    "hr": frozenset({"all", "managers", "hr"}),
}
AUTHORITATIVE_TYPES = frozenset({"policy", "addendum", "memo", "handbook", "calendar"})  # faq, wiki rank lower


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str

    @classmethod
    def from_row(cls, row: Any) -> "DocMeta":
        regions = row["regions"]
        if isinstance(regions, str):
            regions = json.loads(regions)
        return cls(row["doc_id"], row["title"], row["doc_type"], tuple(regions),
                   date.fromisoformat(row["effective_date"]), row["access_level"])


def can_read(role: str, access_level: str) -> bool:
    return access_level in ROLE_ACCESS.get(role, frozenset())  # unknown role -> nothing


def region_matches(doc: DocMeta, region: str) -> bool:
    return "GLOBAL" in doc.regions or region in doc.regions


def is_eligible(doc: DocMeta, role: str, region: str, ref: date) -> bool:
    """All three must hold: role can read it, region matches (or GLOBAL), already effective."""
    return can_read(role, doc.access_level) and region_matches(doc, region) and doc.effective_date <= ref


def authority_rank(doc: DocMeta) -> int:
    return 1 if doc.doc_type in AUTHORITATIVE_TYPES else 0


def scope_rank(doc: DocMeta) -> int:
    return 0 if "GLOBAL" in doc.regions else 1


@dataclass(frozen=True)
class Claim:
    """One document stating a value for one rule."""
    doc: DocMeta
    value: Any


def _key(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)


@dataclass
class Resolution:
    winners: list[Claim] = field(default_factory=list)
    conflict: bool = False          # winners disagree on the value -> report all, pick none
    decided_by: str = "none"        # which test settled it: authority | scope | recency | tie | single | none
    trace: list[str] = field(default_factory=list)   # audit trail: who was eliminated, by which test

    @property
    def value(self) -> Any:
        return None if self.conflict or not self.winners else self.winners[0].value


def resolve(claims: Sequence[Claim]) -> Resolution:
    """Apply authority -> scope -> recency in order; move on only if the test did not decide.
    Documents that tie on all three but state the SAME value are not a conflict."""
    res = Resolution()
    pool = list(claims)
    if not pool:
        return res
    if len(pool) == 1:
        res.winners, res.decided_by = pool, "single"
        return res
    tests: list[tuple[str, Callable[[Claim], Any]]] = [
        ("authority", lambda c: authority_rank(c.doc)),
        ("scope", lambda c: scope_rank(c.doc)),
        ("recency", lambda c: c.doc.effective_date),
    ]
    for name, key in tests:
        best = max(key(c) for c in pool)
        keep = [c for c in pool if key(c) == best]
        for c in pool:
            if c not in keep:
                res.trace.append(f"{c.doc.doc_id} eliminated by {name}")
        if len(keep) < len(pool) and len(keep) == 1:
            res.winners, res.decided_by = keep, name
            return res
        pool = keep
    res.winners = pool
    if len({_key(c.value) for c in pool}) > 1:
        res.conflict, res.decided_by = True, "tie"
        res.trace.append("unresolved conflict: " + ", ".join(c.doc.doc_id for c in pool))
    else:
        res.decided_by = "tie-same-value"
    return res


def upcoming_changes(claims: Iterable[Claim], ref: date) -> list[Claim]:
    """Claims from documents NOT yet effective (readable + region-matching already assumed by the
    caller) that would win on their own effective date AND change the answer (spec 5.1).
    The caller passes every claim the requester may read for their region, any date."""
    all_claims = list(claims)
    current = resolve([c for c in all_claims if c.doc.effective_date <= ref])
    out: list[Claim] = []
    for fut in sorted((c for c in all_claims if c.doc.effective_date > ref), key=lambda c: c.doc.effective_date):
        at = resolve([c for c in all_claims if c.doc.effective_date <= fut.doc.effective_date])
        wins = any(w.doc.doc_id == fut.doc.doc_id for w in at.winners) and not at.conflict
        changes = current.conflict or not current.winners or _key(fut.value) != _key(current.value)
        if wins and changes:
            out.append(fut)
    return out
