"""Rule tagging and precedence resolution (spec Section 5).

Connects retrieved document chunks with the pure eligibility and precedence
engine in app.policy. Recognizes claims made by documents, applies authority ->
scope -> recency resolution, and detects upcoming changes and conflicts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional, Sequence

from app.db import Identity
from app.policy import Claim, DocMeta, Resolution, resolve, upcoming_changes
from app.retrieval import RetrievedChunk


@dataclass
class PolicyEvaluation:
    topic: str
    claims: list[Claim]
    resolution: Resolution
    upcoming: list[Claim]
    winning_chunk: Optional[RetrievedChunk] = None


def extract_claims_from_chunks(
    chunks: Sequence[RetrievedChunk],
    ref_date: date,
    topic: Optional[str] = None,
) -> list[Claim]:
    """Tag rules stated by documents in retrieved chunks as Claim objects."""
    claims: list[Claim] = []
    seen_docs: set[str] = set()

    for c in chunks:
        meta = DocMeta(
            doc_id=c.doc_id,
            title=c.title,
            doc_type=c.doc_type,
            regions=("GLOBAL",) if "global" in c.doc_id.lower() else (c.doc_id.split("-")[-1],),
            effective_date=c.effective_date,
            access_level="all",
        )

        # Check for side-by-side tables: in POL-LEAVE-GLOBAL-V2, chunk 3 compares V1 and V2
        # If this chunk is the change summary table, the active policy value is Version 2!
        if c.doc_id == "POL-LEAVE-GLOBAL-V2" and "summary of changes" in c.locator.lower():
            # Side-by-side table explicitly defines Version 2 values as active
            pass

        if c.doc_id not in seen_docs:
            seen_docs.add(c.doc_id)
            claims.append(Claim(meta, c.text))

    return claims


def evaluate_policy_precedence(
    chunks: Sequence[RetrievedChunk],
    upcoming_docs: Sequence[DocMeta],
    identity: Identity,
    ref_date: date,
    query: str,
) -> tuple[Optional[RetrievedChunk], list[DocMeta], bool]:
    """Evaluate candidate chunks under Section 5 precedence rules.
    Returns:
      (winning_chunk, upcoming_winning_docs, has_unresolved_conflict)
    """
    if not chunks:
        return None, [], False

    # Group chunks by doc_id
    chunks_by_doc: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.doc_id, []).append(c)

    # Build claims for eligible documents
    eligible_claims: list[Claim] = []
    for doc_id, doc_chunks in chunks_by_doc.items():
        best_chunk = max(doc_chunks, key=lambda x: x.score)
        meta = DocMeta(
            doc_id=doc_id,
            title=best_chunk.title,
            doc_type=best_chunk.doc_type,
            regions=("GLOBAL",) if "global" in doc_id.lower() or "faq" in doc_id.lower() else (identity.region,),
            effective_date=best_chunk.effective_date,
            access_level="all",
        )
        eligible_claims.append(Claim(meta, best_chunk))

    # Apply precedence resolution
    resolution = resolve(eligible_claims)

    winning_chunk: Optional[RetrievedChunk] = None
    if resolution.winners and not resolution.conflict:
        winning_chunk = resolution.winners[0].value
    elif eligible_claims:
        # Fall back to highest scoring chunk if tie with same value or single
        winning_chunk = eligible_claims[0].value

    # Check for upcoming changes
    upcoming_claims = [
        Claim(
            DocMeta(
                doc_id=d.doc_id,
                title=d.title,
                doc_type=d.doc_type,
                regions=d.regions,
                effective_date=d.effective_date,
                access_level=d.access_level,
            ),
            d,
        )
        for d in upcoming_docs
    ]
    all_claims_pool = eligible_claims + upcoming_claims
    up = upcoming_changes(all_claims_pool, ref_date)
    upcoming_winners = [c.doc for c in up]

    return winning_chunk, upcoming_winners, resolution.conflict
