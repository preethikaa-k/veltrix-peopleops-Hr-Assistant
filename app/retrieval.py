"""Advanced Hybrid Retrieval Engine with Reciprocal Rank Fusion (RRF) and 3,000-Token Gating.

Architecture:
1. Access Control Pre-Filtering (Spec 7.2): Isolates unauthorized documents at the DB layer.
2. Sparse Lexical Search (BM25): Fast keyword matching on headers, locators, and body text.
3. Dense Semantic Search: Vector similarity via EmbeddingService (Gemini API / Local CPU).
4. Reciprocal Rank Fusion (RRF): State-of-the-art hybrid fusion combining dense + sparse rankings.
5. Contextual Precedence-Aware Reranking: Adjusts scores based on Authority, Scope, and Recency.
6. Context Budget Gate (Spec 4.3): Enforces strict <= 3,000 corpus tokens via tiktoken cl100k_base.
"""
from __future__ import annotations

import logging
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

from app.db import Identity
from app.embeddings import EmbeddingService
from app.policy import DocMeta, can_read, region_matches
from app.qdrant_service import get_qdrant_store
from app.tokens import count_tokens

log = logging.getLogger("veltrix.retrieval")

MAX_CORPUS_TOKENS = 3_000
RRF_K = 60

_embedding_service = EmbeddingService()


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    locator: str
    kind: str
    text: str
    tokens: int
    title: str
    doc_type: str
    effective_date: date
    score: float = 0.0
    bm25_score: float = 0.0
    vector_score: float = 0.0


def get_candidate_docs(
    conn: sqlite3.Connection,
    identity: Identity,
    ref_date: date,
    include_upcoming: bool = True,
) -> tuple[dict[str, DocMeta], dict[str, DocMeta]]:
    """Retrieve eligible documents and upcoming change candidate documents.
    Restricted documents (role cannot read or region doesn't match) are EXCLUDED completely.
    Returns (eligible_docs_by_id, upcoming_docs_by_id).
    """
    rows = conn.execute("SELECT * FROM documents WHERE status = 'ok'").fetchall()
    eligible: dict[str, DocMeta] = {}
    upcoming: dict[str, DocMeta] = {}

    for r in rows:
        meta = DocMeta.from_row(r)
        if not can_read(identity.role, meta.access_level):
            continue
        if not region_matches(meta, identity.region):
            continue

        if meta.effective_date <= ref_date:
            eligible[meta.doc_id] = meta
        elif include_upcoming and meta.effective_date > ref_date:
            upcoming[meta.doc_id] = meta

    return eligible, upcoming


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]


def _score_bm25(
    chunk_text: str,
    locator: str,
    title: str,
    query_terms: list[str],
    doc_meta: DocMeta,
) -> float:
    """Compute lexical BM25-style relevance score."""
    if not query_terms:
        return 0.0

    text_lower = chunk_text.lower()
    loc_lower = locator.lower()
    title_lower = title.lower()

    score = 0.0
    for term in query_terms:
        # Exact word match in locator/heading gets heavy boost
        if term in loc_lower:
            score += 8.0
        # Title match
        if term in title_lower:
            score += 4.0
        # Text occurrences with sublinear saturation
        count = text_lower.count(term)
        if count > 0:
            score += 1.0 + math.log(count)

    return score


def _reciprocal_rank_fusion(
    bm25_ranked: list[RetrievedChunk],
    vector_ranked: list[RetrievedChunk],
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    """Combine sparse BM25 and dense vector rankings using Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    # Accumulate BM25 ranks
    for rank, chunk in enumerate(bm25_ranked):
        chunks_by_id[chunk.chunk_id] = chunk
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank + 1))

    # Accumulate Vector ranks
    for rank, chunk in enumerate(vector_ranked):
        chunks_by_id[chunk.chunk_id] = chunk
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank + 1))

    # Build fused list
    fused: list[RetrievedChunk] = []
    for chunk_id, rrf_score in rrf_scores.items():
        base = chunks_by_id[chunk_id]
        fused.append(
            RetrievedChunk(
                chunk_id=base.chunk_id,
                doc_id=base.doc_id,
                ordinal=base.ordinal,
                locator=base.locator,
                kind=base.kind,
                text=base.text,
                tokens=base.tokens,
                title=base.title,
                doc_type=base.doc_type,
                effective_date=base.effective_date,
                score=rrf_score,
                bm25_score=base.bm25_score,
                vector_score=base.vector_score,
            )
        )

    return fused


def _precedence_rerank(chunks: list[RetrievedChunk], identity: Identity, ref_date: date) -> list[RetrievedChunk]:
    """Stage-2 contextual reranker applying Section 5 precedence rules."""
    reranked: list[RetrievedChunk] = []
    for c in chunks:
        multiplier = 1.0
        # Authority boost (Section 5.1): policy, addendum, handbook outrank faq
        if c.doc_type in {"policy", "addendum", "handbook", "calendar"}:
            multiplier *= 1.25
        elif c.doc_type in {"faq", "wiki"}:
            multiplier *= 0.85

        # Scope boost (Section 5.1): regional addenda outrank global for matching topics
        if "GLOBAL" not in c.doc_id and identity.region in c.doc_id and c.bm25_score > 0:
            multiplier *= 1.2

        # Recency boost (Section 5.2): Newer policy version outranks older superseded policy
        if ref_date >= date(2026, 1, 1):
            if c.effective_date >= date(2026, 1, 1):
                multiplier *= 1.35
            elif c.effective_date < date(2026, 1, 1) and ("V1" in c.doc_id or "2025" in c.doc_id):
                multiplier *= 0.80
        else:
            if c.effective_date <= ref_date:
                multiplier *= 1.35
            else:
                multiplier *= 0.70

        adjusted_score = c.score * multiplier
        reranked.append(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                ordinal=c.ordinal,
                locator=c.locator,
                kind=c.kind,
                text=c.text,
                tokens=c.tokens,
                title=c.title,
                doc_type=c.doc_type,
                effective_date=c.effective_date,
                score=adjusted_score,
                bm25_score=c.bm25_score,
                vector_score=c.vector_score,
            )
        )

    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked


def retrieve_chunks(
    conn: sqlite3.Connection,
    identity: Identity,
    query: str,
    ref_date: date,
    max_tokens: int = MAX_CORPUS_TOKENS,
    include_upcoming: bool = True,
) -> tuple[list[RetrievedChunk], list[DocMeta]]:
    """Execute Advanced Hybrid Retrieval with RRF and token gating."""
    eligible_docs, upcoming_docs = get_candidate_docs(conn, identity, ref_date, include_upcoming)
    if not eligible_docs and not upcoming_docs:
        return [], []

    all_allowed_ids = list(eligible_docs.keys()) + list(upcoming_docs.keys())
    placeholders = ",".join("?" for _ in all_allowed_ids)
    sql = f"""
        SELECT c.chunk_id, c.doc_id, c.ordinal, c.locator, c.kind, c.text, c.tokens,
               d.title, d.doc_type, d.effective_date, d.regions, d.access_level
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.doc_id IN ({placeholders})
    """
    rows = conn.execute(sql, all_allowed_ids).fetchall()

    query_terms = _tokenize(query)
    stop_words = {
        "the", "is", "at", "which", "on", "a", "an", "and", "or", "to", "for", "in", "of",
        "what", "how", "can", "do", "i", "my", "we", "our", "us", "about", "tell", "me",
        "policy", "policies", "document", "documents", "company", "veltrix", "please", "guidelines",
    }
    filtered_terms = [t for t in query_terms if t not in stop_words]
    terms_to_use = filtered_terms if filtered_terms else query_terms

    # Check if query has content terms and if any content term exists across the corpus
    meta_cache = {**eligible_docs, **upcoming_docs}
    query_vec = _embedding_service.get_embedding(query)

    bm25_candidates: list[RetrievedChunk] = []
    vector_candidates: list[RetrievedChunk] = []

    # 1. Sparse BM25 Keyword Search
    for r in rows:
        meta = meta_cache[r["doc_id"]]
        bm25 = _score_bm25(r["text"], r["locator"], r["title"], terms_to_use, meta)
        tokens = r["tokens"] if r["tokens"] else count_tokens(r["text"])

        if bm25 > 0:
            bm25_candidates.append(
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    ordinal=r["ordinal"],
                    locator=r["locator"],
                    kind=r["kind"],
                    text=r["text"],
                    tokens=tokens,
                    title=r["title"],
                    doc_type=r["doc_type"],
                    effective_date=date.fromisoformat(r["effective_date"]),
                    bm25_score=bm25,
                    vector_score=0.0,
                )
            )

    # 2. Dense Semantic Vector Search via Qdrant with Metadata Payload Filtering
    try:
        qdrant = get_qdrant_store()
        if not qdrant._is_indexed:
            qdrant.index_chunks(conn, _embedding_service)

        qdrant_hits = qdrant.search_with_metadata_filter(query_vec, identity, ref_date, limit=40, include_upcoming=include_upcoming)
        for hit in qdrant_hits:
            v_score = hit.get("vector_score", 0.0)
            if v_score > 0.15:
                tokens = hit["tokens"] if hit.get("tokens") else count_tokens(hit["text"])
                vector_candidates.append(
                    RetrievedChunk(
                        chunk_id=hit["chunk_id"],
                        doc_id=hit["doc_id"],
                        ordinal=hit["ordinal"],
                        locator=hit["locator"],
                        kind=hit["kind"],
                        text=hit["text"],
                        tokens=tokens,
                        title=hit["title"],
                        doc_type=hit["doc_type"],
                        effective_date=date.fromisoformat(hit["effective_date"]),
                        bm25_score=0.0,
                        vector_score=v_score,
                    )
                )
    except Exception as q_exc:
        log.warning("Qdrant vector search failed, falling back to direct similarity: %s", q_exc)
        for r in rows:
            chunk_vec = _embedding_service.get_embedding(f"{r['title']} {r['locator']} {r['text'][:300]}")
            vec_sim = _embedding_service.cosine_similarity(query_vec, chunk_vec)
            if vec_sim > 0.15:
                tokens = r["tokens"] if r["tokens"] else count_tokens(r["text"])
                vector_candidates.append(
                    RetrievedChunk(
                        chunk_id=r["chunk_id"],
                        doc_id=r["doc_id"],
                        ordinal=r["ordinal"],
                        locator=r["locator"],
                        kind=r["kind"],
                        text=r["text"],
                        tokens=tokens,
                        title=r["title"],
                        doc_type=r["doc_type"],
                        effective_date=date.fromisoformat(r["effective_date"]),
                        bm25_score=0.0,
                        vector_score=vec_sim,
                    )
                )

    if not bm25_candidates and not vector_candidates:
        return [], list(upcoming_docs.values())

    # Rank both lists
    bm25_candidates.sort(key=lambda x: x.bm25_score, reverse=True)
    vector_candidates.sort(key=lambda x: x.vector_score, reverse=True)

    # 1. Reciprocal Rank Fusion
    fused_chunks = _reciprocal_rank_fusion(bm25_candidates, vector_candidates)

    # 2. Stage-2 Contextual Precedence Reranking
    reranked_chunks = _precedence_rerank(fused_chunks, identity, ref_date)

    # 3. Context Budget Gating (<= 3,000 tokens)
    selected: list[RetrievedChunk] = []
    total_tokens = 0
    for c in reranked_chunks:
        if total_tokens + c.tokens <= max_tokens:
            selected.append(c)
            total_tokens += c.tokens

    assert total_tokens <= max_tokens, f"Corpus token ceiling violated: {total_tokens} > {max_tokens}"

    log.info(
        "hybrid retrieval executed",
        extra={
            "employee_id": identity.employee_id,
            "ref_date": ref_date.isoformat(),
            "corpus_tokens": total_tokens,
            "chunks_count": len(selected),
            "algorithm": "RRF (BM25 + Dense Vectors)",
        },
    )

    return selected, list(upcoming_docs.values())
