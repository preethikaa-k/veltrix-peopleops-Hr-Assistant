"""Qdrant Vector Database service with metadata payload filtering (Section 10.3 / 7.2).

Provides high-performance vector search with pre-filtering on role, region, and effective date.
Runs embedded locally in memory or disk with zero external dependencies.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from typing import Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.db import Identity
from app.embeddings import EmbeddingService
from app.policy import can_read

log = logging.getLogger("veltrix.qdrant")

COLLECTION_NAME = "veltrix_policy_chunks"


class QdrantVectorStore:
    def __init__(self, path: Optional[str] = None):
        """Initialize Qdrant client. If path is None, runs in-memory."""
        if path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
        self._dim: Optional[int] = None
        self._is_indexed = False

    def init_collection(self, dim: int) -> None:
        """Create or reset collection with cosine distance."""
        self._dim = dim
        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self._is_indexed = False

    def index_chunks(
        self,
        conn: sqlite3.Connection,
        embedding_service: EmbeddingService,
    ) -> int:
        """Index all document chunks from SQLite into Qdrant with full metadata payload."""
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.ordinal, c.locator, c.kind, c.text, c.tokens,
                   d.title, d.doc_type, d.regions, d.effective_date, d.access_level
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE d.status = 'ok'
            ORDER BY c.doc_id, c.ordinal
            """
        ).fetchall()

        if not rows:
            return 0

        # Sample first vector to get dimension
        sample_vec = embedding_service.get_embedding(f"{rows[0]['title']} {rows[0]['locator']}")
        dim = len(sample_vec)
        self.init_collection(dim)

        points: list[PointStruct] = []
        for idx, r in enumerate(rows, start=1):
            text_to_embed = f"{r['title']} {r['locator']} {r['text'][:300]}"
            vec = embedding_service.get_embedding(text_to_embed)
            if len(vec) != dim:
                continue

            try:
                regions = json.loads(r["regions"]) if r["regions"] else ["GLOBAL"]
            except Exception:
                regions = ["GLOBAL"]

            eff_str = str(r["effective_date"]).strip()
            eff_int = int(eff_str.replace("-", "")) if eff_str else 0

            payload = {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "ordinal": r["ordinal"],
                "locator": r["locator"],
                "kind": r["kind"],
                "text": r["text"],
                "tokens": r["tokens"],
                "title": r["title"],
                "doc_type": r["doc_type"],
                "regions": regions,
                "effective_date": eff_str,
                "effective_date_int": eff_int,
                "access_level": r["access_level"],
            }

            points.append(PointStruct(id=idx, vector=vec, payload=payload))

        # Batch upsert into Qdrant
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        self._is_indexed = True
        log.info("indexed chunks into Qdrant vector database", extra={"count": len(points), "dim": dim})
        return len(points)

    def search_with_metadata_filter(
        self,
        query_vec: Sequence[float],
        identity: Identity,
        ref_date: date,
        limit: int = 50,
        include_upcoming: bool = True,
    ) -> list[dict]:
        """Perform vector search in Qdrant applying strict access control and region metadata filters."""
        if not self._is_indexed or not self.client.collection_exists(COLLECTION_NAME):
            return []

        # 1. Access Level filter based on user role (Spec 7.2)
        allowed_access = [lvl for lvl in ["all", "employee", "manager", "hr"] if can_read(identity.role, lvl)]

        # 2. Region filter: Global or user region (Spec 7.2)
        allowed_regions = ["GLOBAL", identity.region]

        # Build Qdrant metadata filter
        conditions = [
            FieldCondition(key="access_level", match=MatchAny(any=allowed_access)),
            FieldCondition(key="regions", match=MatchAny(any=allowed_regions)),
        ]

        if not include_upcoming:
            ref_date_int = int(ref_date.strftime("%Y%m%d"))
            conditions.append(FieldCondition(key="effective_date_int", range=Range(lte=ref_date_int)))

        q_filter = Filter(must=conditions)

        res = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=list(query_vec),
            query_filter=q_filter,
            limit=limit,
        )

        results: list[dict] = []
        for pt in res.points:
            item = dict(pt.payload)
            item["vector_score"] = float(pt.score)
            results.append(item)

        return results


# Global singleton instance
_qdrant_store: Optional[QdrantVectorStore] = None


def get_qdrant_store() -> QdrantVectorStore:
    global _qdrant_store
    if _qdrant_store is None:
        _qdrant_store = QdrantVectorStore()
    return _qdrant_store
