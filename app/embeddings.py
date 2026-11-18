"""Dense semantic embedding provider with Gemini API adapter and local CPU vectorizer.

Spec compliance:
- Section 10.3: Supports Gemini text-embedding-004 API and local CPU embedding.
- Fast, zero external heavy dependency, thread-safe in-memory caching.
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from typing import Optional, Sequence

import httpx

log = logging.getLogger("veltrix.embeddings")

GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"


class EmbeddingService:
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-004"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model
        self._cache: dict[str, list[float]] = {}

    def get_embedding(self, text: str) -> list[float]:
        """Compute embedding vector. Uses Gemini API if key is available, else local CPU vectorizer."""
        text = text.strip()
        if not text:
            return []

        if text in self._cache:
            return self._cache[text]

        if self.api_key:
            try:
                vec = self._call_gemini_embedding(text)
                self._cache[text] = vec
                return vec
            except Exception as exc:
                log.warning("Gemini embedding API call failed; falling back to local vectorizer", extra={"error": str(exc)})

        vec = self._local_vectorize(text)
        self._cache[text] = vec
        return vec

    def _call_gemini_embedding(self, text: str) -> list[float]:
        url = GEMINI_EMBED_URL.format(model=self.model)
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text[:2000]}]},
        }
        with httpx.Client(timeout=10.0) as client:
            res = client.post(url, params={"key": self.api_key}, json=payload)
            res.raise_for_status()
            data = res.json()
            return data["embedding"]["values"]

    def _local_vectorize(self, text: str, dim: int = 128) -> list[float]:
        """Fast, robust subword hashing vectorizer running entirely on CPU with zero dependencies."""
        words = re.findall(r"[a-z0-9]+", text.lower())
        if not words:
            return [0.0] * dim

        # Extract unigrams and bigrams
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")

        counts = Counter(tokens)
        vec = [0.0] * dim
        for tok, count in counts.items():
            # Hash token to dimension index
            idx = abs(hash(tok)) % dim
            # Log-frequency weighting
            vec[idx] += 1.0 + math.log(count)

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        """Compute cosine similarity between two normalized vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        return sum(a * b for a, b in zip(vec1, vec2))
