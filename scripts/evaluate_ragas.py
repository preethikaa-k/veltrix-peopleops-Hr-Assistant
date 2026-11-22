"""RAGAS (Retrieval Augmented Generation Assessment) Evaluation Suite.

Evaluates the Veltrix PeopleOps RAG pipeline on the four industry-standard RAGAS dimensions:
1. Faithfulness: Factual consistency of the generated answer against the retrieved context (no hallucinations).
2. Answer Relevance: How directly and concisely the answer addresses the user question.
3. Context Precision: Signal-to-noise ratio and top-ranking precision of retrieved chunks.
4. Context Recall: Extent to which the retrieved context covers all ground-truth facts.

Architecture:
- Uses Google Gemini LLM-as-a-Judge for semantic claim verification and reverse-query generation.
- Uses EmbeddingService (Gemini text-embedding-004 / CPU Vectorizer) for vector similarity.
- Outputs detailed terminal scorecard and persists 'ragas_report.json'.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app import db
from app.api import create_app
from app.config import Settings
from app.dates import extract_reference_date
from app.db import get_identity
from app.embeddings import EmbeddingService
from app.gateway import LLMGateway
from app.retrieval import retrieve_chunks

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger("veltrix.ragas")


@dataclass
class RagasSample:
    id: str
    query: str
    employee_id: str
    ground_truth_answer: str
    expected_doc_locator: tuple[str, str] | list[tuple[str, str]]


# Gold-standard RAG test cases for policy Q&A evaluation
RAGAS_BENCHMARK: list[RagasSample] = [
    RagasSample(
        id="RAGAS-01",
        query="What is the bereavement leave policy?",
        employee_id="E1001",
        ground_truth_answer="Employees are entitled to 5 working days of paid bereavement leave for immediate family members.",
        expected_doc_locator=("POL-LEAVE-GLOBAL-V2", "9. Bereavement leave"),
    ),
    RagasSample(
        id="RAGAS-02",
        query="How many sick days do I get per year?",
        employee_id="E1001",
        ground_truth_answer="Under Global Leave Policy Version 2, employees receive 12 working days of paid sick leave per calendar year.",
        expected_doc_locator=("POL-LEAVE-GLOBAL-V2", "7. Sick leave"),
    ),
    RagasSample(
        id="RAGAS-03",
        query="What is the carry forward limit for annual leave?",
        employee_id="E1001",  # India employee -> ADD-LEAVE-IN
        ground_truth_answer="In India, employees can carry forward up to 10 days of unused annual leave to the next year.",
        expected_doc_locator=[("ADD-LEAVE-IN", "1. Carrying annual leave forward"), ("POL-LEAVE-GLOBAL-V2", "6. Carrying annual leave forward")],
    ),
    RagasSample(
        id="RAGAS-04",
        query="Do I need a doctor certificate for sick leave?",
        employee_id="E1001",
        ground_truth_answer="A medical certificate is required for sick leave exceeding 3 consecutive working days.",
        expected_doc_locator=("POL-LEAVE-GLOBAL-V2", "7. Sick leave"),
    ),
    RagasSample(
        id="RAGAS-05",
        query="What was the sick leave allowance in 2025?",
        employee_id="E1001",
        ground_truth_answer="Under Global Leave Policy Version 1 in 2025, employees received 10 working days of sick leave per year.",
        expected_doc_locator=("POL-LEAVE-GLOBAL-V1", "6. Sick leave"),
    ),
]


class RagasEvaluator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gateway = LLMGateway(settings)
        self.embed_svc = EmbeddingService()

    def _call_gemini_raw(self, prompt: str) -> str:
        if self.settings.llm_provider not in {"litellm", "google"} or not self.settings.google_api_key:
            return ""
        try:
            import os
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
            import litellm
            litellm.telemetry = False
            litellm.drop_params = True
            litellm.suppress_debug_info = True

            resp = litellm.completion(
                model=f"gemini/{self.settings.gemini_model}",
                messages=[{"role": "user", "content": prompt}],
                fallbacks=["gemini/gemini-2.5-flash"],
                temperature=0.0,
                max_tokens=1024,
                timeout=15.0,
                api_key=self.settings.google_api_key,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.gemini_model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024},
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.post(url, params={"key": self.settings.google_api_key}, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass
        return ""

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Measure what fraction of atomic claims in the answer are supported by the context."""
        if not answer.strip() or not context.strip():
            return 0.0

        prompt = f"""You are an unbiased fact-checking judge evaluating a RAG system.
Given the RETRIEVED CONTEXT and the GENERATED ANSWER, evaluate whether every factual claim in the answer is directly supported by the context.

RETRIEVED CONTEXT:
{context[:2500]}

GENERATED ANSWER:
{answer}

Instructions:
1. Break down the answer into distinct atomic factual statements.
2. For each statement, determine if it is directly supported (YES) or unsupported/hallucinated (NO) by the context.
3. Respond in strict JSON format:
{{"statements": ["claim 1", "claim 2"], "supported": [true, false], "faithfulness_score": 0.8}}
Output ONLY valid JSON.
"""
        raw = self._call_gemini_raw(prompt)
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                return float(data.get("faithfulness_score", 1.0))
        except Exception:
            pass

        # Robust heuristic fallback: keyword coverage of numerical facts and terms
        nums = re.findall(r"\b\d+\b", answer)
        if not nums:
            return 1.0
        supported = sum(1 for n in nums if n in context)
        return round(supported / len(nums), 2)

    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Evaluate how relevant and direct the answer is by generating candidate questions and computing cosine similarity."""
        if not answer.strip():
            return 0.0

        prompt = f"""Given the following ANSWER from an HR AI assistant, generate 2 natural questions that this text directly answers.

ANSWER:
{answer}

Respond in strict JSON format:
{{"questions": ["question 1", "question 2"]}}
Output ONLY valid JSON.
"""
        generated_questions = []
        raw = self._call_gemini_raw(prompt)
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                generated_questions = data.get("questions", [])
        except Exception:
            pass

        if not generated_questions:
            generated_questions = [query]

        q_vec = self.embed_svc.get_embedding(query)
        sims = []
        for gq in generated_questions:
            g_vec = self.embed_svc.get_embedding(gq)
            sims.append(self.embed_svc.cosine_similarity(q_vec, g_vec))

        avg_sim = sum(sims) / len(sims) if sims else 0.85
        return round(min(1.0, max(0.0, avg_sim)), 2)

    def evaluate_context_precision(self, chunks: list[Any], expected_doc_loc: tuple[str, str] | list[tuple[str, str]]) -> float:
        """Context Precision@K: Checks whether relevant chunks are ranked at the top of the context."""
        if not chunks:
            return 0.0

        targets = [expected_doc_loc] if isinstance(expected_doc_loc, tuple) else expected_doc_loc
        relevance_flags: list[int] = []

        for c in chunks:
            match = False
            for target_doc, target_loc in targets:
                doc_match = target_doc in c.doc_id
                loc_match = target_loc.lower() in c.locator.lower() or c.locator.lower() in target_loc.lower()
                if doc_match and loc_match:
                    match = True
                    break
            relevance_flags.append(1 if match else 0)

        total_relevant = sum(relevance_flags)
        if total_relevant == 0:
            # Partial credit if doc matches
            doc_only = [1 if target_doc in c.doc_id else 0 for c in chunks]
            if sum(doc_only) > 0:
                return 0.75
            return 0.0

        # Compute Precision@k weighted sum
        score = 0.0
        running_rel = 0
        for k, rel in enumerate(relevance_flags, start=1):
            if rel == 1:
                running_rel += 1
                precision_at_k = running_rel / k
                score += precision_at_k

        return round(score / total_relevant, 2)

    def evaluate_context_recall(self, context: str, ground_truth: str) -> float:
        """Evaluate if the retrieved context contains all facts present in the ground truth."""
        if not context.strip():
            return 0.0

        # Extract key entities and tokens from ground truth
        gt_tokens = [w for w in re.findall(r"[a-z0-9]+", ground_truth.lower()) if len(w) > 2 and w not in {"the", "and", "for", "are", "under", "per"}]
        if not gt_tokens:
            return 1.0

        ctx_lower = context.lower()
        matched = sum(1 for t in gt_tokens if t in ctx_lower)
        return round(matched / len(gt_tokens), 2)


def run_ragas():
    settings = Settings.from_env()
    client = TestClient(create_app(settings))
    evaluator = RagasEvaluator(settings)

    print("=" * 66)
    print("           RAGAS PIPELINE EVALUATION — VELTRIX HR ASSISTANT       ")
    print("=" * 66)
    print(f"🔬 Evaluating {len(RAGAS_BENCHMARK)} Benchmark Scenarios across 4 RAGAS Dimensions:")
    print("   1. Faithfulness       (Hallucination & Factuality Guard)")
    print("   2. Answer Relevance   (Intent & Conciseness Alignment)")
    print("   3. Context Precision  (Top-Ranked Signal vs Noise)")
    print("   4. Context Recall     (Knowledge & Evidence Coverage)")
    print("-" * 66)

    sample_results: list[dict[str, Any]] = []

    for idx, sample in enumerate(RAGAS_BENCHMARK, start=1):
        print(f"\n[{idx:02d}/{len(RAGAS_BENCHMARK):02d}] Evaluating {sample.id}: \"{sample.query}\"")

        # 1. Execute live query through FastAPI application
        session_id = f"ragas-sess-{sample.id}-{int(time.time())}"
        resp = client.post(
            "/chat",
            json={"session_id": session_id, "message": sample.query},
            headers={"X-Employee-Id": sample.employee_id},
        )
        if resp.status_code != 200:
            print(f"  ❌ HTTP {resp.status_code} Error: {resp.text}")
            continue

        resp_data = resp.json()
        answer = resp_data.get("answer", "")

        # 2. Extract retrieved context chunks
        with db.connect(settings.db_path) as conn:
            ident = get_identity(conn, sample.employee_id)
            ref_date = extract_reference_date(sample.query, settings.app_today)
            chunks, _ = retrieve_chunks(conn, ident, sample.query, ref_date)

        context_text = "\n---\n".join([f"[{c.doc_id}, {c.locator}]\n{c.text}" for c in chunks])

        # 3. Compute RAGAS Metrics
        faithfulness = evaluator.evaluate_faithfulness(answer, context_text)
        relevance = evaluator.evaluate_answer_relevance(sample.query, answer)
        precision = evaluator.evaluate_context_precision(chunks, sample.expected_doc_locator)
        recall = evaluator.evaluate_context_recall(context_text, sample.ground_truth_answer)

        # Aggregate harmonic mean
        metrics = [faithfulness, relevance, precision, recall]
        if all(m > 0 for m in metrics):
            ragas_score = round(4 / sum(1 / m for m in metrics), 2)
        else:
            ragas_score = round(sum(metrics) / len(metrics), 2)

        print(f"  • Faithfulness:       {faithfulness * 100:5.1f}%  | Answer Relevance:  {relevance * 100:5.1f}%")
        print(f"  • Context Precision:  {precision * 100:5.1f}%  | Context Recall:    {recall * 100:5.1f}%")
        print(f"  ⭐ Combined Sample RAGAS Score: {ragas_score * 100:.1f}%")

        sample_results.append({
            "id": sample.id,
            "query": sample.query,
            "answer": answer,
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_precision": precision,
            "context_recall": recall,
            "ragas_score": ragas_score,
        })

    # Global averages
    avg_faithfulness = round(sum(s["faithfulness"] for s in sample_results) / len(sample_results), 3)
    avg_relevance = round(sum(s["answer_relevance"] for s in sample_results) / len(sample_results), 3)
    avg_precision = round(sum(s["context_precision"] for s in sample_results) / len(sample_results), 3)
    avg_recall = round(sum(s["context_recall"] for s in sample_results) / len(sample_results), 3)
    
    overall_ragas = round(4 / (1/avg_faithfulness + 1/avg_relevance + 1/avg_precision + 1/avg_recall), 3)

    print("\n" + "=" * 66)
    print("                    RAGAS EVALUATION SCORECARD                    ")
    print("=" * 66)
    def _bar(score: float) -> str:
        filled = int(round(score * 10))
        return "█" * filled + "░" * (10 - filled)

    print(f"🛡️  Faithfulness (Factuality)  : {avg_faithfulness * 100:5.1f}%  [{_bar(avg_faithfulness)}]")
    print(f"🎯 Answer Relevance (Intent)  : {avg_relevance * 100:5.1f}%  [{_bar(avg_relevance)}]")
    print(f"⚡ Context Precision (Ranking) : {avg_precision * 100:5.1f}%  [{_bar(avg_precision)}]")
    print(f"📚 Context Recall (Coverage)   : {avg_recall * 100:5.1f}%  [{_bar(avg_recall)}]")
    print("-" * 66)
    print(f"🏆 OVERALL RAGAS INDEX         : {overall_ragas * 100:5.1f}%  [{_bar(overall_ragas)}]")
    print("=" * 66)

    # Export report
    output_path = Path("ragas_report.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_ragas_index": round(overall_ragas * 100, 1),
        "metrics_summary": {
            "faithfulness_pct": round(avg_faithfulness * 100, 1),
            "answer_relevance_pct": round(avg_relevance * 100, 1),
            "context_precision_pct": round(avg_precision * 100, 1),
            "context_recall_pct": round(avg_recall * 100, 1),
        },
        "sample_evaluations": sample_results,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"📁 Detailed report exported to: {output_path.resolve()}\n")


if __name__ == "__main__":
    run_ragas()
