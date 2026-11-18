"""Ingestion package: parsers (PDF/DOCX/HTML/MD + OCR), chunker, holiday extraction, pipeline."""
from app.ingestion.pipeline import run_ingestion

__all__ = ["run_ingestion"]
