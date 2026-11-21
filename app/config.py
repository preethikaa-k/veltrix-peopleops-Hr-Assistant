"""Environment-driven settings. Fail fast on bad values so misconfiguration never reaches runtime."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    manifest_path: Path
    db_path: Path
    app_today: date
    llm_provider: str
    log_level: str = "INFO"
    log_file: Optional[str] = None
    max_body_bytes: int = 16_384
    # Hard per-request ceiling; the spec fails anything slower than 30 s.
    request_timeout_s: float = 25.0
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        e = dict(os.environ) if env is None else dict(env)
        if env is None:
            env_file = Path(".env")
            if env_file.is_file():
                try:
                    for line in env_file.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k and k not in e:
                                e[k] = v
                except Exception:
                    pass
        raw_today = (e.get("APP_TODAY") or "").strip()
        try:
            today = date.fromisoformat(raw_today) if raw_today else date.today()
        except ValueError as exc:
            raise ValueError(f"APP_TODAY must be YYYY-MM-DD, got {raw_today!r}") from exc
        provider = (e.get("LLM_PROVIDER") or "google").strip().lower()
        if provider not in {"anthropic", "openai", "google", "mock"}:
            raise ValueError(f"LLM_PROVIDER must be anthropic|openai|google|mock, got {provider!r}")
        return cls(
            corpus_dir=Path(e.get("CORPUS_DIR") or "./data/corpus"),
            manifest_path=Path(e.get("MANIFEST_PATH") or "./data/manifest.json"),
            db_path=Path(e.get("DB_PATH") or "./data/veltrix.db"),
            app_today=today,
            llm_provider=provider,
            log_level=(e.get("LOG_LEVEL") or "INFO").upper(),
            log_file=(e.get("LOG_FILE") or None),
            google_api_key=(e.get("GOOGLE_API_KEY") or e.get("GEMINI_API_KEY") or None),
            anthropic_api_key=(e.get("ANTHROPIC_API_KEY") or None),
            openai_api_key=(e.get("OPENAI_API_KEY") or None),
            gemini_model=(e.get("GEMINI_MODEL") or "gemini-1.5-flash").strip(),
        )
