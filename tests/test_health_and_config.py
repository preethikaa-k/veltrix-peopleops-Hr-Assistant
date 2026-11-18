import threading

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from tests.conftest import wait_ready


def test_health_is_503_before_startup(settings):
    app = create_app(settings, ingest_fn=lambda s: None)
    assert TestClient(app).get("/health").status_code == 503  # lifespan not run yet


def test_health_stays_503_until_ingestion_finishes(settings):
    gate = threading.Event()
    app = create_app(settings, ingest_fn=lambda s: gate.wait(5))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        gate.set()
        wait_ready(c)
        assert c.get("/health").json() == {"status": "ok"}


def test_failed_ingestion_never_reports_ready(settings):
    def boom(_):
        raise RuntimeError("bad corpus")

    app = create_app(settings, ingest_fn=boom)
    with TestClient(app) as c:
        import time

        time.sleep(0.2)
        assert c.get("/health").status_code == 503


def test_app_today_parsing_and_validation():
    assert Settings.from_env({"APP_TODAY": "2026-04-01"}).app_today.isoformat() == "2026-04-01"
    with pytest.raises(ValueError):
        Settings.from_env({"APP_TODAY": "01/04/2026"})
    with pytest.raises(ValueError):
        Settings.from_env({"LLM_PROVIDER": "nope"})
