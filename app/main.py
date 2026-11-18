"""ASGI entry point: `uvicorn app.main:app`. Single worker: readiness state is in-process."""
from app.api import create_app

app = create_app()
