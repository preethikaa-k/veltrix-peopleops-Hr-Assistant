import io
import logging

from app.logging_setup import register_sensitive_values, scrub, setup_logging
from tests.conftest import EMPLOYEES, post


def test_scrubs_emails_and_phones_but_keeps_dates_and_ids():
    out = scrub("mail riya@veltrix.example call +91 98765 43210 on 2026-04-01 for E1001 request 2026-04-01")
    assert "riya@veltrix.example" not in out and "98765" not in out
    assert "2026-04-01" in out and "E1001" in out


def test_registered_names_are_scrubbed_case_insensitively():
    register_sensitive_values(["Riya Sharma"])
    assert "riya sharma" not in scrub("user riya sharma logged in").lower()


def test_no_pii_in_logs_end_to_end(client):
    buf = io.StringIO()
    setup_logging("DEBUG", stream=buf)
    post(client, "E1001", message="my email is riya.sharma@veltrix.example and phone +91 98765 43210, I am Riya Sharma")
    post(client, "E9999", message="Marcus Webb")
    logging.getLogger("veltrix.test").info("oops leaked name Marcus Webb and arjun.rao@veltrix.example")
    logs = buf.getvalue()
    assert logs, "expected log output"
    for _id, name, email, phone, *_ in EMPLOYEES:
        assert name not in logs and email not in logs and phone not in logs
    assert "E1001" in logs  # employee IDs are allowed and useful
