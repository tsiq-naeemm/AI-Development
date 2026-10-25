"""Privacy-conscious structured JSON audit logging."""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "audit.jsonl"

LOGGER_NAME = "insurance_claims.audit"


class JsonAuditFormatter(logging.Formatter):
    """Write one JSON object per audit event."""

    def format(self, record):
        event = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }

        for field in (
            "request_id",
            "claim_id",
            "model",
            "validation_status",
            "duration_ms",
            "error_code",
        ):
            value = getattr(record, field, None)
            if value is not None:
                event[field] = value

        return json.dumps(event)


def get_audit_logger():
    """Return a configured audit logger."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        )
        handler.setFormatter(JsonAuditFormatter())
        logger.addHandler(handler)

    return logger


def new_request_id():
    """Generate a unique identifier for a validation request."""
    return str(uuid.uuid4())


def elapsed_ms(start_time):
    """Return elapsed time in milliseconds."""
    return round((time.perf_counter() - start_time) * 1000, 2)
