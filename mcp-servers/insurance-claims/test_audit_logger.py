
"""Automated tests for structured audit logging."""

from asyncio import events
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import audit_logger


class TestAuditLogger(unittest.TestCase):

    def test_validation_started_event(self):
        """Audit records contain expected metadata."""

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.jsonl"

            with patch.object(audit_logger, "LOG_DIR", Path(tmpdir)), \
                 patch.object(audit_logger, "LOG_FILE", log_file):

                logger = audit_logger.get_audit_logger()

                # Temporarily replace the production file handler.
                original_handlers = logger.handlers[:]

                for handler in original_handlers:
                    logger.removeHandler(handler)

                try:
                    logger = audit_logger.get_audit_logger()

                    request_id = audit_logger.new_request_id()

                    logger.info(
                        "VALIDATION_STARTED",
                        extra={
                            "request_id": request_id,
                            "claim_id": "CLM001",
                        },
                    )

                    for handler in logger.handlers:
                        handler.flush()

                    record = json.loads(
                        log_file.read_text(encoding="utf-8").strip()
                    )

                    self.assertEqual(
                        record["event"], "VALIDATION_STARTED"
                    )
                    self.assertEqual(
                        record["claim_id"], "CLM001"
                    )
                    self.assertEqual(
                        record["request_id"], request_id
                    )
                    self.assertEqual(record["level"], "INFO")
                    self.assertIn("timestamp", record)

                    # Ensure clinical content is not logged.
                    self.assertNotIn("medical_evidence", record)
                    self.assertNotIn("patient_name", record)

                finally:
                    for handler in logger.handlers[:]:
                        logger.removeHandler(handler)
                        handler.close()

                    for handler in original_handlers:
                        logger.addHandler(handler)

    def test_request_id_correlation(self):
        """Started and completed events share the same request ID."""

        formatter = audit_logger.JsonAuditFormatter()
        request_id = audit_logger.new_request_id()

        events = []

        for event_name in (
            "VALIDATION_STARTED",
            "VALIDATION_COMPLETED",
        ):
            record = audit_logger.logging.LogRecord(
                name="insurance_claims.audit",
                level=audit_logger.logging.INFO,
                pathname=__file__,
                lineno=0,
                msg=event_name,
                args=(),
                exc_info=None,
            )

            record.request_id = request_id
            record.claim_id = "CLM001"
            record.model = "llama3.2"

            if event_name == "VALIDATION_COMPLETED":
                record.validation_status = "PASS"
                record.duration_ms = 1250.5

            events.append(json.loads(formatter.format(record)))

        self.assertEqual(len(events), 2)
        self.assertEqual(
            events[0]["request_id"],
            events[1]["request_id"],
        )
        self.assertEqual(events[0]["claim_id"], "CLM001")
        self.assertEqual(events[1]["claim_id"], "CLM001")
        self.assertEqual(events[1]["validation_status"], "PASS")
        self.assertEqual(events[1]["duration_ms"], 1250.5)
        self.assertEqual(events[0]["model"], "llama3.2")
        self.assertEqual(events[1]["model"], "llama3.2")

    def test_validation_failed_event(self):
        """Failure events contain error metadata without sensitive details."""

        formatter = audit_logger.JsonAuditFormatter()
        request_id = audit_logger.new_request_id()

        record = audit_logger.logging.LogRecord(
            name="insurance_claims.audit",
            level=audit_logger.logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg="VALIDATION_FAILED",
            args=(),
            exc_info=None,
        )

        record.request_id = request_id
        record.claim_id = "CLM001"
        record.error_code = "RuntimeError"
        record.duration_ms = 15.75

        event = json.loads(formatter.format(record))

        self.assertEqual(event["event"], "VALIDATION_FAILED")
        self.assertEqual(event["level"], "ERROR")
        self.assertEqual(event["request_id"], request_id)
        self.assertEqual(event["claim_id"], "CLM001")
        self.assertEqual(event["error_code"], "RuntimeError")
        self.assertEqual(event["duration_ms"], 15.75)

        # Exception messages and medical details must not be recorded.
        self.assertNotIn("error_message", event)
        self.assertNotIn("medical_evidence", event)
        self.assertNotIn("patient_name", event)


if __name__ == "__main__":
    unittest.main()
