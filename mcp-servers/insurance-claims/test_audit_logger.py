
"""Automated tests for structured audit logging."""

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


if __name__ == "__main__":
    unittest.main()
