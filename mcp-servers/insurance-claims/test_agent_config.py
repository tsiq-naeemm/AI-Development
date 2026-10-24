import os
import unittest
from unittest.mock import patch

from langgraph_rag_agent import (
    nonempty_string_env,
    positive_int_env,
    positive_float_env,
)


class TestAgentConfig(unittest.TestCase):

    def test_model_name_validation(self):
        with patch.dict(os.environ, {"TEST_MODEL": " llama3.2 "}):
            self.assertEqual(
                nonempty_string_env("TEST_MODEL", "default"),
                "llama3.2",
            )

        with patch.dict(os.environ, {"TEST_MODEL": "   "}):
            with self.assertRaises(ValueError):
                nonempty_string_env("TEST_MODEL", "default")

    def test_recursion_limit_validation(self):
        with patch.dict(os.environ, {"TEST_LIMIT": "20"}):
            self.assertEqual(
                positive_int_env("TEST_LIMIT", 13),
                20,
            )

        for invalid in ("0", "-1", "abc"):
            with self.subTest(value=invalid):
                with patch.dict(os.environ, {"TEST_LIMIT": invalid}):
                    with self.assertRaises(ValueError):
                        positive_int_env("TEST_LIMIT", 13)

    def test_timeout_validation(self):
        with patch.dict(os.environ, {"TEST_TIMEOUT": "180.5"}):
            self.assertEqual(
                positive_float_env("TEST_TIMEOUT", 120.0),
                180.5,
            )

        for invalid in ("0", "-10", "abc", "nan", "inf"):
            with self.subTest(value=invalid):
                with patch.dict(os.environ, {"TEST_TIMEOUT": invalid}):
                    with self.assertRaises(ValueError):
                        positive_float_env("TEST_TIMEOUT", 120.0)


if __name__ == "__main__":
    unittest.main()