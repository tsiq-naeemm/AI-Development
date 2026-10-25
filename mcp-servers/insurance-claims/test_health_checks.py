import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from health_checks import check_ollama_health, check_mcp_health


class TestOllamaHealth(unittest.IsolatedAsyncioTestCase):

    @patch("health_checks.AsyncClient")
    async def test_healthy_model(self, mock_client_class):
        client = mock_client_class.return_value
        client.list = AsyncMock(
            return_value=SimpleNamespace(
                models=[
                    SimpleNamespace(model="llama3.2:latest")
                ]
            )
        )

        result = await check_ollama_health("llama3.2:latest")

        self.assertEqual(result["status"], "HEALTHY")
        self.assertEqual(result["service"], "ollama")
        client.list.assert_awaited_once()

    @patch("health_checks.AsyncClient")
    async def test_model_not_found(self, mock_client_class):
        client = mock_client_class.return_value
        client.list = AsyncMock(
            return_value=SimpleNamespace(
                models=[
                    SimpleNamespace(model="embeddinggemma:latest")
                ]
            )
        )

        result = await check_ollama_health("llama3.2:latest")

        self.assertEqual(result["status"], "MODEL_NOT_FOUND")

    @patch("health_checks.AsyncClient")
    async def test_server_unavailable(self, mock_client_class):
        client = mock_client_class.return_value
        client.list = AsyncMock(
            side_effect=ConnectionError("SIMULATED_FAILURE")
        )

        result = await check_ollama_health("llama3.2:latest")

        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["error_code"], "ConnectionError")
        self.assertNotIn("SIMULATED_FAILURE", str(result))


class TestMCPHealth(unittest.IsolatedAsyncioTestCase):

    async def test_healthy_mcp_server(self):
        """MCP health check succeeds when all required tools exist."""

        required_tools = [
            "get_claim_status",
            "get_claim_amount",
            "list_open_claims",
            "get_claim_medical_evidence",
            "search_medical_documents",
            "validate_claim_referral",
        ]

        mock_stdio = AsyncMock()
        mock_stdio.__aenter__.return_value = ("read", "write")

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session

        mock_session.list_tools.return_value = SimpleNamespace(
            tools=[
                SimpleNamespace(name=name)
                for name in required_tools
            ]
        )

        with (
            patch("health_checks.stdio_client", return_value=mock_stdio),
            patch("health_checks.ClientSession", return_value=mock_session),
        ):
            result = await check_mcp_health()

        self.assertEqual(result["service"], "mcp")
        self.assertEqual(result["status"], "HEALTHY")
        self.assertEqual(result["tool_count"], 6)
        self.assertEqual(result["missing_tools"], [])

        mock_session.initialize.assert_awaited_once()
        mock_session.list_tools.assert_awaited_once()

    async def test_missing_mcp_tools(self):
        """MCP health check identifies missing required tools."""

        mock_stdio = AsyncMock()
        mock_stdio.__aenter__.return_value = ("read", "write")

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session

        mock_session.list_tools.return_value = SimpleNamespace(
            tools=[
                SimpleNamespace(name="get_claim_status"),
                SimpleNamespace(name="validate_claim_referral"),
            ]
        )

        with (
            patch("health_checks.stdio_client", return_value=mock_stdio),
            patch("health_checks.ClientSession", return_value=mock_session),
        ):
            result = await check_mcp_health()

        self.assertEqual(result["service"], "mcp")
        self.assertEqual(result["status"], "TOOLS_MISSING")
        self.assertEqual(result["tool_count"], 2)

        self.assertEqual(
            result["missing_tools"],
            [
                "get_claim_amount",
                "get_claim_medical_evidence",
                "list_open_claims",
                "search_medical_documents",
            ],
        )

    async def test_mcp_server_unavailable(self):
        """MCP health check safely handles connection failures."""

        mock_stdio = AsyncMock()
        mock_stdio.__aenter__.side_effect = ConnectionError(
            "SIMULATED_MCP_CONNECTION_FAILURE"
        )

        with patch(
            "health_checks.stdio_client",
            return_value=mock_stdio,
        ):
            result = await check_mcp_health()

        self.assertEqual(result["service"], "mcp")
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["error_code"], "ConnectionError")

        # Sensitive exception messages must not appear in health results.
        self.assertNotIn(
            "SIMULATED_MCP_CONNECTION_FAILURE",
            str(result),
        )


if __name__ == "__main__":
    unittest.main()