import asyncio
import httpx
import unittest
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage
from langgraph_rag_agent import (
    create_agent,
    RECURSION_LIMIT,
    classify_agent_error,
)

class TestAgentReliability(unittest.IsolatedAsyncioTestCase):

    async def test_ollama_timeout_propagates(self):
        """A timeout from Ollama must propagate through LangGraph."""

        mock_ollama = AsyncMock()
        mock_ollama.chat.side_effect = TimeoutError(
            "SIMULATED_TIMEOUT"
        )

        graph = create_agent(
            session=None,
            ollama=mock_ollama,
            discovered_tools=[],
        )

        with self.assertRaises(TimeoutError):
            await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Explain the insurance claims process."
                        )
                    ]
                },
                config={"recursion_limit": RECURSION_LIMIT},
            )

        mock_ollama.chat.assert_awaited_once()

    async def test_ollama_connection_error_propagates(self):
        """A non-timeout Ollama failure preserves its exception type."""

        mock_ollama = AsyncMock()
        mock_ollama.chat.side_effect = ConnectionError(
            "SIMULATED_CONNECTION_FAILURE"
        )

        graph = create_agent(
            session=None,
            ollama=mock_ollama,
            discovered_tools=[],
        )

        with self.assertRaises(ConnectionError):
            await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Explain the insurance claims process."
                        )
                    ]
                },
                config={"recursion_limit": RECURSION_LIMIT},
            )

        mock_ollama.chat.assert_awaited_once()

    def test_error_classification(self):
        """Classify timeout and non-timeout failures correctly."""

        self.assertEqual(
            classify_agent_error(TimeoutError()),
            "OLLAMA_TIMEOUT",
        )

        self.assertEqual(
            classify_agent_error(asyncio.TimeoutError()),
            "OLLAMA_TIMEOUT",
        )

        self.assertEqual(
            classify_agent_error(ConnectionError()),
            "ConnectionError",
        )

        self.assertEqual(
            classify_agent_error(ValueError()),
            "ValueError",
        )

        self.assertEqual(
            classify_agent_error(httpx.ReadTimeout("Simulated timeout")),
            "OLLAMA_TIMEOUT",
        )

        self.assertEqual(
            classify_agent_error(httpx.ConnectTimeout("Simulated timeout")),
            "OLLAMA_TIMEOUT",
        )

        self.assertEqual(
            classify_agent_error(httpx.PoolTimeout("Simulated timeout")),
            "OLLAMA_TIMEOUT",
        )

        self.assertEqual(
            classify_agent_error(httpx.ConnectError("Simulated connection failure")),
            "ConnectError",
        )
    
    async def test_httpx_timeout_propagates(self):
        """An HTTPX read timeout propagates through LangGraph."""

        mock_ollama = AsyncMock()
        mock_ollama.chat.side_effect = httpx.ReadTimeout(
            "SIMULATED_HTTPX_TIMEOUT"
        )

        graph = create_agent(
            session=None,
            ollama=mock_ollama,
            discovered_tools=[],
        )

        with self.assertRaises(httpx.ReadTimeout) as caught:
            await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Explain the insurance claims process."
                        )
                    ]
                },
                config={"recursion_limit": RECURSION_LIMIT},
            )

        mock_ollama.chat.assert_awaited_once()

        self.assertEqual(
            classify_agent_error(caught.exception),
            "OLLAMA_TIMEOUT",
        )

if __name__ == "__main__":
    unittest.main()