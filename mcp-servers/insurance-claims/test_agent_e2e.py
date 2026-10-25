import asyncio
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

from langgraph_rag_agent import (
    SYSTEM_PROMPT,
    create_agent,
    get_referral_validation_report,
)


async def test_clm001():
    server_path = Path(__file__).with_name("rag_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            discovered = await session.list_tools()

            graph = create_agent(
                session,
                AsyncClient(),
                discovered.tools,
            )

            result = await graph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(
                            content="Validate the specialist referral for CLM001."
                        ),
                    ]
                },
                config={"recursion_limit": 13},
            )

            validation_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
                and message.name == "validate_claim_referral"
            ]

            assert validation_messages, "Agent did not call validate_claim_referral"

            payload = json.loads(validation_messages[-1].content)

            assert payload["success"], "MCP validation failed"
            assert payload["arguments"]["claim_id"] == "CLM001"

            validation = json.loads(payload["result"])

            assert validation["overall_finding"] == "PASS"

            report = get_referral_validation_report(result["messages"])

            assert report is not None
            assert "Overall finding: PASS" in report
            assert "RIGHT knee" in report
            assert "Dr. A. Malik" in report
            assert "Conclusion: The available evidence supports" in report

            print("PASS: CLM001 end-to-end agent test")


async def test_clm002():
    server_path = Path(__file__).with_name("rag_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            discovered = await session.list_tools()

            graph = create_agent(
                session,
                AsyncClient(),
                discovered.tools,
            )

            result = await graph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(
                            content="Validate the specialist referral for CLM002."
                        ),
                    ]
                },
                config={"recursion_limit": 13},
            )

            validation_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
                and message.name == "validate_claim_referral"
            ]

            assert (
                validation_messages
            ), "Agent did not call validate_claim_referral for CLM002"

            payload = json.loads(validation_messages[-1].content)

            assert payload["success"], "CLM002 MCP validation failed"
            assert payload["arguments"]["claim_id"] == "CLM002"

            validation = json.loads(payload["result"])

            assert validation["overall_finding"] == "REVIEW"

            expected_checks = {
                "referral_presence": "PASS",
                "referral_date": "PASS",
                "anatomy": "REVIEW",
                "specialist": "PASS",
            }

            for check, expected in expected_checks.items():
                assert validation["checks"][check]["finding"] == expected

            assert (
                validation["checks"]["anatomy"]["referral_anatomy"] == "LEFT shoulder"
            )
            assert not validation["checks"]["anatomy"]["treatment_anatomy"]

            report = get_referral_validation_report(result["messages"])

            assert report is not None
            assert "Overall finding: REVIEW" in report
            assert "Body part information is missing" in report
            assert (
                "Conclusion: The available evidence requires manual review." in report
            )

            print("PASS: CLM002 end-to-end agent test")


async def test_clm003():
    server_path = Path(__file__).with_name("rag_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            discovered = await session.list_tools()

            graph = create_agent(
                session,
                AsyncClient(),
                discovered.tools,
            )

            result = await graph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(
                            content="Validate the specialist referral for CLM003."
                        ),
                    ]
                },
                config={"recursion_limit": 13},
            )

            validation_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
                and message.name == "validate_claim_referral"
            ]

            assert (
                validation_messages
            ), "Agent did not call validate_claim_referral for CLM003"

            payload = json.loads(validation_messages[-1].content)

            assert payload["success"], "CLM003 MCP validation failed"
            assert payload["arguments"]["claim_id"] == "CLM003"

            validation = json.loads(payload["result"])

            assert validation["overall_finding"] == "REVIEW"

            for check in (
                "referral_presence",
                "referral_date",
                "anatomy",
                "specialist",
            ):
                assert (
                    validation["checks"][check]["finding"] == "REVIEW"
                ), f"Unexpected finding for {check}"

            explanation = validation["checks"]["referral_presence"]["explanation"]

            assert "does not prove" in explanation.lower()

            report = get_referral_validation_report(result["messages"])

            assert report is not None
            assert "Claim: CLM003" in report
            assert "Overall finding: REVIEW" in report
            assert "No referral document found" in report
            assert "does not prove" in report.lower()
            assert (
                "Conclusion: The available evidence requires manual review." in report
            )

            print("PASS: CLM003 end-to-end agent test")


async def test_clm004():
    server_path = Path(__file__).with_name("rag_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            discovered = await session.list_tools()

            graph = create_agent(
                session,
                AsyncClient(),
                discovered.tools,
            )

            result = await graph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(
                            content="Validate the specialist referral for CLM004."
                        ),
                    ]
                },
                config={"recursion_limit": 13},
            )

            validation_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
                and message.name == "validate_claim_referral"
            ]

            assert (
                validation_messages
            ), "Agent did not call validate_claim_referral for CLM004"

            payload = json.loads(validation_messages[-1].content)

            assert payload["success"], "CLM004 MCP validation failed"
            assert payload["arguments"]["claim_id"] == "CLM004"

            validation = json.loads(payload["result"])

            assert validation["overall_finding"] == "REVIEW"

            expected_checks = {
                "referral_presence": "PASS",
                "referral_date": "PASS",
                "anatomy": "REVIEW",
                "specialist": "PASS",
            }

            for check, expected in expected_checks.items():
                actual = validation["checks"][check]["finding"]
                assert (
                    actual == expected
                ), f"CLM004 {check}: expected {expected}, got {actual}"

            anatomy = validation["checks"]["anatomy"]

            assert anatomy["referral_anatomy"] == "LEFT ankle"
            assert anatomy["treatment_anatomy"] == "RIGHT ankle"

            report = get_referral_validation_report(result["messages"])

            assert report is not None
            assert "Claim: CLM004" in report
            assert "Overall finding: REVIEW" in report
            assert (
                "referral specifies LEFT ankle but treatment specifies RIGHT ankle"
                in report
            )
            assert "CLM004-DOC02" in report
            assert "CLM004-DOC03" in report
            assert (
                "Conclusion: The available evidence requires manual review." in report
            )

            print("PASS: CLM004 end-to-end agent test")


async def test_invalid_claim_id():
    server_path = Path(__file__).with_name("rag_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            discovered = await session.list_tools()

            graph = create_agent(
                session,
                AsyncClient(),
                discovered.tools,
            )

            result = await graph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(
                            content="Validate the specialist referral for ABC123."
                        ),
                    ]
                },
                config={"recursion_limit": 13},
            )

            validation_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
                and message.name == "validate_claim_referral"
            ]

            assert validation_messages, "Agent did not attempt referral validation"

            payload = json.loads(validation_messages[-1].content)

            assert payload["success"] is False
            assert payload["arguments"]["claim_id"] == "ABC123"
            assert "Invalid claim ID format" in payload["error"]

            # The MCP tool must not have executed successfully.
            assert "result" not in payload

            report = get_referral_validation_report(result["messages"])

            assert report is not None
            assert "Claim: ABC123" in report
            assert "Status: INVALID_CLAIM_ID" in report
            assert "Expected CLM followed by 3 digits" in report
            assert "Conclusion: No referral validation was performed." in report

            print("PASS: ABC123 invalid-claim-ID end-to-end test")


async def test_unknown_claim_id():
    server_path = Path(__file__).with_name("rag_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            discovered = await session.list_tools()

            graph = create_agent(
                session,
                AsyncClient(),
                discovered.tools,
            )

            result = await graph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(
                            content="Validate the specialist referral for CLM999."
                        ),
                    ]
                },
                config={"recursion_limit": 13},
            )

            validation_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
                and message.name == "validate_claim_referral"
            ]

            assert (
                validation_messages
            ), "Agent did not call validate_claim_referral for CLM999"

            payload = json.loads(validation_messages[-1].content)

            assert payload["success"]
            assert payload["arguments"]["claim_id"] == "CLM999"

            validation = json.loads(payload["result"])

            assert validation["error"] == "CLAIM_NOT_FOUND"
            assert "overall_finding" not in validation
            assert "checks" not in validation

            report = get_referral_validation_report(result["messages"])

            assert report is not None
            assert "Claim: CLM999" in report
            assert "Status: CLAIM_NOT_FOUND" in report
            assert "Conclusion: No referral validation was performed." in report

            print("PASS: CLM999 unknown-claim end-to-end test")


async def test_missing_claim_id():
    server_path = Path(__file__).with_name("rag_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            discovered = await session.list_tools()

            graph = create_agent(
                session,
                AsyncClient(),
                discovered.tools,
            )

            result = await graph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content="Validate the specialist referral."),
                    ]
                },
                config={"recursion_limit": 13},
            )

            # No MCP tools should be called.
            tool_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
            ]

            assert not tool_messages, "MCP tool was called without an explicit claim ID"

            response = result["messages"][-1].content

            assert "Please provide a claim ID" in response
            assert "CLM001" in response

            print("PASS: Missing claim ID end-to-end test")


async def main():
    await test_clm001()
    await test_clm002()
    await test_clm003()
    await test_clm004()
    await test_invalid_claim_id()
    await test_unknown_claim_id()
    await test_missing_claim_id()


if __name__ == "__main__":
    asyncio.run(main())
