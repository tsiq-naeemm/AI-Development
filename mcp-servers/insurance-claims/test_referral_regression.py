
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_path = Path(__file__).with_name("rag_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "validate_claim_referral",
                arguments={"claim_id": "CLM001"},
            )

            if result.is_error:
                raise RuntimeError("MCP validation tool failed")

            content = "\n".join(
                item.text
                for item in result.content
                if item.type == "text"
            )

            data = json.loads(content)

            assert data["claim_id"] == "CLM001"
            assert data["overall_finding"] == "PASS"

            for check in (
                "referral_presence",
                "referral_date",
                "anatomy",
                "specialist",
            ):
                assert data["checks"][check]["finding"] == "PASS", (
                    f"Unexpected result for {check}"
                )

            print("PASS: CLM001 regression test")

            # CLM002: Referral exists, but treatment anatomy is missing.
            result = await session.call_tool(
                "validate_claim_referral",
                arguments={"claim_id": "CLM002"},
            )

            if result.is_error:
                raise RuntimeError("CLM002 validation tool failed")

            content = "\n".join(
                item.text
                for item in result.content
                if item.type == "text"
            )

            data = json.loads(content)

            assert data["claim_id"] == "CLM002"
            assert data["overall_finding"] == "REVIEW"

            expected_checks = {
                "referral_presence": "PASS",
                "referral_date": "PASS",
                "anatomy": "REVIEW",
                "specialist": "PASS",
            }

            for check, expected in expected_checks.items():
                actual = data["checks"][check]["finding"]
                assert actual == expected, (
                    f"CLM002 {check}: expected {expected}, got {actual}"
                )

            assert data["checks"]["anatomy"]["referral_anatomy"] == "LEFT shoulder"
            assert not data["checks"]["anatomy"]["treatment_anatomy"]

            print("PASS: CLM002 regression test")


            # CLM003: No referral found in available evidence.
            result = await session.call_tool(
                "validate_claim_referral",
                arguments={"claim_id": "CLM003"},
            )

            if result.is_error:
                raise RuntimeError("CLM003 validation tool failed")

            content = "\n".join(
                item.text
                for item in result.content
                if item.type == "text"
            )

            data = json.loads(content)

            assert data["claim_id"] == "CLM003"
            assert data["overall_finding"] == "REVIEW"

            expected_checks = {
                "referral_presence": "REVIEW",
                "referral_date": "REVIEW",
                "anatomy": "REVIEW",
                "specialist": "REVIEW",
            }

            for check, expected in expected_checks.items():
                actual = data["checks"][check]["finding"]
                assert actual == expected, (
                    f"CLM003 {check}: expected {expected}, got {actual}"
                )

            # Missing evidence must not be treated as proof
            # that a referral does not exist elsewhere.
            explanation = data["checks"]["referral_presence"]["explanation"]
            assert "does not prove" in explanation.lower()

            print("PASS: CLM003 regression test")


            # CLM004: Referral and treatment anatomy conflict.
            result = await session.call_tool(
                "validate_claim_referral",
                arguments={"claim_id": "CLM004"},
            )

            if result.is_error:
                raise RuntimeError("CLM004 validation tool failed")

            content = "\n".join(
                item.text
                for item in result.content
                if item.type == "text"
            )

            data = json.loads(content)

            assert data["claim_id"] == "CLM004"
            assert data["overall_finding"] == "REVIEW"

            expected_checks = {
                "referral_presence": "PASS",
                "referral_date": "PASS",
                "anatomy": "REVIEW",
                "specialist": "PASS",
            }

            for check, expected in expected_checks.items():
                actual = data["checks"][check]["finding"]
                assert actual == expected, (
                    f"CLM004 {check}: expected {expected}, got {actual}"
                )

            anatomy = data["checks"]["anatomy"]

            assert anatomy["referral_anatomy"] == "LEFT ankle"
            assert anatomy["treatment_anatomy"] == "RIGHT ankle"

            assert "CLM004-DOC02" in anatomy["source_document_ids"]
            assert "CLM004-DOC03" in anatomy["source_document_ids"]

            print("PASS: CLM004 regression test")


            # CLM999: Claim does not exist in SQLite.
            result = await session.call_tool(
                "validate_claim_referral",
                arguments={"claim_id": "CLM999"},
            )

            if result.is_error:
                raise RuntimeError("CLM999 MCP tool execution failed")

            content = "\n".join(
                item.text
                for item in result.content
                if item.type == "text"
            )

            data = json.loads(content)

            assert data["claim_id"] == "CLM999"
            assert data["error"] == "CLAIM_NOT_FOUND"
            assert "does not exist" in data["explanation"].lower()

            # A nonexistent claim must not receive
            # a medical referral validation finding.
            assert "overall_finding" not in data
            assert "checks" not in data

            print("PASS: CLM999 unknown-claim regression test")




if __name__ == "__main__":
    asyncio.run(main())
