from ollama import AsyncClient
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def check_ollama_health(
    model_name: str,
    timeout: float = 120.0,
) -> dict:
    """Check Ollama connectivity and configured model availability."""

    client = AsyncClient(timeout=timeout)

    try:
        response = await client.list()
    except Exception as exc:
        return {
            "service": "ollama",
            "status": "UNAVAILABLE",
            "error_code": type(exc).__name__,
        }

    installed_models = {model.model for model in response.models}

    return {
        "service": "ollama",
        "status": ("HEALTHY" if model_name in installed_models else "MODEL_NOT_FOUND"),
        "model": model_name,
    }


async def check_mcp_health() -> dict:
    """Check MCP server connectivity and required tool availability."""

    server_path = Path(__file__).with_name("rag_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()

        tool_names = {tool.name for tool in response.tools}

        required_tools = {
            "get_claim_status",
            "get_claim_amount",
            "list_open_claims",
            "get_claim_medical_evidence",
            "search_medical_documents",
            "validate_claim_referral",
        }

        missing_tools = sorted(required_tools - tool_names)

        return {
            "service": "mcp",
            "status": "HEALTHY" if not missing_tools else "TOOLS_MISSING",
            "tool_count": len(tool_names),
            "missing_tools": missing_tools,
        }

    except Exception as exc:
        return {
            "service": "mcp",
            "status": "UNAVAILABLE",
            "error_code": type(exc).__name__,
        }


async def check_system_health(
    model_name: str,
    timeout: float = 120.0,
) -> dict:
    """Return the combined Ollama and MCP system health."""

    ollama_result = await check_ollama_health(
        model_name=model_name,
        timeout=timeout,
    )

    mcp_result = await check_mcp_health()

    return {
        "status": (
            "HEALTHY"
            if (
                ollama_result["status"] == "HEALTHY"
                and mcp_result["status"] == "HEALTHY"
            )
            else "DEGRADED"
        ),
        "services": {
            "ollama": ollama_result,
            "mcp": mcp_result,
        },
    }


if __name__ == "__main__":
    import asyncio
    import json

    from langgraph_rag_agent import MODEL, OLLAMA_TIMEOUT

    async def main():
        result = await check_system_health(
            model_name=MODEL,
            timeout=OLLAMA_TIMEOUT,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "HEALTHY" else 1

    raise SystemExit(asyncio.run(main()))
