
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_path = Path(__file__).with_name("server.py")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # Step 1: Initialize connection
            await session.initialize()
            print("Connected to InsuranceClaims MCP Server")

            # Step 2: Discover available tools
            tools = await session.list_tools()

            print("\nAvailable MCP Tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Step 3: Execute claim status tool
            result = await session.call_tool(
                "get_claim_status",
                arguments={"claim_id": "CLM001"},
            )
            print("\nClaim Status:", result.content)

            # Step 4: Execute claim amount tool
            result = await session.call_tool(
                "get_claim_amount",
                arguments={"claim_id": "CLM003"},
            )
            print("\nClaim Amount:", result.content)

            # Step 5: Execute open claims tool
            result = await session.call_tool(
                "list_open_claims",
                arguments={},
            )
            print("\nOpen Claims:", result.content)


if __name__ == "__main__":
    asyncio.run(main())
