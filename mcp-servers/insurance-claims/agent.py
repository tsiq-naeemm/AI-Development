import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

MODEL = "llama3.2"


async def main():
    server_path = Path(__file__).with_name("server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP Server")

            # Discover tools from the actual MCP server
            discovered = await session.list_tools()

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
                for tool in discovered.tools
            ]

            print(f"Discovered {len(tools)} MCP tools")

            question = "What is the status of claim CLM001?"
            print(f"\nQuestion: {question}")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an insurance claims assistant. "
                        "You have access to a local synthetic insurance claims "
                        "database through MCP tools. "
                        "Tool results are the authoritative source for claim data. "
                        "When a tool returns a claim status, use that result "
                        "to answer the user's question. "
                        "Do not claim that you lack access to information "
                        "that has already been returned by a tool. "
                        "Never invent claim information."
                    ),
                },
                {"role": "user", "content": question},
            ]

            ollama = AsyncClient()

            # Ask LLM to select a tool
            response = await ollama.chat(
                model=MODEL,
                messages=messages,
                tools=tools,
            )

            messages.append(response.message)

            if not response.message.tool_calls:
                print("\nAnswer:", response.message.content)
                return

            # Execute the requested MCP tools
            for tool_call in response.message.tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments

                print(f"\nLLM selected: {name}")
                print(f"Arguments: {arguments}")

                result = await session.call_tool(
                    name,
                    arguments=arguments,
                )
                print("\n--- MCP TOOL DEBUG ---")
                print("Tool:", name)
                print("Is error:", result.is_error)
                print("Raw content:", result.content)
                print("Structured content:", result.structured_content)
                print("----------------------\n")

                if result.is_error:
                    print(f"MCP tool error: {result.content}")

                # Convert MCP response to readable text
                content = "\n".join(
                    item.text for item in result.content if item.type == "text"
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(
                        {
                            "claim_id": arguments.get("claim_id"),
                            "tool": name,
                            "result": content,
                            "source": "Local synthetic insurance claims database",
                            "success": not result.is_error,
                        }
                    ),
                }
            )

            print("\n--- MESSAGES SENT TO OLLAMA ---")
            for message in messages:
                print(message)
            print("--------------------------------\n")

            # Ask LLM to formulate the final answer
            final_response = await ollama.chat(
                model=MODEL,
                messages=messages,
            )

            print("\nFinal Answer:")
            print(final_response.message.content)


if __name__ == "__main__":
    asyncio.run(main())
