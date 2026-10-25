import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

MODEL = "llama3.2"

SYSTEM_PROMPT = """
You are an insurance claims assistant.

You have access to a local synthetic insurance claims database
through MCP tools.

Rules:
1. Use MCP tools when a question requires claim information.
2. Treat successful tool results as authoritative for this dataset.
3. Never invent claim information.
4. Do not claim that you lack access to information already
   returned by a tool.
5. Answer clearly and concisely.
"""


async def process_question(
    question: str,
    session: ClientSession,
    ollama: AsyncClient,
    tools: list,
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    response = await ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools,
    )

    messages.append(response.message)

    if not response.message.tool_calls:
        print("\nAgent:", response.message.content)
        return

    for tool_call in response.message.tool_calls:
        name = tool_call.function.name
        arguments = tool_call.function.arguments

        print(f"\n[MCP] Calling: {name}")
        print(f"[MCP] Arguments: {arguments}")

        result = await session.call_tool(
            name,
            arguments=arguments,
        )

        content = "\n".join(item.text for item in result.content if item.type == "text")

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

    final_response = await ollama.chat(
        model=MODEL,
        messages=messages,
    )

    print("\nAgent:", final_response.message.content)


async def main():
    server_path = Path(__file__).with_name("server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

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

            ollama = AsyncClient()

            print("=" * 55)
            print(" LOCAL INSURANCE CLAIMS AI AGENT")
            print("=" * 55)
            print(f"Model: {MODEL}")
            print(f"MCP tools discovered: {len(tools)}")
            print("Type 'exit' or 'quit' to stop.\n")

            while True:
                try:
                    question = await asyncio.to_thread(input, "You: ")

                    if question.strip().lower() in ("exit", "quit"):
                        print("Agent stopped.")
                        break

                    if not question.strip():
                        continue

                    await process_question(
                        question,
                        session,
                        ollama,
                        tools,
                    )

                    print()

                except (KeyboardInterrupt, EOFError):
                    print("\nAgent stopped.")
                    break

                except Exception as exc:
                    print(f"\nError: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
