import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

MODEL = "llama3.2"
MAX_ITERATIONS = 6

SYSTEM_PROMPT = """
You are an insurance claims assistant.

You have access to a local synthetic insurance claims
database through MCP tools.

Rules:
1. Use MCP tools whenever claim information is required.
2. Successful tool results are authoritative for this dataset.
3. Never invent claim information.
4. Never claim that you lack access to information
   already returned by a tool.
5. You may call multiple tools to answer a question.
6. After receiving tool results, decide whether more
   tools are required.
7. When sufficient information is available, provide
   a concise final answer.
8. Do not call tools repeatedly for information
   you have already retrieved.
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

    available_tools = {tool["function"]["name"] for tool in tools}

    tool_calls_count = 0

    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n[Agent] Reasoning iteration {iteration}")

        response = await ollama.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        messages.append(response.message)

        if not response.message.tool_calls:
            print("\nAgent:", response.message.content)
            print(f"\n[Agent] Total MCP calls: " f"{tool_calls_count}")
            return

        for tool_call in response.message.tool_calls:

            name = tool_call.function.name
            arguments = tool_call.function.arguments

            if name not in available_tools:
                print(f"[Agent] Unknown tool: {name}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps({"error": "Unknown MCP tool"}),
                    }
                )
                continue

            print(f"[MCP] Calling: {name}")
            print(f"[MCP] Arguments: {arguments}")

            try:
                result = await session.call_tool(
                    name,
                    arguments=arguments,
                )

                tool_calls_count += 1

                content = "\n".join(
                    item.text for item in result.content if item.type == "text"
                )

                tool_response = {
                    "tool": name,
                    "arguments": arguments,
                    "result": content,
                    "source": ("Local synthetic insurance claims database"),
                    "success": not result.is_error,
                }

                print(f"[MCP] Result: {content}")

            except Exception as exc:
                tool_response = {
                    "tool": name,
                    "arguments": arguments,
                    "success": False,
                    "error": str(exc),
                }

                print(f"[MCP] Error: {exc}")

            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(tool_response),
                }
            )

    print(f"\n[Agent] Maximum reasoning iterations " f"({MAX_ITERATIONS}) reached.")
    print("[Agent] Stopping to prevent an infinite loop.")


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
            print(" LOCAL MULTI-TOOL INSURANCE AGENT")
            print("=" * 55)
            print(f"Model: {MODEL}")
            print(f"Available MCP tools: {len(tools)}")
            print(f"Maximum iterations: {MAX_ITERATIONS}")
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
                    print(f"\nAgent error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
