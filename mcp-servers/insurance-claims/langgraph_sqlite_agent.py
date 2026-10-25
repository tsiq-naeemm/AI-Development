import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

MODEL = "llama3.2"
MAX_MODEL_CALLS = 6

SYSTEM_PROMPT = """
You are an insurance claims assistant working with
a local synthetic insurance claims database.

Rules:
1. Use MCP tools to retrieve claim information.
2. Tool results are authoritative for this dataset.
3. Never invent claim data.
4. Reuse information already returned by tools.
5. Never call a tool again for information already available.
6. A list_open_claims result includes claim IDs, statuses,
   and amounts. You can compare these amounts directly.
7. Only use actual claim IDs such as CLM001 or CLM003.
8. If a claim is not found, explain that clearly.
9. Once sufficient information is available, answer
   without requesting more tools.
"""


def to_ollama_messages(messages):
    """Convert LangChain messages into Ollama-compatible dicts."""
    converted = []

    for message in messages:
        if isinstance(message, SystemMessage):
            converted.append(
                {
                    "role": "system",
                    "content": message.content,
                }
            )

        elif isinstance(message, HumanMessage):
            converted.append(
                {
                    "role": "user",
                    "content": message.content,
                }
            )

        elif isinstance(message, AIMessage):
            item = {
                "role": "assistant",
                "content": message.content or "",
            }

            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "function": {
                            "name": call["name"],
                            "arguments": call["args"],
                        }
                    }
                    for call in message.tool_calls
                ]

            converted.append(item)

        elif isinstance(message, ToolMessage):
            converted.append(
                {
                    "role": "tool",
                    "tool_name": message.name,
                    "content": message.content,
                }
            )

    return converted


def create_agent(session, ollama, discovered_tools):
    """Construct the LangGraph workflow."""

    tool_schemas = {tool.name: tool.input_schema for tool in discovered_tools}

    ollama_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in discovered_tools
    ]

    async def agent_node(state: MessagesState):
        """Ask Ollama to answer or select tools."""

        print("\n[LangGraph] Agent node")

        response = await ollama.chat(
            model=MODEL,
            messages=to_ollama_messages(state["messages"]),
            tools=ollama_tools,
        )

        calls = []

        for index, call in enumerate(response.message.tool_calls or []):
            calls.append(
                {
                    "name": call.function.name,
                    "args": call.function.arguments,
                    "id": f"call_{index}_{len(state['messages'])}",
                    "type": "tool_call",
                }
            )

        message = AIMessage(
            content=response.message.content or "",
            tool_calls=calls,
        )

        if calls:
            print(
                "[LangGraph] Requested:",
                [call["name"] for call in calls],
            )
        else:
            print("[LangGraph] Final answer generated")

        return {"messages": [message]}

    async def tool_node(state: MessagesState):
        """Validate, deduplicate and execute MCP calls."""

        print("[LangGraph] MCP tool node")

        last_message = state["messages"][-1]

        # Reconstruct results from previous tool messages.
        # This cache is scoped to the current question.
        cache = {}

        for previous in state["messages"]:
            if not isinstance(previous, ToolMessage):
                continue

            try:
                payload = json.loads(previous.content)
                key = (
                    payload["tool"],
                    json.dumps(
                        payload["arguments"],
                        sort_keys=True,
                    ),
                )
                if payload.get("success"):
                    cache[key] = payload
            except (ValueError, KeyError, TypeError):
                continue

        results = []

        for call in last_message.tool_calls:
            name = call["name"]
            arguments = call["args"]
            call_id = call["id"]

            print(f"[MCP] Requested: {name} {arguments}")

            payload: dict[str, Any] = {
                "tool": name,
                "arguments": arguments,
                "success": False,
            }

            if name not in tool_schemas:
                payload["error"] = "Unknown MCP tool"

            else:
                try:
                    validate(
                        instance=arguments,
                        schema=tool_schemas[name],
                    )

                    # Domain validation for this prototype:
                    # claim IDs must follow CLM + 3 digits.
                    claim_id = arguments.get("claim_id")

                    if claim_id is not None:
                        if (
                            not isinstance(claim_id, str)
                            or len(claim_id) != 6
                            or not claim_id.startswith("CLM")
                            or not claim_id[3:].isdigit()
                        ):
                            raise ValueError(
                                "Invalid claim ID format. "
                                "Expected CLM followed by 3 digits."
                            )

                    key = (
                        name,
                        json.dumps(arguments, sort_keys=True),
                    )

                    if key in cache:
                        print("[MCP] Reusing cached result")
                        payload = cache[key]

                    else:
                        result = await session.call_tool(
                            name,
                            arguments=arguments,
                        )

                        content = "\n".join(
                            item.text for item in result.content if item.type == "text"
                        )

                        payload.update(
                            {
                                "result": content,
                                "success": not result.is_error,
                            }
                        )

                        if payload["success"]:
                            cache[key] = payload

                        print(f"[MCP] Result: {content}")

                except (ValidationError, ValueError) as exc:
                    payload["error"] = str(exc)
                    print(f"[MCP] Validation error: {exc}")

                except Exception as exc:
                    payload["error"] = str(exc)
                    print(f"[MCP] Execution error: {exc}")

            results.append(
                ToolMessage(
                    content=json.dumps(payload),
                    name=name,
                    tool_call_id=call_id,
                )
            )

        return {"messages": results}

    def route_after_agent(state: MessagesState):
        """Choose between executing tools and ending."""

        last_message = state["messages"][-1]

        if last_message.tool_calls:
            return "tools"

        return END

    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END},
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


async def main():
    server_path = Path(__file__).with_name("sqlite_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            discovered = await session.list_tools()
            ollama = AsyncClient()

            graph = create_agent(
                session,
                ollama,
                discovered.tools,
            )

            print("=" * 55)
            print(" LOCAL LANGGRAPH INSURANCE AGENT")
            print("=" * 55)
            print(f"Model: {MODEL}")
            print(f"MCP tools: {len(discovered.tools)}")
            print("Type 'exit' or 'quit' to stop.\n")

            while True:
                try:
                    question = await asyncio.to_thread(input, "You: ")

                    if question.strip().lower() in ("exit", "quit"):
                        print("Agent stopped.")
                        break

                    if not question.strip():
                        continue

                    initial_state = {
                        "messages": [
                            SystemMessage(content=SYSTEM_PROMPT),
                            HumanMessage(content=question),
                        ]
                    }

                    result = await graph.ainvoke(
                        initial_state,
                        config={"recursion_limit": 13},
                    )

                    final_message = result["messages"][-1]

                    print(
                        "\nAgent:",
                        final_message.content,
                    )
                    print()

                except (KeyboardInterrupt, EOFError):
                    print("\nAgent stopped.")
                    break

                except Exception as exc:
                    print(f"\nAgent error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
