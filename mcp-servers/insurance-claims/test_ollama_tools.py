from ollama import chat

response = chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is the status of claim CLM001?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_claim_status",
                "description": "Get the status of an insurance claim.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim_id": {
                            "type": "string",
                            "description": "Insurance claim identifier",
                        }
                    },
                    "required": ["claim_id"],
                },
            },
        }
    ],
)

print("Tool calls:", response.message.tool_calls)
print("Model response:", response.message.content)
