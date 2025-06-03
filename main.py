# backend.py

import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

app = FastAPI()
model = init_chat_model(model='gpt-4.1', api_key=OPENAI_API_KEY)


async def get_agent():
    client = MultiServerMCPClient(
        {
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "sse"
            }
        }
    )
    tools = await client.get_tools()
    return create_react_agent(
        model=model,
        tools=tools,
        prompt="You are a helpful assistant"
    )


async def stream_chat_response(query: str):
    agent = await get_agent()
    async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode=["messages"]
    ):
        print(chunk)
        print('\n')
        # Each chunk looks like: ("messages", (AIMessageChunk(...), metadata_dict))
        if isinstance(chunk, tuple) and len(chunk) == 2:
            tag, inner = chunk
            if isinstance(inner, tuple) and len(inner) == 2:
                message_obj, metadata = inner  # message_obj is AIMessageChunk

                # Determine event type
                if (hasattr(message_obj, "additional_kwargs") and
                        message_obj.additional_kwargs and
                        message_obj.additional_kwargs.get("tool_calls")):
                    event_type = "tool_call"
                elif getattr(message_obj, "name", None):
                    event_type = "tool_response"
                else:
                    event_type = "assistant"

                # Build a plain-dict representation
                message_dict = {
                    "type": "message",
                    "event_type": event_type,
                    "content": getattr(message_obj, "content", None),
                    "name": getattr(message_obj, "name", None),
                    "additional_kwargs": dict(getattr(message_obj, "additional_kwargs", {})),
                    "response_metadata": dict(getattr(message_obj, "response_metadata", {})),
                    "id": getattr(message_obj, "id", None),
                    "metadata": metadata
                }
                yield f"data: {json.dumps(message_dict)}\n\n"


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")
    return StreamingResponse(
        stream_chat_response(query),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
