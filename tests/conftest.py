import json

import pytest

from holmes_mcp.config import Settings

HOLMES_URL = "http://holmes.test"
CHAT_URL = f"{HOLMES_URL}/api/chat"


def sse(*events: tuple[str, object]) -> bytes:
    """Build an SSE body the way HolmesGPT's /api/chat streams it."""
    chunks = []
    for event, data in events:
        payload = data if isinstance(data, str) else json.dumps(data)
        chunks.append(f"event: {event}\ndata: {payload}\n\n")
    return "".join(chunks).encode()


def tool_events(tool_call_id: str, name: str, params: dict, data, status="success"):
    return [
        ("start_tool_calling", {"tool_name": name, "id": tool_call_id}),
        (
            "tool_calling_result",
            {
                "tool_call_id": tool_call_id,
                "name": name,
                "result": {"status": status, "params": params, "data": data},
            },
        ),
    ]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        HOLMESGPT_URL=HOLMES_URL,
        GENERIC_MODEL_NAME="generic",
        STALL_TIMEOUT_SECONDS=5,
        TOTAL_TIMEOUT_SECONDS=30,
        REDIS_URL=None,
        # Never trace to a real Langfuse from tests, whatever the shell has.
        LANGFUSE_HOST=None,
        LANGFUSE_PUBLIC_KEY=None,
        LANGFUSE_SECRET_KEY=None,
    )
