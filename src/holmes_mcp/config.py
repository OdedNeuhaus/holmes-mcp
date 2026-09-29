"""Server configuration, read from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOLMESGPT_URL: str = Field(
        default="http://holmesgpt-holmes",
        description="Base URL of the HolmesGPT server",
    )
    GENERIC_MODEL_NAME: str = Field(
        default="generic",
        description="HolmesGPT model name sent with every request",
    )
    ADDITIONAL_SYSTEM_PROMPT: str = Field(
        default="",
        description=(
            "Extra system prompt appended to HolmesGPT's own. Useful for facts "
            "the model cannot discover via tools."
        ),
    )
    STALL_TIMEOUT_SECONDS: float = Field(
        default=120,
        description=(
            "Max seconds of silence between SSE chunks before aborting. Must be "
            "longer than the slowest single HolmesGPT tool call, since HolmesGPT "
            "goes quiet between start_tool_calling and tool_calling_result."
        ),
    )
    TOTAL_TIMEOUT_SECONDS: float = Field(
        default=1800,
        description="Overall ceiling for one HolmesGPT request, in seconds.",
    )
    HEARTBEAT_SECONDS: float = Field(
        default=30,
        description=(
            "Interval of 'still investigating' progress notifications. Claude "
            "Code aborts an HTTP MCP call after 5 minutes without a response "
            "or progress notification, so this must stay well below that."
        ),
    )
    MAX_INPUT_CHARS: int = Field(
        default=30000,
        description=(
            "Largest question (+ context) accepted. Bigger inputs are rejected "
            "with a message asking Claude to include only the relevant parts."
        ),
    )
    REDIS_URL: str | None = Field(
        default=None,
        description=(
            "Redis URL for the conversation store, e.g. "
            "redis://:password@redis:6379/0. Required when running more than "
            "one replica. Unset = in-memory store (single replica / local dev)."
        ),
    )
    REDIS_KEY_PREFIX: str = Field(
        default="holmes-mcp:conv:",
        description="Prefix for conversation keys in Redis.",
    )
    CONVERSATION_TTL_SECONDS: int = Field(
        default=7200,
        description="How long an idle conversation stays available for follow-ups.",
    )
    MAX_CONVERSATIONS: int = Field(
        default=500,
        description=(
            "In-memory store only: max conversations kept; least recently "
            "used are evicted. Redis relies on the TTL instead."
        ),
    )
    MAX_HISTORY_CHARS: int = Field(
        default=1_000_000,
        description=(
            "Size cap for a stored conversation. Above it, the largest tool "
            "outputs in the history are shortened. Keeps Redis values and the "
            "follow-up requests sent to HolmesGPT bounded."
        ),
    )
    MAX_RESPONSE_CHARS: int = Field(
        default=60000,
        description=(
            "Cap on the text returned to the MCP client. Claude Code rejects "
            "tool results above its MCP output token limit."
        ),
    )
    USER_HEADER: str = Field(
        default="X-User-Email",
        description=(
            "Request header carrying the caller's identity. Used only for "
            "Langfuse attribution, never for access control."
        ),
    )
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
