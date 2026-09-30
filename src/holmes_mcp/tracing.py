"""
Langfuse tracing for holmes-mcp.

One trace per MCP tool call (ask_holmes / holmes_follow_up), grouped into a
Langfuse session per HolmesGPT conversation, with a child span for every tool
HolmesGPT runs. Sending is asynchronous (the SDK batches in the background)
and every call into the SDK is guarded: tracing can fail, answers must not.

Disabled (a no-op) unless LANGFUSE_HOST and both keys are configured.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from holmes_mcp import text
from holmes_mcp.config import Settings

logger = logging.getLogger(__name__)

TAGS = ["holmes-mcp", "claude-code"]

# How much of each HolmesGPT tool's output a tool span keeps.
TOOL_OUTPUT_CHARS = 2000

# Langfuse trace attributes must be US-ASCII and at most 200 characters.
_ATTR_MAX = 200


def _attr(value) -> str:
    return str(value).encode("ascii", "ignore").decode("ascii")[:_ATTR_MAX]


@dataclass
class TraceInfo:
    """What one traced call is about."""

    name: str  # the MCP tool: ask_holmes / holmes_follow_up
    trace_id: str  # 32 hex chars, also sent to HolmesGPT as metadata.trace_id
    session_id: str  # the conversation_id
    user_id: str
    turn: int
    input: dict
    # Small, filterable dimensions (client ip, hostname...). Values are
    # coerced to short strings and propagated to every span of the trace.
    metadata: dict[str, str] = field(default_factory=dict)


class ToolObserver:
    """Receives HolmesGPT tool events while a request streams."""

    def tool_started(self, tool_call_id: str, name: str) -> None:
        pass

    def tool_finished(
        self,
        tool_call_id: str,
        name: str,
        params: dict,
        status: str,
        output: str,
    ) -> None:
        pass


class ActiveTrace(ToolObserver):
    """No-op trace, used when tracing is disabled."""

    def finish(
        self,
        *,
        status: str,
        output: str | None = None,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        pass


class Tracer:
    """No-op tracer. `trace()` yields an ActiveTrace for one call."""

    enabled = False

    @contextmanager
    def trace(self, info: TraceInfo) -> Iterator[ActiveTrace]:
        yield ActiveTrace()

    def shutdown(self) -> None:
        pass


class _LangfuseTrace(ActiveTrace):
    def __init__(self, root):
        self._root = root
        self._tools: dict[str, object] = {}
        self._finished = False

    def tool_started(self, tool_call_id: str, name: str) -> None:
        try:
            span = self._root.start_observation(name=name or "tool", as_type="tool")
            if tool_call_id:
                self._tools[tool_call_id] = span
        except Exception:  # noqa: BLE001
            logger.debug("Langfuse tool span start failed", exc_info=True)

    def tool_finished(
        self,
        tool_call_id: str,
        name: str,
        params: dict,
        status: str,
        output: str,
    ) -> None:
        try:
            span = self._tools.pop(tool_call_id, None) if tool_call_id else None
            if span is None:
                # A result without a matching start event: record it anyway.
                span = self._root.start_observation(
                    name=name or "tool", as_type="tool"
                )
            failed = status.lower() in ("error", "failed", "failure")
            span.update(
                name=name or None,
                input=params,
                output=text.truncate(output, TOOL_OUTPUT_CHARS),
                metadata={"status": status} if status else None,
                level="ERROR" if failed else None,
                status_message=f"tool {status}" if failed else None,
            )
            span.end()
        except Exception:  # noqa: BLE001
            logger.debug("Langfuse tool span end failed", exc_info=True)

    def finish(
        self,
        *,
        status: str,
        output: str | None = None,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._finished = True
        try:
            # Tool calls still open were cut off by a timeout or cancellation.
            for span in self._tools.values():
                span.update(level="WARNING", status_message="did not finish")
                span.end()
            self._tools.clear()

            level = {"error": "ERROR", "cut_off": "WARNING", "cancelled": "WARNING"}
            self._root.update(
                output=output,
                metadata={"status": status, **(metadata or {})},
                level=level.get(status),
                status_message=error,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Langfuse trace finish failed", exc_info=True)


class LangfuseTracer(Tracer):
    enabled = True

    def __init__(self, client):
        # `client` is a langfuse.Langfuse instance.
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "LangfuseTracer":
        from langfuse import Langfuse, is_langfuse_span

        return cls(
            Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
                environment=settings.LANGFUSE_ENVIRONMENT or None,
                # Only our own spans. The MCP SDK has OpenTelemetry
                # instrumentation of its own, which would otherwise show up in
                # Langfuse as a second, empty trace per call.
                should_export_span=is_langfuse_span,
            )
        )

    @contextmanager
    def trace(self, info: TraceInfo) -> Iterator[ActiveTrace]:
        from langfuse import propagate_attributes

        try:
            root_cm = self.client.start_as_current_observation(
                trace_context={"trace_id": info.trace_id},
                name=info.name,
                as_type="agent",
                input=info.input,
            )
            root = root_cm.__enter__()
            attrs_cm = propagate_attributes(
                user_id=_attr(info.user_id),
                session_id=_attr(info.session_id),
                tags=TAGS,
                trace_name=_attr(info.name),
                metadata={
                    "turn": str(info.turn),
                    **{k: _attr(v) for k, v in info.metadata.items() if v},
                },
            )
            attrs_cm.__enter__()
        except Exception:  # noqa: BLE001 - never let tracing block a request
            logger.warning("Langfuse trace start failed", exc_info=True)
            yield ActiveTrace()
            return

        active = _LangfuseTrace(root)
        exc_info: tuple = (None, None, None)
        try:
            yield active
        except BaseException as exc:
            exc_info = (type(exc), exc, exc.__traceback__)
            if not active._finished:
                cancelled = type(exc).__name__ == "CancelledError"
                active.finish(
                    status="cancelled" if cancelled else "error",
                    error=(
                        "cancelled by the client"
                        if cancelled
                        else f"{type(exc).__name__}: {exc}"
                    ),
                )
            raise
        finally:
            try:
                attrs_cm.__exit__(*exc_info)
                root_cm.__exit__(*exc_info)
            except Exception:  # noqa: BLE001
                logger.debug("Langfuse trace close failed", exc_info=True)

    def shutdown(self) -> None:
        try:
            self.client.shutdown()
        except Exception:  # noqa: BLE001
            logger.warning("Langfuse shutdown failed", exc_info=True)


def create_tracer(settings: Settings) -> Tracer:
    if settings.LANGFUSE_HOST and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
        try:
            tracer = LangfuseTracer.from_settings(settings)
            logger.info("Langfuse tracing enabled: %s", settings.LANGFUSE_HOST)
            return tracer
        except Exception:  # noqa: BLE001
            logger.exception("Langfuse tracing could not start; continuing without it")
    elif settings.LANGFUSE_HOST or settings.LANGFUSE_PUBLIC_KEY or settings.LANGFUSE_SECRET_KEY:
        logger.warning(
            "Langfuse tracing is off: LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY must all be set"
        )
    return Tracer()
