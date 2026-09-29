"""
HolmesGPT /api/chat client.

Always streams, even though the MCP tool returns one final result: streaming
is what lets a stalled HolmesGPT (e.g. an unresponsive downstream tool) be
detected within STALL_TIMEOUT_SECONDS, and what lets progress be reported
while HolmesGPT runs its tools.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from holmes_mcp import text
from holmes_mcp.config import Settings

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]

# Characters of each tool's output kept for the "evidence" summary. The full
# output stays in the stored conversation_history for follow-ups.
TOOL_PREVIEW_CHARS = 300


@dataclass
class ToolCall:
    name: str
    params: dict
    status: str = ""
    preview: str = ""


@dataclass
class HolmesResult:
    answer: str = ""
    error: str | None = None
    # The stream ended without `ai_answer_end`; `answer` is what was salvaged.
    incomplete: bool = False
    tools_used: list[ToolCall] = field(default_factory=list)
    # Holmes's own history from `ai_answer_end`, including tool calls and
    # results. None when the event did not carry one.
    conversation_history: list | None = None
    duration_s: float = 0.0


def _stringify(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _extract_tool_call(event_data: dict) -> tuple[str, str, dict, str, str]:
    """
    Pull (tool_call_id, tool_name, params, status, result_text) out of a
    `tool_calling_result` event payload.
    """
    tool_call_id = event_data.get("tool_call_id") or event_data.get("id") or ""
    tool_name = event_data.get("tool_name") or event_data.get("name") or ""

    result = event_data.get("result") or {}
    if not isinstance(result, dict):
        result = {}

    params = result.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    status = str(result.get("status") or "")

    # `result.data` is already a JSON string when present; fall back to the
    # whole result for error responses.
    result_text = result.get("data")
    if result_text in (None, ""):
        result_text = result.get("error") or result
    return tool_call_id, tool_name, params, status, _stringify(result_text)


def _format_pending_approvals(pending_approvals: list) -> str:
    names = [
        a.get("tool_name") or a.get("name") or "tool"
        for a in pending_approvals or []
        if isinstance(a, dict)
    ]
    listed = ", ".join(names) if names else "unknown tools"
    return (
        "HolmesGPT paused to ask for approval to run: "
        f"{listed}. Tool approval is not supported through this MCP server. "
        "Rephrase the question so it does not need that action."
    )


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


@dataclass
class _StreamState:
    """Everything gathered so far, kept outside the stream loop so a timeout
    or cancellation can still report it."""

    result: HolmesResult = field(default_factory=HolmesResult)
    started: float = field(default_factory=time.monotonic)
    # Per-round accumulators for the two text channels. Both reset on
    # `start_tool_calling` so only the round after the final tool call - the
    # answer - survives.
    answer_buf: str = ""
    reasoning_buf: str = ""
    pending_tool_names: dict[str, str] = field(default_factory=dict)
    last_tool: str = ""
    tools_started: int = 0
    answer_event: dict | None = None

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def salvage(self) -> str:
        """Best text available when the stream did not finish."""
        answer = text.sanitize_answer(self.answer_buf)
        if answer:
            return answer
        if self.reasoning_buf:
            return text.label_recovered_reasoning(self.reasoning_buf)
        return ""


class HolmesClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _build_payload(
        self,
        ask: str,
        conversation_history: list | None,
        metadata: dict,
        user_id: str | None,
    ) -> dict:
        payload = {
            "ask": ask,
            "stream": True,
            "model": self.settings.GENERIC_MODEL_NAME,
            "metadata": metadata,
        }
        if user_id:
            payload["user_id"] = user_id
        if conversation_history:
            payload["conversation_history"] = conversation_history
        if self.settings.ADDITIONAL_SYSTEM_PROMPT.strip():
            payload["additional_system_prompt"] = (
                self.settings.ADDITIONAL_SYSTEM_PROMPT.strip()
            )
        # `enable_tool_approval` is deliberately never sent: approval-gated
        # tools then fail back into the LLM, which works around them.
        return payload

    @staticmethod
    def _build_headers(session_id: str, user_id: str | None) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Session-Id": session_id,
            # LiteLLM proxy/Langfuse metadata header form. Some proxies drop
            # underscore headers, so this is supplemental to the JSON metadata.
            "langfuse_session_id": session_id,
        }
        if user_id:
            headers["X-User-Id"] = user_id
            headers["langfuse_trace_user_id"] = user_id
        return headers

    async def _heartbeat(
        self, state: _StreamState, on_progress: ProgressCallback
    ) -> None:
        """
        Periodic 'still investigating' progress. Claude Code aborts an HTTP
        MCP call after 5 minutes with no response and no progress
        notification; tool-start notifications alone can leave longer gaps
        (a long reasoning step, a slow tool).
        """
        while True:
            await asyncio.sleep(self.settings.HEARTBEAT_SECONDS)
            calls = state.tools_started
            message = (
                f"HolmesGPT still investigating: {_format_duration(state.elapsed())} "
                f"elapsed, {calls} tool call(s) so far"
            )
            if state.last_tool:
                message += f" (last: {state.last_tool})"
            await on_progress(message)

    async def ask(
        self,
        ask: str,
        *,
        session_id: str,
        metadata: dict,
        conversation_history: list | None = None,
        user_id: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> HolmesResult:
        state = _StreamState()
        result = state.result
        heartbeat = (
            asyncio.create_task(self._heartbeat(state, on_progress))
            if on_progress and self.settings.HEARTBEAT_SECONDS > 0
            else None
        )
        try:
            async with asyncio.timeout(self.settings.TOTAL_TIMEOUT_SECONDS):
                await self._ask(
                    state,
                    ask,
                    session_id=session_id,
                    metadata=metadata,
                    conversation_history=conversation_history,
                    user_id=user_id,
                    on_progress=on_progress,
                )
        except TimeoutError:
            result.error = (
                "HolmesGPT did not finish within "
                f"{_format_duration(self.settings.TOTAL_TIMEOUT_SECONDS)}. Try a "
                "narrower question (a specific namespace, workload or time window)."
            )
            result.incomplete = True
            result.answer = state.salvage()
        except httpx.ReadTimeout:
            result.error = (
                "HolmesGPT stopped responding mid-request (no data for "
                f"{self.settings.STALL_TIMEOUT_SECONDS:.0f}s). It may be stuck on "
                "an unresponsive tool or downstream service. Retrying with a "
                "narrower question may avoid the slow tool."
            )
            result.incomplete = True
            result.answer = state.salvage()
        except httpx.HTTPError as exc:
            result.error = (
                "Connection error while talking to HolmesGPT at "
                f"{self.settings.HOLMESGPT_URL}: {type(exc).__name__}: {exc}"
            )
            result.answer = state.salvage()
        except asyncio.CancelledError:
            # The MCP client went away (user cancelled, or Claude Code gave
            # up). Leaving the `async with` closes the HolmesGPT stream, which
            # stops the investigation there too.
            logger.info(
                "HolmesGPT request cancelled by the client: conversation=%s "
                "after=%s tool_calls=%d",
                session_id,
                _format_duration(state.elapsed()),
                len(result.tools_used),
            )
            raise
        finally:
            if heartbeat:
                heartbeat.cancel()

        result.duration_s = state.elapsed()
        return result

    async def _ask(
        self,
        state: _StreamState,
        ask: str,
        *,
        session_id: str,
        metadata: dict,
        conversation_history: list | None,
        user_id: str | None,
        on_progress: ProgressCallback | None,
    ) -> None:
        url = f"{self.settings.HOLMESGPT_URL.rstrip('/')}/api/chat"
        payload = self._build_payload(ask, conversation_history, metadata, user_id)
        headers = self._build_headers(session_id, user_id)
        result = state.result

        # `read` bounds the gap between chunks, not the whole request.
        timeout = httpx.Timeout(
            connect=10.0,
            read=self.settings.STALL_TIMEOUT_SECONDS,
            write=30.0,
            pool=10.0,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as resp:
                if resp.status_code >= 400:
                    body = " ".join((await resp.aread()).decode(
                        "utf-8", errors="replace"
                    ).split())[:400]
                    result.error = f"HolmesGPT returned HTTP {resp.status_code}" + (
                        f": {body}" if body else ""
                    )
                    return

                current_event = None
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()

                    if not line:
                        current_event = None
                        continue
                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        event_data = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Skipping undecodable SSE data line")
                        continue
                    if not isinstance(event_data, dict):
                        continue

                    if await self._handle_event(
                        state, current_event, event_data, on_progress
                    ):
                        break

        self._finish(state)

    async def _handle_event(
        self,
        state: _StreamState,
        event: str | None,
        event_data: dict,
        on_progress: ProgressCallback | None,
    ) -> bool:
        """Apply one SSE event to `state`. Returns True on a terminal event."""
        result = state.result

        if event == "ai_message":
            # HolmesGPT forwards the OpenAI-compatible field
            # `reasoning_content`; some versions use `reasoning`.
            reasoning = (
                event_data.get("reasoning")
                or event_data.get("reasoning_content")
                or ""
            )
            content = event_data.get("content") or ""

            if reasoning:
                head, leaked, _ = text.split_leaked_answer(reasoning)
                head = text.sanitize_reasoning(head)
                if head:
                    state.reasoning_buf, _ = text.merge_stream_text(
                        state.reasoning_buf, head, "\n\n"
                    )
                if leaked:
                    state.answer_buf, _ = text.merge_stream_text(
                        state.answer_buf, leaked, "\n\n"
                    )

            if content and content != reasoning:
                state.answer_buf, _ = text.merge_stream_text(
                    state.answer_buf, content, "\n\n"
                )

        elif event == "start_tool_calling":
            tool_name = event_data.get("tool_name") or ""
            tool_call_id = event_data.get("id") or event_data.get("tool_call_id") or ""
            if tool_call_id:
                state.pending_tool_names[tool_call_id] = tool_name
            state.last_tool = tool_name or state.last_tool
            state.tools_started += 1

            # Still investigating: everything buffered so far is narration
            # leading up to this call, not the answer.
            state.answer_buf = ""
            state.reasoning_buf = ""

            if on_progress:
                await on_progress(
                    f"HolmesGPT is running tool: {tool_name or 'unknown'}"
                )

        elif event == "tool_calling_result":
            tool_call_id, tool_name, params, status, result_text = (
                _extract_tool_call(event_data)
            )
            tool_name = tool_name or state.pending_tool_names.pop(tool_call_id, "")
            result.tools_used.append(
                ToolCall(
                    name=tool_name or "unknown",
                    params=params,
                    status=status,
                    preview=text.truncate(result_text, TOOL_PREVIEW_CHARS),
                )
            )

        elif event == "error":
            result.error = (
                event_data.get("description")
                or event_data.get("msg")
                or "HolmesGPT reported an unknown error"
            )
            return True

        elif event == "approval_required":
            result.error = _format_pending_approvals(
                event_data.get("pending_approvals") or []
            )
            return True

        elif event == "ai_answer_end":
            state.answer_event = event_data
            return True

        return False

    @staticmethod
    def _finish(state: _StreamState) -> None:
        """Turn the collected stream into the final answer (or error)."""
        result = state.result

        if result.error:
            # Keep whatever answer text arrived, so the caller can still show it.
            result.answer = text.sanitize_answer(state.answer_buf)
            return

        answer_event = state.answer_event
        if answer_event is not None:
            history = answer_event.get("conversation_history")
            if isinstance(history, list):
                result.conversation_history = history

            answer = text.choose_final_answer(
                answer_event.get("analysis"), state.answer_buf
            )
            if not answer:
                answer = text.extract_answer_from_history(history)
            if not answer:
                reasoning = state.reasoning_buf or text.extract_reasoning_from_history(
                    history
                )
                if reasoning:
                    answer = text.label_recovered_reasoning(reasoning)
            if not answer:
                result.error = (
                    "HolmesGPT finished the investigation but returned an empty "
                    "answer."
                )
            result.answer = answer
            return

        # The stream ended without a terminal event (server hiccup or a
        # truncated response): salvage what arrived.
        result.incomplete = True
        result.answer = state.salvage()
        if not result.answer:
            result.error = (
                "HolmesGPT ended the stream without an answer. Please try again."
            )
