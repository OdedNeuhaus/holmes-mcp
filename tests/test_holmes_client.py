import asyncio
import json

import httpx
import pytest
import respx

from conftest import CHAT_URL, sse, tool_events
from holmes_mcp.holmes_client import HolmesClient


async def ask(settings, **kwargs):
    client = HolmesClient(settings)
    kwargs.setdefault("session_id", "conv1")
    kwargs.setdefault("metadata", {"session_id": "conv1"})
    return await client.ask("why is the pod crashing?", **kwargs)


def sent_payload(route) -> dict:
    return json.loads(route.calls.last.request.content)


@respx.mock
async def test_normal_answer_with_tools(settings):
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "why is the pod crashing?"},
        {"role": "assistant", "content": "It is OOMKilled."},
    ]
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse(
                ("ai_message", {"content": "Let me check the pod."}),
                *tool_events(
                    "t1",
                    "kubectl_describe",
                    {"kind": "pod", "name": "api-1", "namespace": "payments"},
                    "Last State: OOMKilled",
                ),
                ("ai_message", {"content": "It is OOMKilled."}),
                (
                    "ai_answer_end",
                    {"analysis": "It is OOMKilled.", "conversation_history": history},
                ),
            ),
        )
    )

    progress = []

    async def on_progress(message):
        progress.append(message)

    result = await ask(settings, on_progress=on_progress, user_id="dev@org")

    assert result.error is None
    assert result.answer == "It is OOMKilled."
    assert result.conversation_history == history
    assert [c.name for c in result.tools_used] == ["kubectl_describe"]
    assert result.tools_used[0].params["namespace"] == "payments"
    assert progress == ["HolmesGPT is running tool: kubectl_describe"]

    payload = sent_payload(route)
    assert payload["model"] == "generic"
    assert payload["stream"] is True
    assert payload["user_id"] == "dev@org"
    assert "enable_tool_approval" not in payload
    assert "conversation_history" not in payload

    headers = route.calls.last.request.headers
    assert headers["X-Session-Id"] == "conv1"
    assert headers["langfuse_trace_user_id"] == "dev@org"


@respx.mock
async def test_follow_up_sends_history_and_system_prompt(settings):
    settings.ADDITIONAL_SYSTEM_PROMPT = "  ES instances: prod, staging  "
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=sse(("ai_answer_end", {"analysis": "ok"}))
        )
    )
    history = [{"role": "system", "content": "sys"}]

    await ask(settings, conversation_history=history)

    payload = sent_payload(route)
    assert payload["conversation_history"] == history
    assert payload["additional_system_prompt"] == "ES instances: prod, staging"


@respx.mock
async def test_narration_before_tool_calls_is_not_the_answer(settings):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse(
                ("ai_message", {"content": "I'll look at the logs first."}),
                *tool_events("t1", "fetch_pod_logs", {}, "error: boom"),
                ("ai_message", {"content": "The app fails with 'boom'."}),
                ("ai_answer_end", {"analysis": ""}),
            ),
        )
    )

    result = await ask(settings)

    assert result.answer == "The app fails with 'boom'."


@respx.mock
async def test_answer_leaked_into_reasoning_channel(settings):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse(
                (
                    "ai_message",
                    {"reasoning_content": "thinking hard</think>The real answer."},
                ),
                ("ai_answer_end", {"analysis": ""}),
            ),
        )
    )

    result = await ask(settings)

    assert result.answer == "The real answer."


@respx.mock
async def test_empty_analysis_recovered_from_history(settings):
    history = [{"role": "assistant", "content": "Recovered from history."}]
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse(
                ("ai_answer_end", {"analysis": "", "conversation_history": history})
            ),
        )
    )

    result = await ask(settings)

    assert result.answer == "Recovered from history."


@respx.mock
async def test_reasoning_only_is_labelled(settings):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse(
                ("ai_message", {"reasoning_content": "only thoughts here"}),
                ("ai_answer_end", {"analysis": ""}),
            ),
        )
    )

    result = await ask(settings)

    assert result.error is None
    assert "recovered from the model's reasoning" in result.answer
    assert "only thoughts here" in result.answer


@respx.mock
async def test_error_event(settings):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=sse(("error", {"description": "LLM quota exceeded"}))
        )
    )

    result = await ask(settings)

    assert result.error == "LLM quota exceeded"


@respx.mock
async def test_http_error(settings):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(500, text="internal\n  server   error")
    )

    result = await ask(settings)

    assert result.error == "HolmesGPT returned HTTP 500: internal server error"


@respx.mock
async def test_truncated_stream_salvages_answer(settings):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=sse(("ai_message", {"content": "Partial findings."}))
        )
    )

    result = await ask(settings)

    assert result.error is None
    assert result.incomplete is True
    assert result.answer == "Partial findings."


@respx.mock
async def test_truncated_stream_with_nothing(settings):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, content=b""))

    result = await ask(settings)

    assert "without an answer" in result.error


@respx.mock
async def test_approval_required_is_reported(settings):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse(
                (
                    "approval_required",
                    {"pending_approvals": [{"tool_name": "kubectl_delete"}]},
                )
            ),
        )
    )

    result = await ask(settings)

    assert "kubectl_delete" in result.error
    assert "not supported" in result.error


@respx.mock
async def test_stall_gives_clear_error(settings):
    respx.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("stalled"))

    result = await ask(settings)

    assert "stopped responding" in result.error
    assert "5s" in result.error


@respx.mock
async def test_connection_error(settings):
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))

    result = await ask(settings)

    assert result.error.startswith("Connection error while talking to HolmesGPT")


@respx.mock
async def test_failed_tool_status_is_kept(settings):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse(
                *tool_events("t1", "prometheus_query", {"q": "up"}, None, status="error"),
                ("ai_answer_end", {"analysis": "Prometheus was unreachable."}),
            ),
        )
    )

    result = await ask(settings)

    assert result.tools_used[0].status == "error"


@pytest.mark.parametrize("data_line", ["not json", "[DONE]"])
@respx.mock
async def test_odd_data_lines(settings, data_line):
    body = (
        f"event: ai_message\ndata: {data_line}\n\n".encode()
        + sse(("ai_answer_end", {"analysis": "fine"}))
    )
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, content=body))

    result = await ask(settings)

    if data_line == "[DONE]":
        # [DONE] ends the stream before the answer event.
        assert result.incomplete or result.error
    else:
        assert result.answer == "fine"


class SlowStream(httpx.AsyncByteStream):
    """An SSE body delivered in timed steps; a step may raise instead."""

    def __init__(self, *steps):
        # steps: bytes to send, a float to sleep, or an exception to raise
        self.steps = steps

    async def __aiter__(self):
        for step in self.steps:
            if isinstance(step, float):
                await asyncio.sleep(step)
            elif isinstance(step, BaseException):
                raise step
            else:
                yield step


@respx.mock
async def test_heartbeat_progress_during_long_silence(settings):
    settings.HEARTBEAT_SECONDS = 0.05
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            stream=SlowStream(
                sse(*tool_events("t1", "fetch_pod_logs", {}, "logs")),
                0.3,
                sse(("ai_answer_end", {"analysis": "done"})),
            ),
        )
    )
    progress = []

    async def on_progress(message):
        progress.append(message)

    result = await ask(settings, on_progress=on_progress)

    assert result.answer == "done"
    heartbeats = [m for m in progress if m.startswith("HolmesGPT still investigating")]
    assert heartbeats, progress
    assert "1 tool call(s) so far (last: fetch_pod_logs)" in heartbeats[-1]


@respx.mock
async def test_total_timeout_returns_partial_findings(settings):
    settings.TOTAL_TIMEOUT_SECONDS = 0.3
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            stream=SlowStream(
                sse(
                    *tool_events("t1", "kubectl_describe", {"name": "api-1"}, "OOMKilled"),
                    ("ai_message", {"content": "api-1 was OOMKilled; checking limits."}),
                ),
                5.0,
            ),
        )
    )

    result = await ask(settings)

    assert "did not finish within 0s" in result.error
    assert result.incomplete is True
    assert result.answer == "api-1 was OOMKilled; checking limits."
    assert [c.name for c in result.tools_used] == ["kubectl_describe"]


@respx.mock
async def test_stall_mid_stream_returns_partial_findings(settings):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            stream=SlowStream(
                sse(
                    *tool_events("t1", "prometheus_query", {"q": "up"}, "1"),
                    ("ai_message", {"content": "Prometheus looks fine."}),
                ),
                httpx.ReadTimeout("stalled"),
            ),
        )
    )

    result = await ask(settings)

    assert "stopped responding" in result.error
    assert result.incomplete is True
    assert result.answer == "Prometheus looks fine."
    assert len(result.tools_used) == 1


@respx.mock
async def test_client_cancellation_is_logged_and_propagates(settings, caplog):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            stream=SlowStream(
                sse(*tool_events("t1", "fetch_pod_logs", {}, "x")), 5.0
            ),
        )
    )
    task = asyncio.create_task(ask(settings))
    await asyncio.sleep(0.1)
    task.cancel()

    with caplog.at_level("INFO"), pytest.raises(asyncio.CancelledError):
        await task

    assert "cancelled by the client: conversation=conv1" in caplog.text
    assert "tool_calls=1" in caplog.text
