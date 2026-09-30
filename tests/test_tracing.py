"""
Tracing tests run the real Langfuse SDK and capture what it would export in
memory (no Langfuse server needed), so they check the actual span attributes
Langfuse receives.
"""

import asyncio
import re

import pytest
from langfuse import Langfuse
from mcp import Client
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from types import SimpleNamespace

from holmes_mcp.conversations import MemoryConversationStore
from holmes_mcp.holmes_client import HolmesResult, ToolCall
from holmes_mcp.server import Caller, _caller_from_request, create_server
from holmes_mcp.tracing import (
    TOOL_OUTPUT_CHARS,
    LangfuseTracer,
    TraceInfo,
    create_tracer,
)

EXPORTER = InMemorySpanExporter()
# One SDK client for the whole module: the SDK shares resources per public key.
LANGFUSE = Langfuse(
    public_key="pk-lf-holmes-mcp-tests",
    secret_key="sk-lf-holmes-mcp-tests",
    host="http://127.0.0.1:9",
    span_exporter=EXPORTER,
)


@pytest.fixture
def tracer():
    EXPORTER.clear()
    return LangfuseTracer(LANGFUSE)


def exported():
    LANGFUSE.flush()
    return EXPORTER.get_finished_spans()


def roots(spans):
    return [s for s in spans if s.attributes.get("langfuse.internal.as_root")]


def children_of(spans, root):
    return [s for s in spans if s.parent and s.parent.span_id == root.context.span_id]


class ToolEmittingHolmes:
    """Fake HolmesClient that reports its tool calls to the observer."""

    def __init__(self, *results: HolmesResult):
        self.results = list(results)
        self.calls = []

    async def ask(self, ask, **kwargs):
        self.calls.append({"ask": ask, **kwargs})
        result = self.results.pop(0)
        observer = kwargs["observer"]
        for i, call in enumerate(result.tools_used):
            observer.tool_started(f"t{i}", call.name)
            observer.tool_finished(
                f"t{i}", call.name, call.params, call.status, call.preview
            )
        return result


def answer(text, tools=(), history=None):
    return HolmesResult(
        answer=text, tools_used=list(tools), conversation_history=history, duration_s=3.2
    )


async def call(server, tool, args):
    async with Client(server) as client:
        return await client.call_tool(tool, args)


def cid_of(result) -> str:
    return re.search(r"conversation_id: (\w+)", result.content[0].text).group(1)


async def test_conversation_is_one_session_with_one_trace_per_call(settings, tracer):
    holmes = ToolEmittingHolmes(
        answer(
            "api-1 is OOMKilled.",
            [ToolCall("kubectl_describe", {"name": "api-1"}, "success", "OOMKilled")],
            history=[{"role": "system", "content": "s"}],
        ),
        answer("Raise the limit.", [ToolCall("fetch_pod_logs", {}, "success", "ok")]),
    )
    store = MemoryConversationStore(60, 10)
    server = create_server(settings, holmes, store, tracer)

    first = await call(
        server, "ask_holmes", {"question": "Why?", "context": "ns payments"}
    )
    cid = cid_of(first)
    await call(server, "holmes_follow_up", {"conversation_id": cid, "question": "Fix?"})

    spans = exported()
    ask_root, follow_root = sorted(roots(spans), key=lambda s: s.start_time)

    for root in (ask_root, follow_root):
        assert root.attributes["session.id"] == cid
        assert root.attributes["user.id"] == "unknown"
        assert root.attributes["langfuse.trace.tags"] == ("holmes-mcp", "claude-code")
        assert root.attributes["langfuse.observation.metadata.status"] == "success"

    assert ask_root.attributes["langfuse.trace.name"] == "ask_holmes"
    assert follow_root.attributes["langfuse.trace.name"] == "holmes_follow_up"
    assert ask_root.attributes["langfuse.trace.metadata.turn"] == "1"
    assert follow_root.attributes["langfuse.trace.metadata.turn"] == "2"
    assert '"context": "ns payments"' in ask_root.attributes["langfuse.observation.input"]
    assert ask_root.attributes["langfuse.observation.output"] == "api-1 is OOMKilled."

    # The Langfuse trace id is the trace_id HolmesGPT receives in its metadata.
    sent_trace_id = holmes.calls[0]["metadata"]["trace_id"]
    assert format(ask_root.context.trace_id, "032x") == sent_trace_id

    (tool,) = children_of(spans, ask_root)
    assert tool.name == "kubectl_describe"
    assert tool.attributes["langfuse.observation.type"] == "tool"
    assert tool.attributes["langfuse.observation.input"] == '{"name": "api-1"}'
    assert tool.attributes["langfuse.observation.output"] == "OOMKilled"


async def test_failed_call_is_traced_as_error(settings, tracer):
    failed = HolmesResult(
        error="HolmesGPT returned HTTP 502",
        tools_used=[ToolCall("prometheus_query", {"q": "up"}, "error", "refused")],
    )
    server = create_server(
        settings, ToolEmittingHolmes(failed), MemoryConversationStore(60, 10), tracer
    )

    result = await call(server, "ask_holmes", {"question": "q"})

    assert result.is_error
    spans = exported()
    (root,) = roots(spans)
    assert root.attributes["langfuse.observation.level"] == "ERROR"
    assert root.attributes["langfuse.observation.status_message"] == (
        "HolmesGPT returned HTTP 502"
    )
    assert root.attributes["langfuse.observation.metadata.status"] == "error"
    (tool,) = children_of(spans, root)
    assert tool.attributes["langfuse.observation.level"] == "ERROR"


async def test_cut_off_investigation_is_a_warning_with_partial_output(settings, tracer):
    cut_off = HolmesResult(
        error="HolmesGPT did not finish within 30m00s.",
        answer="partial findings",
        incomplete=True,
    )
    server = create_server(
        settings, ToolEmittingHolmes(cut_off), MemoryConversationStore(60, 10), tracer
    )

    await call(server, "ask_holmes", {"question": "q"})

    (root,) = roots(exported())
    assert root.attributes["langfuse.observation.level"] == "WARNING"
    assert root.attributes["langfuse.observation.metadata.status"] == "cut_off"
    assert root.attributes["langfuse.observation.output"] == "partial findings"


async def test_cancellation_is_recorded_and_open_tools_are_closed(tracer):
    info = TraceInfo(
        name="ask_holmes",
        trace_id=LANGFUSE.create_trace_id(),
        session_id="0123456789abcdef",
        user_id="dev@example.org",
        turn=1,
        input={"question": "q"},
    )
    with pytest.raises(asyncio.CancelledError):
        with tracer.trace(info) as trace:
            trace.tool_started("t1", "fetch_pod_logs")
            raise asyncio.CancelledError()

    spans = exported()
    (root,) = roots(spans)
    assert root.attributes["langfuse.observation.metadata.status"] == "cancelled"
    assert root.attributes["langfuse.observation.status_message"] == (
        "cancelled by the client"
    )
    (tool,) = children_of(spans, root)
    assert tool.attributes["langfuse.observation.status_message"] == "did not finish"


async def test_tool_output_is_truncated(tracer):
    info = TraceInfo("ask_holmes", LANGFUSE.create_trace_id(), "s", "u", 1, {})
    with tracer.trace(info) as trace:
        trace.tool_started("t1", "fetch_pod_logs")
        trace.tool_finished("t1", "fetch_pod_logs", {}, "success", "x" * 10_000)
        trace.finish(status="success")

    spans = exported()
    (root,) = roots(spans)
    (tool,) = children_of(spans, root)
    assert len(tool.attributes["langfuse.observation.output"]) <= TOOL_OUTPUT_CHARS


async def test_caller_identity_reaches_the_trace(tracer):
    who = Caller(
        email=None,
        username="alice",
        hostname="alice-laptop",
        client_ip="10.1.2.3",
        user_agent="claude-code/2.1",
    )
    info = TraceInfo(
        "ask_holmes",
        LANGFUSE.create_trace_id(),
        "s",
        who.user_id,
        1,
        {},
        who.trace_metadata(),
    )
    with tracer.trace(info) as trace:
        trace.finish(status="success")

    (root,) = roots(exported())
    assert root.attributes["user.id"] == "alice@alice-laptop"
    assert root.attributes["langfuse.trace.metadata.client_ip"] == "10.1.2.3"
    assert root.attributes["langfuse.trace.metadata.hostname"] == "alice-laptop"


async def test_broken_tracer_never_breaks_a_request(settings):
    broken = LangfuseTracer(client=object())  # every SDK call raises
    server = create_server(
        settings,
        ToolEmittingHolmes(answer("still answered")),
        MemoryConversationStore(60, 10),
        broken,
    )

    result = await call(server, "ask_holmes", {"question": "q"})

    assert not result.is_error
    assert result.content[0].text.startswith("still answered")


def test_caller_from_request_headers():
    request = SimpleNamespace(
        headers={
            "X-User-Email": "dev@example.org",
            "X-Username": "dev",
            "X-Hostname": "dev-laptop",
            "x-forwarded-for": "10.9.8.7, 10.0.0.1",
            "user-agent": "claude-code/2.1",
        },
        client=SimpleNamespace(host="10.0.0.1"),
    )

    who = _caller_from_request(request, "X-User-Email")

    assert who.user_id == "dev@example.org"
    assert who.client_ip == "10.9.8.7"
    assert who.hostname == "dev-laptop"


def test_caller_without_headers_uses_socket_ip():
    request = SimpleNamespace(headers={}, client=SimpleNamespace(host="10.0.0.5"))

    who = _caller_from_request(request, "X-User-Email")

    assert who.user_id is None
    assert who.client_ip == "10.0.0.5"


def test_create_tracer_needs_all_three_settings(settings, caplog, monkeypatch):
    # Don't build a real SDK client here: it would attach to the process-wide
    # OpenTelemetry provider and try to export the other tests' spans.
    built = []
    monkeypatch.setattr(
        LangfuseTracer,
        "from_settings",
        classmethod(lambda cls, s: built.append(s) or cls(client=object())),
    )

    assert create_tracer(settings).enabled is False

    settings.LANGFUSE_HOST = "http://langfuse.test"
    with caplog.at_level("WARNING"):
        assert create_tracer(settings).enabled is False
    assert "must all be set" in caplog.text
    assert built == []

    settings.LANGFUSE_PUBLIC_KEY = "pk-lf-x"
    settings.LANGFUSE_SECRET_KEY = "sk-lf-x"
    assert create_tracer(settings).enabled is True
    assert built == [settings]


def test_traces_are_flushed_during_app_shutdown(settings, monkeypatch):
    from starlette.testclient import TestClient

    from holmes_mcp import server as server_module

    calls = []

    class RecordingTracer(server_module.Tracer):
        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(server_module, "create_tracer", lambda s: RecordingTracer())
    app = server_module.create_app(settings)

    with TestClient(app) as client:  # runs the ASGI lifespan
        assert client.get("/healthz").status_code == 200
        assert calls == []

    assert calls == ["shutdown"]
