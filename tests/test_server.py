import asyncio
import re

import fakeredis
import pytest
from mcp import Client
from starlette.testclient import TestClient

from holmes_mcp.conversations import MemoryConversationStore, RedisConversationStore
from holmes_mcp.holmes_client import HolmesResult, ToolCall
from holmes_mcp.server import create_app, create_server, format_result


class FakeHolmes:
    """Stands in for HolmesClient: records calls, returns queued results."""

    def __init__(self, *results: HolmesResult):
        self.results = list(results)
        self.calls: list[dict] = []

    async def ask(self, ask, **kwargs):
        self.calls.append({"ask": ask, **kwargs})
        if kwargs.get("on_progress"):
            await kwargs["on_progress"]("HolmesGPT is running tool: x")
        return self.results.pop(0)


def answer(text: str, history: list | None = None, tools=()) -> HolmesResult:
    return HolmesResult(
        answer=text,
        conversation_history=history,
        tools_used=list(tools),
        duration_s=12.3,
    )


def conversation_id(result_text: str) -> str:
    match = re.search(r"conversation_id: (\w+)", result_text)
    assert match, result_text
    return match.group(1)


def result_text(result) -> str:
    return result.content[0].text


class BrokenStore(MemoryConversationStore):
    async def put(self, conv):
        raise ConnectionError("redis down")

    async def get(self, conversation_id):
        raise ConnectionError("redis down")


@pytest.fixture
def memory_store():
    return MemoryConversationStore(ttl_seconds=60, max_conversations=10)


async def call(server, tool: str, args: dict):
    async with Client(server) as client:
        return await client.call_tool(tool, args)


async def test_lists_tools_and_no_prompts(settings, memory_store):
    server = create_server(settings, FakeHolmes(), memory_store)
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        prompts = (await client.list_prompts()).prompts

    assert set(tools) == {"ask_holmes", "holmes_follow_up"}
    assert tools["ask_holmes"].annotations.read_only_hint is True
    assert prompts == []


async def test_ask_then_follow_up(settings, memory_store):
    first_history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "It is OOMKilled."},
    ]
    holmes = FakeHolmes(
        answer(
            "It is OOMKilled.",
            first_history,
            [ToolCall("kubectl_describe", {"namespace": "payments"})],
        ),
        answer("Raise the limit to 512Mi.", first_history + [{"role": "user"}]),
    )
    server = create_server(settings, holmes, memory_store)

    first = await call(
        server,
        "ask_holmes",
        {"question": "Why is api crashing?", "context": "deployment api in ns payments"},
    )
    text = result_text(first)
    assert not first.is_error
    assert text.startswith("It is OOMKilled.")
    assert 'kubectl_describe {"namespace":"payments"}' in text
    assert holmes.calls[0]["ask"] == (
        "Why is api crashing?\n\nContext from the developer's workspace:\n"
        "deployment api in ns payments"
    )
    assert holmes.calls[0]["conversation_history"] is None
    cid = conversation_id(text)
    assert holmes.calls[0]["session_id"] == cid
    assert holmes.calls[0]["metadata"]["session_id"] == cid

    second = await call(
        server, "holmes_follow_up", {"conversation_id": cid, "question": "How to fix?"}
    )
    assert not second.is_error
    assert result_text(second).startswith("Raise the limit to 512Mi.")
    assert holmes.calls[1]["conversation_history"] == first_history
    assert holmes.calls[1]["metadata"]["trace_metadata"]["turn"] == 2
    assert (await memory_store.get(cid)).turns == 2


async def test_follow_up_on_another_replica_via_redis(settings):
    """Two server instances sharing Redis behave like two replicas."""
    redis_server = fakeredis.FakeServer()

    def replica(holmes):
        store = RedisConversationStore(
            fakeredis.FakeAsyncRedis(server=redis_server), 60, "test:conv:"
        )
        return create_server(settings, holmes, store)

    history = [{"role": "system", "content": "sys"}]
    holmes_a = FakeHolmes(answer("first", history))
    holmes_b = FakeHolmes(answer("second"))

    first = await call(replica(holmes_a), "ask_holmes", {"question": "q"})
    cid = conversation_id(result_text(first))
    second = await call(
        replica(holmes_b), "holmes_follow_up", {"conversation_id": cid, "question": "q2"}
    )

    assert not second.is_error
    assert holmes_b.calls[0]["conversation_history"] == history


async def test_fallback_history_when_holmes_returns_none(settings, memory_store):
    holmes = FakeHolmes(answer("first answer"), answer("second answer"))
    server = create_server(settings, holmes, memory_store)

    cid = conversation_id(result_text(await call(server, "ask_holmes", {"question": "q1"})))
    await call(server, "holmes_follow_up", {"conversation_id": cid, "question": "q2"})

    history = holmes.calls[1]["conversation_history"]
    assert [m["role"] for m in history] == ["system", "user", "assistant"]
    assert history[2]["content"] == "first answer"


async def test_unknown_conversation(settings, memory_store):
    server = create_server(settings, FakeHolmes(), memory_store)

    result = await call(
        server, "holmes_follow_up", {"conversation_id": "nope", "question": "q"}
    )

    assert result.is_error
    assert "Unknown or expired conversation_id 'nope'" in result_text(result)


async def test_holmes_error_is_a_tool_error(settings, memory_store):
    failed = HolmesResult(error="HolmesGPT returned HTTP 502", answer="partial")
    server = create_server(settings, FakeHolmes(failed), memory_store)

    result = await call(server, "ask_holmes", {"question": "q"})

    assert result.is_error
    assert "HTTP 502" in result_text(result)
    assert "partial" in result_text(result)
    assert len(memory_store) == 0


async def test_store_write_failure_still_returns_answer(settings):
    server = create_server(settings, FakeHolmes(answer("the answer")), BrokenStore(60, 10))

    result = await call(server, "ask_holmes", {"question": "q"})

    assert not result.is_error
    assert result_text(result).startswith("the answer")
    assert "could not be saved" in result_text(result)


async def test_follow_up_save_failure_warns_against_reusing_id(settings, memory_store):
    holmes = FakeHolmes(answer("first"), answer("second"))
    server = create_server(settings, holmes, memory_store)
    cid = conversation_id(result_text(await call(server, "ask_holmes", {"question": "q"})))

    async def failing_put(conv):
        raise ConnectionError("redis down")

    memory_store.put = failing_put
    result = await call(
        server, "holmes_follow_up", {"conversation_id": cid, "question": "q2"}
    )

    text = result_text(result)
    assert not result.is_error
    assert text.startswith("second")
    assert f"Do not use holmes_follow_up with {cid} again" in text
    assert "conversation_id:" not in text


async def test_store_read_failure_is_explained(settings):
    server = create_server(settings, FakeHolmes(), BrokenStore(60, 10))

    result = await call(
        server,
        "holmes_follow_up",
        {"conversation_id": "0123456789abcdef", "question": "q"},
    )

    assert result.is_error
    assert "could not read its conversation store" in result_text(result)


async def test_caller_header_is_used_for_attribution(settings, memory_store):
    # The in-process transport carries no HTTP headers: no user id is sent.
    holmes = FakeHolmes(answer("ok"))
    await call(create_server(settings, holmes, memory_store), "ask_holmes", {"question": "q"})

    assert holmes.calls[0]["user_id"] is None
    assert "trace_user_id" not in holmes.calls[0]["metadata"]


def test_format_result_truncates_answer_but_keeps_footer():
    result = answer("x" * 10_000, tools=[ToolCall("t", {})])

    out = format_result(result, "cid123", max_chars=3000)

    assert len(out) <= 3000
    assert "truncated" in out
    assert "conversation_id: cid123" in out


def test_format_result_marks_failed_tools_and_incomplete():
    result = answer(
        "partial",
        tools=[ToolCall("prometheus_query", {"q": "up"}, "error", "connection refused")],
    )
    result.incomplete = True

    out = format_result(result, "cid", max_chars=60_000)

    assert "-> failed: connection refused" in out
    assert "may be incomplete" in out


def test_health_routes(settings):
    client = TestClient(create_app(settings))

    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ok", "store": "memory"}


# --- input validation (#6) -------------------------------------------------


async def test_blank_question_rejected(settings, memory_store):
    holmes = FakeHolmes()
    server = create_server(settings, holmes, memory_store)

    result = await call(server, "ask_holmes", {"question": "   \n  "})

    assert result.is_error
    assert "question must not be empty" in result_text(result)
    assert holmes.calls == []


async def test_oversized_input_rejected_with_sizes(settings, memory_store):
    settings.MAX_INPUT_CHARS = 1000
    holmes = FakeHolmes()
    server = create_server(settings, holmes, memory_store)

    result = await call(
        server, "ask_holmes", {"question": "why?", "context": "x" * 2000}
    )

    text = result_text(result)
    assert result.is_error
    assert "2,004 characters (question 4, context 2,000)" in text
    assert "the limit is 1,000" in text
    assert holmes.calls == []


async def test_malformed_conversation_id_never_reaches_the_store(settings):
    server = create_server(settings, FakeHolmes(), BrokenStore(60, 10))

    result = await call(
        server,
        "holmes_follow_up",
        {"conversation_id": "../../etc", "question": "q"},
    )

    # BrokenStore would have produced a store error if it had been touched.
    assert "Unknown or expired conversation_id" in result_text(result)


async def test_conversation_id_is_normalized(settings, memory_store):
    holmes = FakeHolmes(answer("first"), answer("second"))
    server = create_server(settings, holmes, memory_store)
    cid = conversation_id(result_text(await call(server, "ask_holmes", {"question": "q"})))

    result = await call(
        server,
        "holmes_follow_up",
        {"conversation_id": f"  {cid.upper()} ", "question": "q2"},
    )

    assert not result.is_error


# --- parallel follow-ups (#5) ----------------------------------------------


class SlowHolmes(FakeHolmes):
    """Blocks inside ask() until released, to overlap two calls."""

    def __init__(self, *results):
        super().__init__(*results)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def ask(self, ask, **kwargs):
        self.entered.set()
        await self.release.wait()
        return await super().ask(ask, **kwargs)


async def test_parallel_follow_up_on_same_conversation_is_rejected(
    settings, memory_store
):
    holmes = SlowHolmes(answer("first"), answer("second"))
    server = create_server(settings, holmes, memory_store)
    holmes.release.set()
    cid = conversation_id(result_text(await call(server, "ask_holmes", {"question": "q"})))
    holmes.release.clear()
    holmes.entered.clear()

    async with Client(server) as client:
        first = asyncio.create_task(
            client.call_tool(
                "holmes_follow_up", {"conversation_id": cid, "question": "a"}
            )
        )
        await holmes.entered.wait()
        second = await client.call_tool(
            "holmes_follow_up", {"conversation_id": cid, "question": "b"}
        )
        holmes.release.set()
        first = await first

    assert not first.is_error
    assert second.is_error
    assert f"A follow-up on conversation {cid} is already running" in result_text(
        second
    )
    assert "started 0s ago" in result_text(second)
    assert len(holmes.calls) == 2  # the ask + one follow-up; "b" never ran


async def test_lock_released_after_holmes_error(settings, memory_store):
    failed = HolmesResult(error="HolmesGPT returned HTTP 502")
    holmes = FakeHolmes(answer("first"), failed, answer("third"))
    server = create_server(settings, holmes, memory_store)
    cid = conversation_id(result_text(await call(server, "ask_holmes", {"question": "q"})))

    second = await call(server, "holmes_follow_up", {"conversation_id": cid, "question": "b"})
    third = await call(server, "holmes_follow_up", {"conversation_id": cid, "question": "c"})

    assert second.is_error
    assert f"Conversation {cid} is unchanged" in result_text(second)
    assert not third.is_error


async def test_follow_up_sees_history_saved_by_the_previous_one(settings, memory_store):
    holmes = FakeHolmes(
        answer("first", [{"role": "system", "content": "s"}]),
        answer("second", [{"role": "system", "content": "s"}, {"role": "user", "content": "b"}]),
        answer("third"),
    )
    server = create_server(settings, holmes, memory_store)
    cid = conversation_id(result_text(await call(server, "ask_holmes", {"question": "q"})))

    await call(server, "holmes_follow_up", {"conversation_id": cid, "question": "b"})
    await call(server, "holmes_follow_up", {"conversation_id": cid, "question": "c"})

    assert len(holmes.calls[2]["conversation_history"]) == 2


# --- partial findings (#4) and clarifying questions (#7) -------------------


async def test_cut_off_investigation_reports_partial_findings(settings, memory_store):
    cut_off = HolmesResult(
        error="HolmesGPT did not finish within 30m00s.",
        answer="api-1 was OOMKilled; checking limits.",
        incomplete=True,
        tools_used=[ToolCall("kubectl_describe", {"name": "api-1"})],
        duration_s=1800,
    )
    server = create_server(settings, FakeHolmes(cut_off), memory_store)

    result = await call(server, "ask_holmes", {"question": "q"})

    text = result_text(result)
    assert result.is_error
    assert "did not finish within 30m00s" in text
    assert "Partial findings before HolmesGPT was cut off (not a final answer)" in text
    assert "api-1 was OOMKilled" in text
    assert 'kubectl_describe {"name":"api-1"}' in text


async def test_clarifying_question_guidance(settings, memory_store):
    server = create_server(
        settings, FakeHolmes(answer("Which namespace is payments-api in?")), memory_store
    )
    async with Client(server) as client:
        result = await client.call_tool("ask_holmes", {"question": "q"})

    assert "If HolmesGPT asked a question or needs more information" in result_text(
        result
    )
    assert "reply with `holmes_follow_up`" in server.instructions
