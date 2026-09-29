import json
import time

import fakeredis
import pytest

from holmes_mcp.conversations import (
    SHRUNK_TOOL_OUTPUT_CHARS,
    Conversation,
    ConversationBusy,
    MemoryConversationStore,
    RedisConversationStore,
    create_store,
    is_valid_conversation_id,
    new_conversation_id,
    shrink_history,
)


def conv(cid: str = "c1") -> Conversation:
    return Conversation(id=cid, history=[{"role": "user", "content": "q"}], user_id="u")


@pytest.fixture
def redis_server():
    return fakeredis.FakeServer()


def redis_store(server, ttl=60) -> RedisConversationStore:
    return RedisConversationStore(
        fakeredis.FakeAsyncRedis(server=server), ttl, "test:conv:"
    )


class TestMemoryStore:
    async def test_round_trip(self):
        store = MemoryConversationStore(ttl_seconds=60, max_conversations=10)
        await store.put(conv())

        loaded = await store.get("c1")

        assert loaded.history == [{"role": "user", "content": "q"}]
        assert loaded.user_id == "u"
        assert await store.get("missing") is None

    async def test_returned_copy_does_not_change_stored_state(self):
        store = MemoryConversationStore(ttl_seconds=60, max_conversations=10)
        await store.put(conv())

        loaded = await store.get("c1")
        loaded.history.append({"role": "user", "content": "changed"})

        assert len((await store.get("c1")).history) == 1

    async def test_ttl_expiry(self, monkeypatch):
        store = MemoryConversationStore(ttl_seconds=60, max_conversations=10)
        now = [1000.0]
        monkeypatch.setattr("holmes_mcp.conversations.time.time", lambda: now[0])
        await store.put(conv())

        now[0] += 61

        assert await store.get("c1") is None

    async def test_lru_cap(self):
        store = MemoryConversationStore(ttl_seconds=60, max_conversations=2)
        for cid in ("a", "b", "c"):
            await store.put(conv(cid))

        assert await store.get("a") is None
        assert await store.get("c") is not None
        assert len(store) == 2


class TestRedisStore:
    async def test_round_trip_with_ttl(self, redis_server):
        store = redis_store(redis_server)
        await store.put(conv())

        loaded = await store.get("c1")

        assert loaded.id == "c1"
        assert loaded.user_id == "u"
        ttl = await store.redis.ttl("test:conv:c1")
        assert 0 < ttl <= 60

    async def test_get_refreshes_ttl(self, redis_server):
        store = redis_store(redis_server)
        await store.put(conv())
        await store.redis.expire("test:conv:c1", 5)

        await store.get("c1")

        assert await store.redis.ttl("test:conv:c1") > 5

    async def test_shared_between_replicas(self, redis_server):
        replica_a = redis_store(redis_server)
        replica_b = redis_store(redis_server)

        await replica_a.put(conv())

        assert (await replica_b.get("c1")).history == conv().history

    async def test_missing_and_corrupt(self, redis_server):
        store = redis_store(redis_server)
        await store.redis.set("test:conv:bad", "{not json")

        assert await store.get("missing") is None
        assert await store.get("bad") is None

    async def test_ping(self, redis_server):
        assert await redis_store(redis_server).ping() is True


def test_create_store_picks_backend(settings):
    assert create_store(settings).backend == "memory"
    settings.REDIS_URL = "redis://localhost:6379/0"
    assert create_store(settings).backend == "redis"


class TestShrinkHistory:
    def history(self):
        return [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "a" * 50_000},
            {"role": "tool", "tool_call_id": "t2", "content": "b" * 20_000},
            {"role": "assistant", "content": "answer"},
        ]

    def test_under_limit_untouched(self):
        history = self.history()
        assert shrink_history(history, 10_000_000) == (history, 0)

    def test_largest_tool_output_shortened_first(self):
        history = self.history()

        shrunk, count = shrink_history(history, 40_000)

        assert count == 1
        assert len(shrunk[2]["content"]) < SHRUNK_TOOL_OUTPUT_CHARS + 200
        assert shrunk[3]["content"] == "b" * 20_000
        assert len(json.dumps(shrunk)) <= 40_000
        # Message count and tool pairing are preserved; the input is not modified.
        assert len(shrunk) == len(history)
        assert history[2]["content"] == "a" * 50_000


@pytest.fixture(params=["memory", "redis"])
def any_store(request):
    if request.param == "memory":
        return MemoryConversationStore(ttl_seconds=60, max_conversations=10)
    return redis_store(fakeredis.FakeServer())


class TestLocks:
    async def test_second_acquire_is_busy_with_start_time(self, any_store):
        await any_store.acquire_lock("c1", 60)

        with pytest.raises(ConversationBusy) as busy:
            await any_store.acquire_lock("c1", 60)

        assert busy.value.since == pytest.approx(time.time(), abs=5)

    async def test_release_frees_the_lock(self, any_store):
        value = await any_store.acquire_lock("c1", 60)
        await any_store.release_lock("c1", value)

        assert await any_store.acquire_lock("c1", 60)

    async def test_release_with_stale_value_keeps_new_owner(self, any_store):
        await any_store.acquire_lock("c1", 60)
        with pytest.raises(ConversationBusy):
            await any_store.acquire_lock("c1", 60)

        await any_store.release_lock("c1", "someone-elses-old-lock:0")

        with pytest.raises(ConversationBusy):
            await any_store.acquire_lock("c1", 60)

    async def test_locks_are_per_conversation(self, any_store):
        await any_store.acquire_lock("c1", 60)
        assert await any_store.acquire_lock("c2", 60)

    async def test_lock_expires(self, monkeypatch):
        store = MemoryConversationStore(ttl_seconds=60, max_conversations=10)
        now = [1000.0]
        monkeypatch.setattr("holmes_mcp.conversations.time.time", lambda: now[0])
        await store.acquire_lock("c1", 30)

        now[0] += 31

        assert await store.acquire_lock("c1", 30)

    async def test_redis_lock_has_ttl(self, redis_server):
        store = redis_store(redis_server)
        await store.acquire_lock("c1", 90)

        assert 0 < await store.redis.ttl("test:conv:c1:lock") <= 90


def test_conversation_id_format():
    assert is_valid_conversation_id(new_conversation_id())
    assert not is_valid_conversation_id("abc")
    assert not is_valid_conversation_id("0123456789ABCDEF")
    assert not is_valid_conversation_id("0123456789abcdef:lock")
