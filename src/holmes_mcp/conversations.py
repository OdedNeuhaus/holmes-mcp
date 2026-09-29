"""
Conversation store: HolmesGPT conversation histories keyed by conversation_id.

Two backends share one interface:

* RedisConversationStore - shared by all replicas, so any replica can serve a
  follow-up, and conversations survive restarts. Used whenever REDIS_URL is set.
* MemoryConversationStore - process-local. Only correct with a single replica;
  meant for local development and tests.

Two concurrent follow-ups on the same conversation are last-write-wins: each
continues from the history it read, and the later one to finish is stored.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Protocol

from redis.exceptions import WatchError

from holmes_mcp.config import Settings

logger = logging.getLogger(__name__)

# How much of a tool output survives when a history is shrunk to fit
# MAX_HISTORY_CHARS.
SHRUNK_TOOL_OUTPUT_CHARS = 4000


@dataclass
class Conversation:
    id: str
    history: list
    user_id: str | None = None
    turns: int = 1
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Conversation":
        data = json.loads(raw)
        return cls(
            id=data["id"],
            history=data.get("history") or [],
            user_id=data.get("user_id"),
            turns=int(data.get("turns") or 1),
            created=float(data.get("created") or time.time()),
            updated=float(data.get("updated") or time.time()),
        )


CONVERSATION_ID_RE = re.compile(r"^[0-9a-f]{16}$")


def new_conversation_id() -> str:
    return uuid.uuid4().hex[:16]


def is_valid_conversation_id(value: str) -> bool:
    return bool(CONVERSATION_ID_RE.match(value))


class ConversationBusy(Exception):
    """Another follow-up on the same conversation is still running."""

    def __init__(self, since: float | None):
        super().__init__("conversation is busy")
        # time.time() when the running follow-up took the lock, if known.
        self.since = since


def _new_lock_value() -> str:
    return f"{uuid.uuid4().hex}:{time.time():.3f}"


def _lock_since(value: str | bytes | None) -> float | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    try:
        return float(value.rsplit(":", 1)[1]) if value else None
    except (IndexError, ValueError):
        return None


def shrink_history(history: list, max_chars: int) -> tuple[list, int]:
    """
    Bring a conversation history under `max_chars` of JSON by shortening the
    largest tool outputs first.

    Only `tool` message contents are touched: dropping messages would break
    the pairing between an assistant `tool_calls` entry and its result, which
    the LLM API rejects. Returns (history, number_of_tool_outputs_shortened).
    The input list is not modified.
    """
    total = len(json.dumps(history, ensure_ascii=False))
    if max_chars <= 0 or total <= max_chars:
        return history, 0

    candidates = sorted(
        (
            i
            for i, msg in enumerate(history)
            if isinstance(msg, dict)
            and msg.get("role") == "tool"
            and isinstance(msg.get("content"), str)
            and len(msg["content"]) > SHRUNK_TOOL_OUTPUT_CHARS
        ),
        key=lambda i: len(history[i]["content"]),
        reverse=True,
    )

    shrunk = list(history)
    count = 0
    for i in candidates:
        if total <= max_chars:
            break
        content = shrunk[i]["content"]
        kept = content[:SHRUNK_TOOL_OUTPUT_CHARS]
        new_content = (
            f"{kept}\n\n…(tool output shortened from {len(content)} characters "
            "to keep the conversation within its size limit)"
        )
        shrunk[i] = {**shrunk[i], "content": new_content}
        total -= len(content) - len(new_content)
        count += 1

    if total > max_chars:
        logger.warning(
            "Conversation history still ~%d chars after shortening %d tool "
            "outputs (limit %d)",
            total,
            count,
            max_chars,
        )
    return shrunk, count


class ConversationStore(Protocol):
    backend: str

    async def get(self, conversation_id: str) -> Conversation | None: ...

    async def put(self, conv: Conversation) -> None: ...

    async def acquire_lock(self, conversation_id: str, ttl_seconds: int) -> str:
        """Take the per-conversation lock; returns a value for release_lock.

        Raises ConversationBusy when another follow-up holds it. The lock
        expires by itself after `ttl_seconds`, so a crashed replica cannot
        block a conversation forever.
        """
        ...

    async def release_lock(self, conversation_id: str, value: str) -> None:
        """Release the lock, but only if `value` still owns it."""
        ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...


class MemoryConversationStore:
    backend = "memory"

    def __init__(self, ttl_seconds: float, max_conversations: int):
        self.ttl_seconds = ttl_seconds
        self.max_conversations = max_conversations
        self._items: OrderedDict[str, Conversation] = OrderedDict()
        self._lock = asyncio.Lock()
        # conversation_id -> (lock value, expires at)
        self._conv_locks: dict[str, tuple[str, float]] = {}

    def _evict_expired(self, now: float) -> None:
        expired = [
            cid
            for cid, conv in self._items.items()
            if now - conv.updated > self.ttl_seconds
        ]
        for cid in expired:
            del self._items[cid]

    async def get(self, conversation_id: str) -> Conversation | None:
        async with self._lock:
            now = time.time()
            self._evict_expired(now)
            conv = self._items.get(conversation_id)
            if conv is None:
                return None
            # Sliding expiry, like Redis GETEX below.
            conv.updated = now
            self._items.move_to_end(conversation_id)
            # Hand out a copy so callers cannot mutate the stored state.
            return Conversation.from_json(conv.to_json())

    async def put(self, conv: Conversation) -> None:
        async with self._lock:
            now = time.time()
            conv.updated = now
            self._items[conv.id] = Conversation.from_json(conv.to_json())
            self._items.move_to_end(conv.id)
            self._evict_expired(now)
            while len(self._items) > self.max_conversations:
                self._items.popitem(last=False)

    async def acquire_lock(self, conversation_id: str, ttl_seconds: int) -> str:
        async with self._lock:
            now = time.time()
            held = self._conv_locks.get(conversation_id)
            if held and held[1] > now:
                raise ConversationBusy(_lock_since(held[0]))
            value = _new_lock_value()
            self._conv_locks[conversation_id] = (value, now + ttl_seconds)
            return value

    async def release_lock(self, conversation_id: str, value: str) -> None:
        async with self._lock:
            held = self._conv_locks.get(conversation_id)
            if held and held[0] == value:
                del self._conv_locks[conversation_id]

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self._items)


class RedisConversationStore:
    backend = "redis"

    def __init__(self, redis, ttl_seconds: int, key_prefix: str):
        # `redis` is a redis.asyncio.Redis (or compatible) client.
        self.redis = redis
        self.ttl_seconds = int(ttl_seconds)
        self.key_prefix = key_prefix

    @classmethod
    def from_url(cls, url: str, ttl_seconds: int, key_prefix: str):
        import redis.asyncio as redis_asyncio

        client = redis_asyncio.from_url(
            url,
            socket_connect_timeout=5,
            socket_timeout=10,
            health_check_interval=30,
        )
        return cls(client, ttl_seconds, key_prefix)

    def _key(self, conversation_id: str) -> str:
        return f"{self.key_prefix}{conversation_id}"

    def _lock_key(self, conversation_id: str) -> str:
        return f"{self.key_prefix}{conversation_id}:lock"

    async def get(self, conversation_id: str) -> Conversation | None:
        # GETEX refreshes the TTL in the same round trip: an actively used
        # conversation never expires mid-investigation.
        raw = await self.redis.getex(self._key(conversation_id), ex=self.ttl_seconds)
        if raw is None:
            return None
        try:
            return Conversation.from_json(raw)
        except (ValueError, KeyError, TypeError):
            logger.warning("Discarding unreadable conversation %s", conversation_id)
            return None

    async def put(self, conv: Conversation) -> None:
        conv.updated = time.time()
        await self.redis.set(self._key(conv.id), conv.to_json(), ex=self.ttl_seconds)

    async def acquire_lock(self, conversation_id: str, ttl_seconds: int) -> str:
        key = self._lock_key(conversation_id)
        value = _new_lock_value()
        if await self.redis.set(key, value, nx=True, ex=int(ttl_seconds)):
            return value
        raise ConversationBusy(_lock_since(await self.redis.get(key)))

    async def release_lock(self, conversation_id: str, value: str) -> None:
        # Compare-and-delete in a WATCH transaction: if the lock expired and
        # another follow-up took it meanwhile, it is not ours to delete.
        key = self._lock_key(conversation_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                current = await pipe.get(key)
                if isinstance(current, bytes):
                    current = current.decode("utf-8", errors="replace")
                if current != value:
                    await pipe.unwatch()
                    return
                pipe.multi()
                pipe.delete(key)
                await pipe.execute()
            except WatchError:
                pass

    async def ping(self) -> bool:
        try:
            return bool(await self.redis.ping())
        except Exception:  # noqa: BLE001 - any failure means not ready
            logger.warning("Redis ping failed", exc_info=True)
            return False

    async def close(self) -> None:
        await self.redis.aclose()


def create_store(settings: Settings) -> ConversationStore:
    if settings.REDIS_URL:
        return RedisConversationStore.from_url(
            settings.REDIS_URL,
            settings.CONVERSATION_TTL_SECONDS,
            settings.REDIS_KEY_PREFIX,
        )

    logger.warning(
        "REDIS_URL is not set: conversations are kept in process memory. "
        "Follow-ups will fail if more than one replica is running."
    )
    return MemoryConversationStore(
        settings.CONVERSATION_TTL_SECONDS, settings.MAX_CONVERSATIONS
    )
