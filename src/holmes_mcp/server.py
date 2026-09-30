"""HolmesGPT MCP server: lets Claude Code ask HolmesGPT questions."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Annotated

import uvicorn
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from holmes_mcp import text
from holmes_mcp.config import Settings, get_settings
from holmes_mcp.conversations import (
    Conversation,
    ConversationBusy,
    ConversationStore,
    create_store,
    is_valid_conversation_id,
    new_conversation_id,
    shrink_history,
)
from holmes_mcp.holmes_client import HolmesClient, HolmesResult, ToolCall
from holmes_mcp.tracing import TraceInfo, Tracer, create_tracer

logger = logging.getLogger(__name__)

SERVER_NAME = "holmesgpt"
VERSION = "0.1.0"

# Used only when HolmesGPT's answer event carries no conversation_history,
# mirroring the Open WebUI pipe's default.
FALLBACK_SYSTEM_PROMPT = "You are a helpful kubernetes troubleshooting assistant"

# Tool calls listed individually in a result; the rest are counted.
MAX_LISTED_TOOL_CALLS = 40

INSTRUCTIONS = """\
HolmesGPT is the organization's AI troubleshooting agent. It investigates \
LIVE production and cluster state by running its own read-only tools: \
Kubernetes (pods, deployments, events, restarts, OOMKills), logs, Prometheus \
metrics, alerts, Elasticsearch and other observability sources.

Use HolmesGPT when the question is about how something behaves at runtime - \
"why is X crashing / slow / erroring in the cluster", "what do the logs say", \
"is this alert related to my change" - rather than about the code itself.

How to get good answers:
- HolmesGPT cannot see the local repository. Pass what you know in `context`: \
service / deployment / namespace / cluster names, the exact error text, \
relevant manifest or config snippets, and what recently changed.
- Investigations are slow (often one to several minutes). Ask one \
well-scoped question instead of several quick ones.
- To dig deeper, call `holmes_follow_up` with the returned conversation_id. \
HolmesGPT keeps everything it already ran, so do not start over.
- If HolmesGPT asks a question or says it needs more information (which \
namespace, cluster, service, time window...), look for the answer in the \
workspace first and ask the user only if you cannot find it. Then reply with \
`holmes_follow_up` on the same conversation_id.
- Run follow-ups on one conversation one at a time; a second follow-up \
while one is running is rejected. Separate questions can use separate \
ask_holmes calls in parallel.
- Treat HolmesGPT's conclusions as evidence to check against the code, not \
as ground truth. Tell the user what HolmesGPT found and which tools it ran.
"""

READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)


def _clean_user_id(value: str | None) -> str | None:
    """Langfuse user values must be US-ASCII strings under 200 chars."""
    if not value:
        return None
    cleaned = value.encode("ascii", "ignore").decode("ascii").strip()
    return cleaned[:199] or None


# Self-reported identity headers, set once by each developer with
# `claude mcp add ... --header "X-Username: $(whoami)" ...`. They label usage;
# they are not authentication.
USERNAME_HEADER = "X-Username"
HOSTNAME_HEADER = "X-Hostname"


@dataclass
class Caller:
    """Who is calling, as far as the request tells us."""

    email: str | None = None
    username: str | None = None
    hostname: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None

    @property
    def user_id(self) -> str | None:
        """The label used for attribution: email, else username@hostname."""
        if self.email:
            return self.email
        if self.username and self.hostname:
            return f"{self.username}@{self.hostname}"
        return self.username

    def trace_metadata(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "email": self.email,
                "username": self.username,
                "hostname": self.hostname,
                "client_ip": self.client_ip,
                "user_agent": self.user_agent,
            }.items()
            if value
        }


def _caller_from_request(request, email_header: str) -> Caller:
    headers = getattr(request, "headers", None) or {}
    forwarded = (headers.get("x-forwarded-for") or "").split(",")[0].strip()
    client = getattr(request, "client", None)
    return Caller(
        email=_clean_user_id(headers.get(email_header)),
        username=_clean_user_id(headers.get(USERNAME_HEADER)),
        hostname=_clean_user_id(headers.get(HOSTNAME_HEADER)),
        client_ip=forwarded
        or headers.get("x-real-ip")
        or (getattr(client, "host", None) if client else None),
        user_agent=_clean_user_id(headers.get("user-agent")),
    )


def _build_metadata(conversation_id: str, user_id: str | None, turn: int) -> dict:
    metadata = {
        "session_id": conversation_id,
        "trace_id": uuid.uuid4().hex,
        "trace_name": "claude-code-holmesgpt",
        "tags": ["claude-code", "holmesgpt", "mcp"],
        "trace_metadata": {
            "source": "claude-code",
            "mcp_server": "holmes-mcp",
            "turn": turn,
        },
    }
    if user_id:
        metadata["trace_user_id"] = user_id
    return metadata


def _compose_ask(question: str, context: str | None) -> str:
    question = question.strip()
    if context and context.strip():
        return (
            f"{question}\n\nContext from the developer's workspace:\n"
            f"{context.strip()}"
        )
    return question


def _next_history(previous: list | None, ask: str, result: HolmesResult) -> list:
    """The history to store for the next follow-up."""
    if result.conversation_history:
        return result.conversation_history

    history = list(previous or [])
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": FALLBACK_SYSTEM_PROMPT})
    history.append({"role": "user", "content": ask})
    history.append({"role": "assistant", "content": result.answer})
    return history


def _format_tool_line(call: ToolCall) -> str:
    params = json.dumps(call.params, ensure_ascii=False, separators=(",", ":"))
    line = f"- {call.name} {text.truncate(params, 200)}"
    if call.status.lower() in ("error", "failed", "failure"):
        preview = " ".join(call.preview.split())
        line += f" -> failed: {text.truncate(preview, 200)}"
    return line


def _tools_section(result: HolmesResult) -> list[str]:
    if not result.tools_used:
        return [
            f"HolmesGPT answered in {result.duration_s:.0f}s without running tools."
        ]
    lines = [
        f"HolmesGPT ran {len(result.tools_used)} tool call(s) in "
        f"{result.duration_s:.0f}s:"
    ]
    lines.extend(
        _format_tool_line(c) for c in result.tools_used[:MAX_LISTED_TOOL_CALLS]
    )
    hidden = len(result.tools_used) - MAX_LISTED_TOOL_CALLS
    if hidden > 0:
        lines.append(f"- ... and {hidden} more")
    return lines


def format_error(result: HolmesResult, max_chars: int, note: str = "") -> str:
    """The message for a failed HolmesGPT call, with whatever it found first."""
    parts = [result.error or "HolmesGPT failed."]
    if result.answer:
        label = (
            "Partial findings before HolmesGPT was cut off (not a final answer)"
            if result.incomplete
            else "Partial answer before the failure"
        )
        parts.append(f"{label}:\n\n{result.answer}")
    if result.tools_used:
        parts.append("\n".join(_tools_section(result)))
    if note:
        parts.append(note)
    return text.truncate("\n\n".join(parts), max_chars)


def format_result(
    result: HolmesResult,
    conversation_id: str | None,
    max_chars: int,
    stale_conversation_id: str | None = None,
) -> str:
    """
    `conversation_id` is None when saving failed. `stale_conversation_id` is
    then the follow-up's existing conversation, whose stored history no longer
    includes this exchange.
    """
    footer_lines = ["", "---"]
    if conversation_id:
        footer_lines.append(
            f"conversation_id: {conversation_id} "
            "(pass it to holmes_follow_up to continue this investigation)"
        )
        footer_lines.append(
            "If HolmesGPT asked a question or needs more information, find the "
            "answer in the workspace (ask the user only if you cannot), then "
            "reply with holmes_follow_up."
        )
    elif stale_conversation_id:
        footer_lines.append(
            "This exchange could not be saved, so follow-ups are not available. "
            f"Do not use holmes_follow_up with {stale_conversation_id} again: "
            "its stored history is missing this exchange. To continue, start a "
            "new conversation with ask_holmes and pass the findings so far as "
            "context."
        )
    else:
        footer_lines.append(
            "This conversation could not be saved, so follow-ups are not "
            "available for it. Use ask_holmes again to continue."
        )

    footer_lines.extend(_tools_section(result))
    footer = "\n".join(footer_lines)

    answer = result.answer
    if result.incomplete:
        answer += (
            "\n\n*(HolmesGPT ended the stream without a final analysis - the "
            "answer above may be incomplete.)*"
        )

    budget = max(1000, max_chars - len(footer))
    return text.truncate(answer, budget, "answer exceeded the size limit") + footer


def create_server(
    settings: Settings | None = None,
    client: HolmesClient | None = None,
    store: ConversationStore | None = None,
    tracer: Tracer | None = None,
) -> MCPServer:
    # Explicit None checks: an empty MemoryConversationStore has len() == 0
    # and would be falsy.
    if settings is None:
        settings = get_settings()
    if client is None:
        client = HolmesClient(settings)
    if store is None:
        store = create_store(settings)
    if tracer is None:
        tracer = create_tracer(settings)

    mcp = MCPServer(
        SERVER_NAME,
        title="HolmesGPT",
        instructions=INSTRUCTIONS,
        version=VERSION,
        log_level=settings.LOG_LEVEL,
    )
    # Exposed for tests, the health routes and shutdown.
    mcp.conversation_store = store  # type: ignore[attr-defined]
    mcp.tracer = tracer  # type: ignore[attr-defined]

    def caller(ctx: Context) -> Caller:
        try:
            request = ctx.request_context.request
        except ValueError:
            return Caller()
        return _caller_from_request(request, settings.USER_HEADER)

    def progress_reporter(ctx: Context):
        count = 0

        async def report(message: str) -> None:
            nonlocal count
            count += 1
            try:
                await ctx.report_progress(count, None, message)
            except Exception:  # noqa: BLE001 - progress is best-effort
                logger.debug("Progress notification failed", exc_info=True)

        return report

    def validate_input(question: str, context: str | None = None) -> str:
        question = question.strip()
        if not question:
            raise ToolError("question must not be empty.")
        size = len(question) + len((context or "").strip())
        if size > settings.MAX_INPUT_CHARS:
            raise ToolError(
                f"The input is {size:,} characters (question {len(question):,}, "
                f"context {len((context or '').strip()):,}); the limit is "
                f"{settings.MAX_INPUT_CHARS:,}. Include only the relevant parts, "
                "e.g. the failing resource's manifest and the exact error lines, "
                "not whole files or logs."
            )
        return question

    def store_error(exc: Exception, action: str) -> ToolError:
        logger.exception("Conversation store %s failed", action)
        return ToolError(
            f"The HolmesGPT MCP server could not {action} its conversation store "
            f"({type(exc).__name__}). This is a server-side problem; retry "
            "later or start a new conversation with ask_holmes."
        )

    async def load(conversation_id: str) -> Conversation | None:
        try:
            return await store.get(conversation_id)
        except Exception as exc:  # noqa: BLE001
            raise store_error(exc, "read") from exc

    async def acquire(conversation_id: str) -> str:
        # Outlives the longest possible follow-up, so it cannot expire while
        # one is still running; it still expires if a replica dies holding it.
        ttl = int(settings.TOTAL_TIMEOUT_SECONDS) + 60
        try:
            return await store.acquire_lock(conversation_id, ttl)
        except ConversationBusy as busy:
            # max(): the lock's start time is stored rounded to milliseconds,
            # so it can sit a hair after "now".
            running = (
                f" (started {max(0.0, time.time() - busy.since):.0f}s ago)"
                if busy.since
                else ""
            )
            raise ToolError(
                f"A follow-up on conversation {conversation_id} is already "
                f"running{running}. Wait for it to finish, then ask again: "
                "HolmesGPT will have that answer in its context."
            ) from None
        except Exception as exc:  # noqa: BLE001
            raise store_error(exc, "lock") from exc

    async def release(conversation_id: str, lock: str) -> None:
        try:
            await store.release_lock(conversation_id, lock)
        except Exception:  # noqa: BLE001 - the lock expires by itself
            logger.warning(
                "Could not release the lock on %s; it will expire", conversation_id,
                exc_info=True,
            )

    async def save(conv: Conversation) -> bool:
        conv.history, shrunk = shrink_history(conv.history, settings.MAX_HISTORY_CHARS)
        if shrunk:
            logger.info(
                "Shortened %d tool output(s) in conversation %s", shrunk, conv.id
            )
        try:
            await store.put(conv)
            return True
        except Exception:  # noqa: BLE001 - the answer must still be returned
            logger.exception("Conversation store write failed for %s", conv.id)
            return False

    async def run(
        ctx: Context,
        tool: str,
        conversation_id: str,
        ask: str,
        trace_input: dict,
        previous: Conversation | None,
    ) -> str:
        who = caller(ctx)
        user_id = who.user_id or (previous.user_id if previous else None)
        turn = previous.turns + 1 if previous else 1
        metadata = _build_metadata(conversation_id, user_id, turn)

        logger.info(
            "Asking HolmesGPT: conversation=%s turn=%d user=%s ip=%s",
            conversation_id,
            turn,
            user_id or "-",
            who.client_ip or "-",
        )
        info = TraceInfo(
            name=tool,
            trace_id=metadata["trace_id"],
            session_id=conversation_id,
            user_id=user_id or "unknown",
            turn=turn,
            input=trace_input,
            metadata=who.trace_metadata(),
        )
        with tracer.trace(info) as trace:
            result = await client.ask(
                ask,
                session_id=conversation_id,
                metadata=metadata,
                conversation_history=previous.history if previous else None,
                user_id=user_id,
                on_progress=progress_reporter(ctx),
                observer=trace,
            )
            reply, saved = await finish(conversation_id, ask, previous, user_id, turn, result)
            trace.finish(
                status=(
                    ("cut_off" if result.incomplete else "error")
                    if result.error
                    else ("incomplete" if result.incomplete else "success")
                ),
                output=result.answer or None,
                error=result.error,
                metadata={
                    "conversation_id": conversation_id,
                    "duration_s": round(result.duration_s, 1),
                    "tool_calls": len(result.tools_used),
                    "saved": saved,
                },
            )
        if isinstance(reply, ToolError):
            raise reply
        return reply

    async def finish(
        conversation_id: str,
        ask: str,
        previous: Conversation | None,
        user_id: str | None,
        turn: int,
        result: HolmesResult,
    ) -> tuple[str | ToolError, bool]:
        """Store the outcome and build the reply (an error is returned, not raised,
        so the trace can record it before it propagates)."""
        if result.error:
            logger.info(
                "HolmesGPT failed: conversation=%s error=%s",
                conversation_id,
                result.error,
            )
            note = (
                f"Conversation {conversation_id} is unchanged; you can retry the "
                "follow-up."
                if previous
                else ""
            )
            return ToolError(format_error(result, settings.MAX_RESPONSE_CHARS, note)), False

        history = _next_history(previous.history if previous else None, ask, result)
        if previous:
            conv = previous
            conv.history = history
            conv.turns = turn
            conv.user_id = user_id
        else:
            conv = Conversation(id=conversation_id, history=history, user_id=user_id)
        saved = await save(conv)

        return (
            format_result(
                result,
                conversation_id if saved else None,
                settings.MAX_RESPONSE_CHARS,
                stale_conversation_id=conversation_id if previous and not saved else None,
            ),
            saved,
        )

    @mcp.tool(
        title="Ask HolmesGPT",
        annotations=READ_ONLY,
        structured_output=False,
    )
    async def ask_holmes(
        ctx: Context,
        question: Annotated[
            str,
            Field(
                description=(
                    "The question for HolmesGPT, about live runtime state: "
                    "e.g. 'Why is the payments-api deployment in namespace "
                    "payments crash-looping?'"
                ),
                min_length=1,
            ),
        ],
        context: Annotated[
            str | None,
            Field(
                description=(
                    "What you know from the developer's workspace that "
                    "HolmesGPT cannot see: service/deployment/namespace/"
                    "cluster names, exact error text, relevant manifest or "
                    "config snippets, recent changes."
                ),
            ),
        ] = None,
    ) -> str:
        """Start a new HolmesGPT investigation and return its answer.

        HolmesGPT runs its own read-only tools against the cluster and
        observability stack (Kubernetes, logs, metrics, alerts), which can
        take a few minutes. The result ends with a conversation_id to use
        with holmes_follow_up.
        """
        question = validate_input(question, context)
        trace_input = {"question": question}
        if context and context.strip():
            trace_input["context"] = context.strip()
        return await run(
            ctx,
            "ask_holmes",
            new_conversation_id(),
            _compose_ask(question, context),
            trace_input,
            None,
        )

    @mcp.tool(
        title="Follow up with HolmesGPT",
        annotations=READ_ONLY,
        structured_output=False,
    )
    async def holmes_follow_up(
        ctx: Context,
        conversation_id: Annotated[
            str,
            Field(description="The conversation_id returned by ask_holmes."),
        ],
        question: Annotated[
            str,
            Field(
                description=(
                    "The follow-up question. HolmesGPT still has everything "
                    "from earlier in this conversation, including the tools "
                    "it ran and their output."
                ),
                min_length=1,
            ),
        ],
    ) -> str:
        """Continue an existing HolmesGPT investigation with a follow-up question."""
        conversation_id = conversation_id.strip().lower()
        question = validate_input(question)
        unknown = ToolError(
            f"Unknown or expired conversation_id '{conversation_id}'. "
            "Conversations expire after "
            f"{settings.CONVERSATION_TTL_SECONDS / 60:.0f} minutes of "
            "inactivity. Start a new one with ask_holmes and restate the "
            "relevant context."
        )
        if not is_valid_conversation_id(conversation_id):
            raise unknown

        # Lock before loading, so a follow-up that waited on another one sees
        # the history that one saved.
        lock = await acquire(conversation_id)
        try:
            previous = await load(conversation_id)
            if previous is None:
                raise unknown
            return await run(
                ctx,
                "holmes_follow_up",
                conversation_id,
                question,
                {"question": question},
                previous,
            )
        finally:
            await release(conversation_id, lock)

    @mcp.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def healthz(request: Request) -> JSONResponse:
        # Liveness: the process is serving. Deliberately ignores Redis, so a
        # Redis outage does not make Kubernetes restart every replica.
        return JSONResponse({"status": "ok"})

    @mcp.custom_route("/readyz", methods=["GET"], include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:
        ok = await store.ping()
        return JSONResponse(
            {"status": "ok" if ok else "unavailable", "store": store.backend},
            status_code=200 if ok else 503,
        )

    return mcp


def create_app(settings: Settings | None = None):
    settings = settings or get_settings()
    mcp = create_server(settings)
    # Stateless: no MCP session state lives on a replica, so any replica can
    # serve any request. Conversation state lives in the shared store.
    app = mcp.streamable_http_app(stateless_http=True, host=settings.HOST)
    return _FlushTracesOnShutdown(app, mcp.tracer)


class _FlushTracesOnShutdown:
    """
    ASGI wrapper that sends buffered traces during the app's shutdown.

    It has to happen here: after a graceful SIGTERM shutdown uvicorn re-raises
    the signal, which ends the process before code after `uvicorn.run()` runs.
    """

    def __init__(self, app, tracer: Tracer):
        self.app = app
        self.tracer = tracer

    def __getattr__(self, name):
        # Let callers (tests, TestClient) reach the wrapped Starlette app.
        return getattr(self.app, name)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "lifespan":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "lifespan.shutdown.complete":
                await asyncio.to_thread(self.tracer.shutdown)
            await send(message)

        await self.app(scope, receive, send_wrapper)


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        create_app(settings),
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        # On SIGTERM, let in-flight investigations finish (bounded).
        timeout_graceful_shutdown=int(settings.TOTAL_TIMEOUT_SECONDS) + 30,
    )


if __name__ == "__main__":
    main()
