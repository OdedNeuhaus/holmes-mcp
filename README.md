# holmes-mcp

An MCP server that lets Claude Code ask [HolmesGPT](https://github.com/robusta-dev/holmesgpt) questions about live systems and follow up on its answers.

HolmesGPT investigates by running its own read-only tools: Kubernetes, logs, Prometheus, alerts, Elasticsearch and so on. This server gives Claude Code access to it. When a developer asks "why is my service crash-looping in staging?", Claude can pass the question to HolmesGPT along with what it knows from the repository, then connect HolmesGPT's findings back to the code.

## For Claude Code users

Add the server once. It will then be available in every project:

```bash
claude mcp add --transport http --scope user holmesgpt \
  https://holmes-mcp.internal.example.com/mcp \
  --header "X-User-Email: $(git config user.email)" \
  --header "X-Username: $(whoami)" \
  --header "X-Hostname: $(hostname)"
```

Copy the command as it is. Your shell fills in your email, username and hostname when you run it, so there's nothing to edit. The HolmesGPT team uses these to see who uses HolmesGPT. They label usage; they aren't a password.

Then ask Claude in plain language, and it calls HolmesGPT when the question is about live systems:

```
Ask HolmesGPT why payments-api has been returning 502s in staging since this morning's deploy.
```

### What Claude gets

| Tool | What it does |
|------|--------------|
| `ask_holmes(question, context?)` | Starts a new HolmesGPT investigation. `context` carries what HolmesGPT can't see: service, namespace and cluster names, error text, manifest snippets, recent changes. |
| `holmes_follow_up(conversation_id, question)` | Continues an investigation. HolmesGPT keeps everything it already ran, including tool output. |

Each answer ends with a `conversation_id` and the list of tool calls HolmesGPT made, so Claude can see what evidence the answer rests on. Both tools are read-only, so you can allow-list them in `/permissions`.

### Long investigations

Investigations often take a few minutes. You don't need to configure anything for that:

- After 2 minutes, Claude Code moves the call to a background task (see `/tasks`), and you can keep working. The answer arrives when HolmesGPT finishes.
- The server sends a "still investigating" progress update every 30 seconds. That stops Claude Code's 5-minute idle timeout from cutting the call off.
- If you cancel (Esc, or stop the task in `/tasks`), HolmesGPT stops right away.
- If HolmesGPT hits the server's own time limit, you get what it found up to that point, clearly marked as incomplete.

### Troubleshooting

- **"Unknown or expired conversation_id".** Conversations expire after 2 hours without use. Ask again with `ask_holmes`.
- **"A follow-up on conversation … is already running".** Follow-ups on one conversation run one at a time. Wait for the running one to finish, then ask again. Separate questions can run in parallel with separate `ask_holmes` calls.
- **"The input is … characters".** HolmesGPT gets only the relevant parts: the failing resource's manifest and the exact error lines, not whole files or logs.
- **"HolmesGPT stopped responding mid-request".** A HolmesGPT tool, such as a slow Elasticsearch, hung for longer than the stall timeout. A narrower question usually avoids it.

## For operators

### Architecture

```
Claude Code ──HTTP (MCP, streamable, stateless)──▶ holmes-mcp (N replicas) ──SSE──▶ HolmesGPT /api/chat
                                                        │
                                                        └──▶ Redis (conversation histories, TTL)
```

- **Stateless MCP transport.** No MCP session state is kept on a replica, so any replica can serve any request.
- **Conversations live in Redis.** Each one is stored as `holmes-mcp:conv:<id>` and holds HolmesGPT's full `conversation_history`, including its tool calls and results. That is what makes follow-ups work on any replica and survive restarts. Reading a conversation refreshes its TTL.
- **History size is bounded.** Above `MAX_HISTORY_CHARS`, the largest stored tool outputs are shortened. Messages are never dropped, since that would break the pairing between each tool call and its result.
- **Degraded mode.** If Redis is down, new questions still return answers, with a note that follow-ups aren't available, and follow-ups fail with a clear message. `/readyz` reports the Redis state; `/healthz` does not check it.
- **One follow-up at a time per conversation.** A lock in Redis (`holmes-mcp:conv:<id>:lock`) rejects a second follow-up while one is running, so parallel calls can't silently overwrite each other's history. The lock expires by itself after the total timeout plus 60s, in case a replica dies while holding it.
- **Long calls stay alive.** Progress notifications every `HEARTBEAT_SECONDS` keep Claude Code's 5-minute idle timeout from firing. When a client cancels or disconnects, the HolmesGPT request is closed immediately and the cancellation is logged.
- **Rollouts drain.** On SIGTERM the pod keeps serving for 10s (`preStop`), so the ingress stops routing to it first. It then finishes in-flight investigations before exiting. `terminationGracePeriodSeconds` is derived from `holmes.totalTimeoutSeconds`, and an idle pod still exits in seconds.
- **Langfuse tracing (optional).** Every `ask_holmes` and `holmes_follow_up` call becomes a trace:
  - **Input and output:** the question and context in, the answer out.
  - **Child spans:** one per HolmesGPT tool, with its parameters and the start of its output.
  - **Metadata:** status (`success`, `error`, `cut_off`, `incomplete` or `cancelled`), duration and tool-call count.
  - **Session:** each HolmesGPT conversation is one Langfuse session, so a question and its follow-ups appear together.
  - **User:** `X-User-Email`, or else `X-Username@X-Hostname`. The client IP (from `X-Forwarded-For`) and user agent go into the metadata.
  - **Trace ID:** the same `trace_id` HolmesGPT receives, so if HolmesGPT's own LLM calls are ever sent to Langfuse, they nest inside these traces.
  - **Sending:** traces are sent in the background, in batches. Only the server's own spans are exported, and buffered traces are flushed on shutdown. If Langfuse is down, answers are unaffected.
- **Tool approval is disabled.** `enable_tool_approval` is never sent, so HolmesGPT works around approval-gated tools itself.
- The HolmesGPT model is always `GENERIC_MODEL_NAME`.

The logic that separates HolmesGPT's reasoning from its answer comes from the Open WebUI pipe. It handles leaked `<think>` tags, answers stranded in `reasoning_content`, empty `analysis`, and delta versus snapshot streams. It lives in `src/holmes_mcp/text.py`.

### Deploy (Helm)

```bash
docker build -t REGISTRY/holmes-mcp:0.1.0 . && docker push REGISTRY/holmes-mcp:0.1.0

helm upgrade --install holmes-mcp deploy/helm/holmes-mcp -n <holmes-namespace> \
  --set image.repository=REGISTRY/holmes-mcp \
  --set ingress.host=holmes-mcp.internal.example.com
```

The chart's [values.yaml](deploy/helm/holmes-mcp/values.yaml) is intentionally short:

| Value | Default | Purpose |
|-------|---------|---------|
| `replicaCount` | `2` | More than 1 requires Redis. The chart refuses to render without it |
| `image.repository` / `image.tag` | `REGISTRY/holmes-mcp` / appVersion | Image |
| `holmes.url` / `holmes.model` | `http://holmesgpt-holmes` / `generic` | HolmesGPT endpoint and model |
| `holmes.additionalSystemPrompt` | empty | Extra facts for HolmesGPT |
| `holmes.stallTimeoutSeconds` / `totalTimeoutSeconds` | `120` / `1800` | Timeouts |
| `conversationTtlSeconds` | `7200` | Follow-up window |
| `ingress.*` | nginx, internal host | Host, class, TLS; annotations for long SSE calls |
| `redis.enabled` | `true` | Bundled Redis with a generated password. Set to `false` with `redis.external.existingSecret` to use your own Redis |
| `langfuse.enabled` / `host` / `existingSecret` / `environment` | off | Langfuse tracing. The secret holds `public-key` and `secret-key` |
| `resources`, `nodeSelector`, `tolerations`, `extraEnv` | | The usual |

After install, `helm` prints the exact `claude mcp add` command for developers.

If you're behind ingress-nginx, keep the default ingress annotations. The long read timeout keeps tool calls open for the whole investigation, and turning buffering off lets progress notifications stream through. For another ingress controller, set the equivalent timeout and buffering options.

### Configuration (environment variables)

The chart sets these for you. The full list is for running the image some other way.

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOLMESGPT_URL` | `http://holmesgpt-holmes` | HolmesGPT base URL |
| `GENERIC_MODEL_NAME` | `generic` | HolmesGPT model sent with every request |
| `ADDITIONAL_SYSTEM_PROMPT` | empty | Extra system prompt for facts HolmesGPT can't discover |
| `REDIS_URL` | unset | Conversation store. **Required for more than one replica.** Unset means an in-memory store, for local development only |
| `REDIS_KEY_PREFIX` | `holmes-mcp:conv:` | Key prefix in Redis |
| `CONVERSATION_TTL_SECONDS` | `7200` | Idle time before a conversation expires |
| `MAX_HISTORY_CHARS` | `1000000` | Size cap for a stored conversation |
| `STALL_TIMEOUT_SECONDS` | `120` | Longest silence allowed from HolmesGPT. Must exceed its slowest tool call |
| `TOTAL_TIMEOUT_SECONDS` | `1800` | Ceiling for a single HolmesGPT request |
| `MAX_RESPONSE_CHARS` | `60000` | Cap on the text returned to Claude |
| `MAX_INPUT_CHARS` | `30000` | Largest question plus context accepted |
| `HEARTBEAT_SECONDS` | `30` | Interval of "still investigating" progress updates. Keep it well under 5 minutes |
| `USER_HEADER` | `X-User-Email` | Header carrying the caller's email, for attribution (not authentication). `X-Username` and `X-Hostname` are read too |
| `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | unset | Langfuse (self-hosted ≥ 3.63.0). Tracing is on only when all three are set |
| `LANGFUSE_ENVIRONMENT` | unset | Optional environment label in Langfuse, e.g. `production` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Listen address |
| `LOG_LEVEL` | `INFO` | |

Endpoints: `/mcp` (MCP), `/healthz` (liveness), `/readyz` (Redis check, for monitoring).

**Security.** The server has no per-user authentication. Expose it only on an internal ingress. The identity headers are supplied by the client, so treat them as labels, not as verified identities. The client IP in the trace metadata is the one value a developer can't easily set. Questions, context and answers are stored in Langfuse when tracing is on, including anything sensitive developers paste in, so restrict access to that Langfuse project accordingly.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e . pytest pytest-asyncio respx fakeredis
.venv/bin/python -m pytest

# Run locally against a port-forwarded HolmesGPT:
kubectl port-forward svc/holmesgpt-holmes 8080:80
HOLMESGPT_URL=http://localhost:8080 .venv/bin/holmes-mcp
claude mcp add --transport http holmesgpt-dev http://localhost:8000/mcp
```

With `uv`, `uv sync && uv run pytest` also works.
