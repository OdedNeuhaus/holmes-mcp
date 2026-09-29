"""
Text helpers for turning HolmesGPT output into a clean answer.

Ported from the Open WebUI HolmesGPT pipe. The pipe's UI-only parts
(<details> blocks, status notices) are dropped; the logic that separates
reasoning from the answer is kept as is, because the same model quirks
show up no matter which client asks.
"""

import re

# Tag names a reasoning model can wrap its chain-of-thought in. vLLM's
# --reasoning-parser is supposed to peel these off into `reasoning_content`,
# but a missing/partial parser (or a turn where the model never emits the
# closing tag) leaks them into normal content.
THINK_TAGS = "think|thinking|reason|reasoning|thought"

# A complete, balanced reasoning block.
THINK_BLOCK_RE = re.compile(
    rf"<({THINK_TAGS})\b[^>]*>.*?</\1\s*>\s*",
    re.DOTALL | re.IGNORECASE,
)
THINK_OPEN_RE = re.compile(rf"<(?:{THINK_TAGS})\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(rf"</(?:{THINK_TAGS})\s*>", re.IGNORECASE)


def content_to_text(content) -> str:
    """Flatten a message `content` (string or list of typed parts) to text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)

    return "" if content is None else str(content)


def sanitize_reasoning(text: str) -> str:
    """Remove think-tag markup from reasoning text while keeping the prose."""
    if not text:
        return ""

    text = THINK_OPEN_RE.sub("", text)
    text = THINK_CLOSE_RE.sub("", text)
    return text.strip()


def sanitize_answer(text: str) -> str:
    """
    Strip reasoning that the model inlined into its answer text.

    Three shapes show up with DeepSeek-style models behind vLLM:

    * a complete `<think>...</think>` block (no reasoning parser at all);
    * an orphan `</think>` because the parser consumed the opening tag but
      not the closing one - everything before it is reasoning;
    * an orphan `<think>` at the end because the turn hit max_tokens -
      everything after it is reasoning.

    Returning "" is a valid outcome: the caller then falls through to the
    next answer candidate rather than returning chain-of-thought.
    """
    if not text:
        return ""

    text = THINK_BLOCK_RE.sub("", text)

    closes = list(THINK_CLOSE_RE.finditer(text))
    if closes:
        text = text[closes[-1].end():]

    opens = THINK_OPEN_RE.search(text)
    if opens:
        text = text[: opens.start()]

    return text.strip()


def split_leaked_answer(text: str) -> tuple[str, str, bool]:
    """
    Split `reasoning_content` at a leaked closing think tag.

    On long turns vLLM sometimes fails to switch from `reasoning_content` to
    `content` when the model closes its thinking block: the closing tag
    arrives *inside* `reasoning_content` and every answer token after it
    stays on the reasoning channel. The tag is the one marker saying where
    the answer starts, so split on it.

    Returns (reasoning_part, answer_part, switched). `switched` reports that
    the model closed its thinking block and did not reopen it.
    """
    if not text:
        return "", "", False

    closes = list(THINK_CLOSE_RE.finditer(text))
    if not closes:
        return text, "", False

    last = closes[-1]
    head, tail = text[: last.start()], text[last.end():]

    # A fresh `<think>` after the close means the model went back to
    # thinking; only what sits between the two tags is answer text.
    reopen = THINK_OPEN_RE.search(tail)
    if reopen:
        return (
            head + "\n" + tail[reopen.end():],
            tail[: reopen.start()],
            False,
        )

    return head, tail, True


def merge_stream_text(
    buffer: str,
    chunk: str,
    separator: str = "",
) -> tuple[str, str]:
    """
    Merge one `ai_message` text payload into an accumulating buffer.

    Deployments disagree on what that payload holds: a token delta, one whole
    assistant message, or a cumulative snapshot of the message so far. Only
    an exact repeat or a strict extension of the buffer is treated as a
    snapshot; anything else is appended.

    Returns (new_buffer, newly_appended_text).
    """
    if not chunk:
        return buffer, ""
    if not buffer:
        return chunk, chunk
    if chunk == buffer:
        return buffer, ""
    if chunk.startswith(buffer):
        return chunk, chunk[len(buffer):]

    addition = chunk
    if separator and not buffer.endswith(("\n", " ")):
        addition = separator + chunk
    return buffer + addition, addition


def _normalize_for_compare(text: str) -> str:
    return " ".join(text.lower().split())


def choose_final_answer(analysis, streamed_content) -> str:
    """
    Decide which text is the answer.

    `ai_answer_end.analysis` is authoritative when it is whole, but with a
    reasoning model it regularly arrives empty or holding only the tail of an
    answer. When one candidate is contained in the other, the longer one is
    the real answer.
    """
    a = sanitize_answer(content_to_text(analysis))
    b = sanitize_answer(content_to_text(streamed_content))

    if not a:
        return b

    if not b or a == b:
        return a

    norm_a = _normalize_for_compare(a)
    norm_b = _normalize_for_compare(b)

    if norm_a and norm_a in norm_b:
        return b
    if norm_b and norm_b in norm_a:
        return a

    return a


def _assistant_messages(history) -> list:
    if not isinstance(history, list):
        return []

    return [
        msg
        for msg in history
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    ]


def extract_answer_from_history(history) -> str:
    """
    Recover the answer from a conversation_history when `analysis` is empty.

    The `content` channel is preferred. Text sitting after a `</think>`
    inside `reasoning_content` is the one exception: it is not
    chain-of-thought, it is the answer vLLM failed to move onto `content`.
    """
    messages = _assistant_messages(history)

    for msg in reversed(messages):
        text = sanitize_answer(content_to_text(msg.get("content")))
        if text:
            return text

    for msg in reversed(messages):
        _, leaked, _ = split_leaked_answer(
            content_to_text(msg.get("reasoning_content"))
        )
        leaked = sanitize_answer(leaked)
        if leaked:
            return leaked

    return ""


def extract_reasoning_from_history(history) -> str:
    """Last-resort text for a turn that produced no answer at all."""
    for msg in reversed(_assistant_messages(history)):
        head, _, _ = split_leaked_answer(
            content_to_text(msg.get("reasoning_content"))
        )
        text = sanitize_reasoning(head)
        if text:
            return text

    return ""


def label_recovered_reasoning(text: str) -> str:
    return (
        "*(HolmesGPT returned no separate answer; the text below was "
        "recovered from the model's reasoning.)*\n\n"
        f"{text}"
    )


def truncate(text: str, limit: int, note: str = "") -> str:
    """Cut `text` to at most `limit` characters, appending `note` when cut."""
    if limit <= 0 or len(text) <= limit:
        return text

    suffix = f"\n\n…(truncated{': ' + note if note else ''})"
    return text[: max(0, limit - len(suffix))].rstrip() + suffix
