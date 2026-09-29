from holmes_mcp import text


class TestSanitizeAnswer:
    def test_plain_text_unchanged(self):
        assert text.sanitize_answer("  the pod is OOMKilled  ") == "the pod is OOMKilled"

    def test_complete_think_block_removed(self):
        raw = "<think>let me look at the logs</think>\nThe pod is OOMKilled."
        assert text.sanitize_answer(raw) == "The pod is OOMKilled."

    def test_orphan_close_tag_drops_everything_before(self):
        raw = "reasoning the parser missed</think>The answer."
        assert text.sanitize_answer(raw) == "The answer."

    def test_orphan_open_tag_at_end_drops_everything_after(self):
        raw = "The answer.<think>unfinished reasoning cut by max_tokens"
        assert text.sanitize_answer(raw) == "The answer."

    def test_reasoning_only_returns_empty(self):
        assert text.sanitize_answer("<thinking>only thoughts</thinking>") == ""


class TestSplitLeakedAnswer:
    def test_no_close_tag(self):
        assert text.split_leaked_answer("just thinking") == ("just thinking", "", False)

    def test_close_tag_splits(self):
        head, tail, switched = text.split_leaked_answer("thinking</think>answer")
        assert (head, tail, switched) == ("thinking", "answer", True)

    def test_reopened_thinking(self):
        head, tail, switched = text.split_leaked_answer(
            "a</think>answer<think>more thinking"
        )
        assert tail == "answer"
        assert "more thinking" in head
        assert switched is False


class TestMergeStreamText:
    def test_deltas_are_appended(self):
        buf = ""
        for chunk in ["The ", "pod ", "crashed"]:
            buf, _ = text.merge_stream_text(buf, chunk)
        assert buf == "The pod crashed"

    def test_cumulative_snapshots_are_not_duplicated(self):
        buf = ""
        for chunk in ["The", "The pod", "The pod crashed"]:
            buf, _ = text.merge_stream_text(buf, chunk)
        assert buf == "The pod crashed"

    def test_exact_repeat_ignored(self):
        buf, new = text.merge_stream_text("same", "same")
        assert (buf, new) == ("same", "")

    def test_separator_between_whole_messages(self):
        buf, _ = text.merge_stream_text("first", "second", "\n\n")
        assert buf == "first\n\nsecond"


class TestChooseFinalAnswer:
    def test_analysis_preferred(self):
        assert text.choose_final_answer("analysis", "") == "analysis"

    def test_falls_back_to_streamed(self):
        assert text.choose_final_answer("", "streamed") == "streamed"

    def test_longer_wins_when_analysis_is_a_tail(self):
        streamed = "Root cause: memory limit too low. Fix: raise it to 512Mi."
        assert text.choose_final_answer("Fix: raise it to 512Mi.", streamed) == streamed

    def test_think_tags_stripped_from_both(self):
        assert text.choose_final_answer("<think>x</think>A", None) == "A"


class TestHistoryRecovery:
    def test_answer_from_last_assistant_content(self):
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "final answer"},
        ]
        assert text.extract_answer_from_history(history) == "final answer"

    def test_answer_leaked_into_reasoning_content(self):
        history = [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "thinking</think>the leaked answer",
            }
        ]
        assert text.extract_answer_from_history(history) == "the leaked answer"

    def test_reasoning_fallback(self):
        history = [{"role": "assistant", "content": "", "reasoning_content": "thoughts"}]
        assert text.extract_answer_from_history(history) == ""
        assert text.extract_reasoning_from_history(history) == "thoughts"

    def test_non_list_history(self):
        assert text.extract_answer_from_history(None) == ""


class TestTruncate:
    def test_short_text_unchanged(self):
        assert text.truncate("abc", 10) == "abc"

    def test_long_text_cut_with_note(self):
        out = text.truncate("x" * 1000, 100, "too big")
        assert len(out) <= 100
        assert out.endswith("…(truncated: too big)")
