"""Regression checks for citation spans, conjunctions, and per-call retries."""
import unittest
from types import SimpleNamespace

from arena.tools import ToolResult
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.retry import Retry


def context(docs):
    observed = "\n".join(doc.body for doc in docs)
    return SimpleNamespace(
        corpus=SimpleNamespace(docs=docs, get=lambda key: next(
            (doc for doc in docs if doc.doc_id == key), None)),
        observed_text=observed, saw=lambda text: bool(text) and text in observed,
    )


class LayerRegressions(unittest.TestCase):
    def test_reattribute_substring_of_one_line(self):
        doc = SimpleNamespace(doc_id="source", body="Prefix: quoted evidence.\nOther line.")
        report = {"claims": [{"text": "quoted evidence", "doc_id": "wrong"}]}
        result = CitationChecker().after_agent(context([doc]), report)
        self.assertEqual(result["claims"][0], {"text": "quoted evidence", "doc_id": "source"})
        self.assertEqual(result["citations"], ["source"])

    def test_split_join_with_internal_conjunction(self):
        docs = [SimpleNamespace(doc_id="a", body="A và B."),
                SimpleNamespace(doc_id="b", body="C và D.")]
        report = {"claims": [{"text": "A và B. và C và D.", "doc_id": "a"}]}
        result = Critic().after_agent(context(docs), report)
        self.assertEqual([c["text"] for c in result["claims"]], ["A và B.", "C và D."])
        self.assertTrue(result["abstain"])

    def test_retries_reset_for_each_call(self):
        ctx = SimpleNamespace(state={}, tools=SimpleNamespace(calls=0), max_tool_calls=None)
        def call(name, args):
            ctx.tools.calls += 1
            return ToolResult(ok=False, content="", error="timeout")
        layer = Retry(max_attempts=3)
        layer.wrap_tool_call(ctx, call, "fetch_doc", {})
        layer.wrap_tool_call(ctx, call, "fetch_doc", {})
        self.assertEqual(ctx.tools.calls, 6)
        self.assertEqual(ctx.state["retry_attempts"], 3)

    def test_retry_preserves_submit_reserve(self):
        ctx = SimpleNamespace(state={}, tools=SimpleNamespace(calls=5), max_tool_calls=8)
        def call(name, args):
            ctx.tools.calls += 1
            return ToolResult(ok=True, content="[TRUNCATED: partial]")
        result = Retry().wrap_tool_call(ctx, call, "fetch_doc", {})
        self.assertEqual(ctx.tools.calls, 7)
        self.assertIn("[TRUNCATED:", result.content)


if __name__ == "__main__":
    unittest.main()
