import unittest
from types import SimpleNamespace as NS

from kb.trace import Trace, extract_reasoning, format_trace, strip_think


class Reasoning(unittest.TestCase):
    def test_separate_field(self):
        text, src = extract_reasoning(NS(content="answer", reasoning_content="step 1, step 2", model_extra=None))
        self.assertEqual((text, src), ("step 1, step 2", "message.reasoning_content"))

    def test_other_field_names_and_model_extra(self):
        self.assertEqual(extract_reasoning(NS(content="x", reasoning="why", model_extra=None))[1], "message.reasoning")
        text, src = extract_reasoning(NS(content="x", model_extra={"thinking": "hmm"}))
        self.assertEqual((text, src), ("hmm", "message.thinking"))

    def test_think_tags_in_content(self):
        msg = NS(content="<think>count the rows</think>\nThere are 3.", model_extra=None)
        text, src = extract_reasoning(msg)
        self.assertEqual(text, "count the rows")
        self.assertIn("<think>", src)
        self.assertEqual(strip_think(msg.content), "There are 3.")

    def test_nothing_returned(self):
        self.assertEqual(extract_reasoning(NS(content="plain", model_extra=None)), (None, None))


class Rendering(unittest.TestCase):
    def test_events_get_timestamps_and_render(self):
        tr = Trace("how many films?", {"llm_model": "m"})
        tr.event("retrieve", question="q", k=2, ms=5, hits=[
            {"id": 1, "kind": "schema", "source": "mysql://sakila", "page": 0, "table": "film", "score": 0.03, "expanded": False, "preview": "x"},
            {"id": 2, "kind": "schema", "source": "mysql://sakila", "page": 0, "table": "film_category", "score": 0.0, "expanded": True, "preview": "y"}])
        tr.event("llm_call", purpose="route", model="m", messages=[{"role": "user", "content": "hi"}], reply_raw="r",
                 reasoning="because", reasoning_source="message.reasoning", finish_reason="length", usage=None, ms=3)
        tr.event("llm_call", purpose="summarise", model="m", messages=[{"role": "user", "content": "hi"}], reply_raw="r",
                 reasoning=None, reasoning_source=None, finish_reason="stop", usage={"prompt": 1}, ms=3)
        tr.event("decision", route="sql", why="reply had SQL")
        tr.event("sql_attempt", attempt=1, sql_in="SELECT 1", outcome="ok", executed_sql="SELECT 1 LIMIT 5", error=None,
                 n_rows=1, columns=["1"], rows_preview=[[1]], ms=2)
        self.assertTrue(all("t_ms" in e for e in tr.events))
        out = format_trace({"id": 1, "created_at": "2026-11-14 10:00:00", "question": "q", "outcome": "answered_with_sql",
                            "total_ms": 9, "meta": {}, "events": tr.events, "final_sql": "SELECT 1", "answer": "one", "error": None})
        for needle in ["added via foreign key", "THINKING (from message.reasoning)", "none returned", "SQL TRY #1",
                       "finish_reason=length", "route=sql"]:
            self.assertIn(needle, out)


if __name__ == "__main__":
    unittest.main()
