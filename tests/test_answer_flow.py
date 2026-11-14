"""Runs the real pipeline code in kb/answer.py against fakes (no DB, no model, no network)."""
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

HEAVY = ["openai", "psycopg", "psycopg.types", "psycopg.types.json", "pgvector", "pgvector.psycopg",
         "sentence_transformers", "pymysql", "numpy", "dotenv"]

HITS = [
    {"id": 1, "kind": "text", "filename": "sakila-en_a4.pdf", "page": 8, "score": 0.03, "metadata": {}, "content": "film_actor links films and actors"},
    {"id": 2, "kind": "schema", "filename": "mysql://sakila", "page": 0, "score": 0.02, "metadata": {"table": "rental"}, "content": "[MySQL table sakila.rental]"},
]


def reply(content, reasoning=None, finish="stop"):
    msg = NS(content=content, model_extra=None)
    if reasoning:
        msg.reasoning_content = reasoning
    return NS(choices=[NS(message=msg, finish_reason=finish)],
              usage=NS(prompt_tokens=10, completion_tokens=5, completion_tokens_details=None))


class Flow(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.dict(sys.modules, {n: MagicMock() for n in HEAVY})
        self.patcher.start()
        for name in [n for n in sys.modules if n.startswith("kb.")]:
            sys.modules.pop(name)                       # re-import kb.* against the fakes; restored in tearDown
        import kb.answer as answer
        import kb.mysql_schema as ms
        self.answer, self.ms = answer, ms
        self.saved = {}

        def fake_save(trace, outcome, answer=None, sql=None, error=None):
            self.saved.update(outcome=outcome, events=trace.events, sql=sql, error=error)
            return 7
        answer.Trace.save = fake_save
        answer.retrieve = lambda q, k: HITS

    def tearDown(self):
        self.patcher.stop()

    def fake_llm(self, *replies):
        client = MagicMock()
        client.chat.completions.create.side_effect = list(replies)
        self.answer._client = lambda: client

    def stages(self):
        return [e["stage"] for e in self.saved["events"]]

    def test_docs_route_records_thinking(self):
        self.fake_llm(reply("film_actor links them [sakila-en_a4.pdf p.8]", reasoning="the question is about design"))
        a = self.answer.ask("Which table links films and actors?")
        self.assertEqual(a.trace_id, 7)
        self.assertEqual(self.saved["outcome"], "answered_from_docs")
        self.assertEqual(self.stages(), ["retrieve", "llm_call", "decision"])
        call = self.saved["events"][1]
        self.assertEqual(call["reasoning"], "the question is about design")
        self.assertEqual(call["reasoning_source"], "message.reasoning_content")
        self.assertEqual(call["usage"], {"prompt": 10, "completion": 5})

    def test_sql_route_with_guard_rejection_then_repair(self):
        self.fake_llm(reply("<think>count rentals</think>\n```sql\nSELECT * FROM rentals\n```"),
                      reply("```sql\nSELECT COUNT(*) AS n FROM rental\n```"),
                      reply("There are 16044 rentals."))
        calls = []

        def fake_run(sql, max_rows):
            calls.append(sql)
            if len(calls) == 1:
                raise self.answer.UnsafeSQL("unknown table 'rentals'. Available tables: rental")
            return ["n"], [[16044]], False, "SELECT COUNT(*) AS n FROM rental LIMIT 101"
        self.ms.run_select = fake_run

        a = self.answer.ask("How many rentals are there?")
        self.assertEqual(self.saved["outcome"], "answered_with_sql")
        self.assertEqual(self.stages(), ["retrieve", "llm_call", "decision", "sql_attempt", "llm_call", "sql_attempt", "llm_call"])
        first, second = [e for e in self.saved["events"] if e["stage"] == "sql_attempt"]
        self.assertEqual(first["outcome"], "rejected by the SQL guard")
        self.assertIn("Available tables", first["error"])
        self.assertEqual(second["outcome"], "ok")
        self.assertEqual(self.saved["events"][1]["reasoning"], "count rentals")        # <think> tags captured
        self.assertEqual(a.sql, "SELECT COUNT(*) AS n FROM rental LIMIT 101")           # shows what actually ran
        self.assertEqual(a.text, "There are 16044 rentals.")

    def test_model_failure_is_logged_and_reraised(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("provider down")
        self.answer._client = lambda: client
        with self.assertRaises(RuntimeError):
            self.answer.ask("anything")
        self.assertEqual(self.saved["outcome"], "error")
        self.assertEqual(self.stages()[-1], "error")


if __name__ == "__main__":
    unittest.main()
