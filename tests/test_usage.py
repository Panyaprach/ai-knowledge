import contextlib
import io
import unittest
from types import SimpleNamespace as NS

import logging

from kb.config import _Compact, setup_logging
from kb.usage import est_tokens, extract_usage, flags, format_summary, normalize, sum_events


class Usage(unittest.TestCase):
    def test_extract_and_normalize(self):
        resp = NS(usage=NS(prompt_tokens=120, completion_tokens=30, completion_tokens_details=NS(reasoning_tokens=12)))
        u = extract_usage(resp)
        self.assertEqual(u, {"prompt": 120, "completion": 30, "reasoning": 12})
        self.assertEqual(normalize(u, 99, 99), {"prompt": 120, "completion": 30, "reasoning": 12, "estimated": False})
        self.assertIsNone(extract_usage(NS(usage=None)))

    def test_estimate_when_provider_reports_nothing(self):
        n = normalize(None, est_prompt=500, est_completion=40)
        self.assertEqual((n["prompt"], n["completion"], n["estimated"]), (500, 40, True))
        self.assertEqual(est_tokens("x" * 400), 100)

    def test_sum_events_ignores_other_stages(self):
        events = [{"stage": "retrieve"},
                  {"stage": "llm_call", "usage": {"prompt": 100, "completion": 10, "reasoning": None}},
                  {"stage": "llm_call", "usage": {"prompt": 50, "completion": 5, "reasoning": 3}}]
        self.assertEqual(sum_events(events), {"calls": 2, "prompt": 150, "completion": 15, "reasoning": 3})

    def test_flags(self):
        self.assertTrue(any("cut off" in f for f in flags(500, 500, "length", "llm")))
        self.assertTrue(any("4096" in f for f in flags(4090, 6000, "stop", "llm")))       # sits right at a context cap
        self.assertTrue(any("fewer prompt tokens" in f for f in flags(3000, 6000, "stop", "llm")))
        self.assertEqual(flags(3000, 3100, "stop", "llm"), [])
        self.assertEqual(flags(3000, 6000, "stop", "vision"), [])                         # image tokens: no text-only comparison

    def test_format_summary(self):
        rows = [{"day": "2026-11-14", "kind": "llm", "purpose": "route", "model": "qwen2.5-coder:7b", "calls": 3,
                 "prompt": 9000, "completion": 300, "reasoning": 0, "avg_ms": 8100, "n_estimated": 0}]
        out = format_summary(rows, [{"ts": "2026-11-14 10:00:00", "kind": "llm", "purpose": "route", "prompt_tokens": 4096,
                                     "flags": ["prompt tokens are right at 4096"]}], 7)
        for needle in ["route", "9,000", "Total: 3 calls", "Embeddings: local", "Warnings"]:
            self.assertIn(needle, out)
        self.assertIn("No model calls", format_summary([], [], 7))


class Logging(unittest.TestCase):
    def tearDown(self):
        lg = logging.getLogger("kb")
        lg.handlers.clear()
        lg.setLevel(logging.NOTSET)

    def test_setup_is_idempotent_and_maps_levels(self):
        setup_logging("debug")
        setup_logging("debug")
        lg = logging.getLogger("kb")
        self.assertEqual(lg.level, logging.DEBUG)
        self.assertEqual(sum(getattr(h, "_kb", False) for h in lg.handlers), 1)   # no duplicate handlers
        setup_logging("quiet")
        self.assertEqual(lg.level, logging.WARNING)
        setup_logging("nonsense")
        self.assertEqual(lg.level, logging.INFO)

    def test_compact_format(self):
        info = _Compact().format(logging.LogRecord("kb.vision", logging.INFO, "f", 1, "hello %s", ("x",), None))
        self.assertRegex(info, r"^\[\d\d:\d\d:\d\d\] vision   hello x$")
        warn = _Compact().format(logging.LogRecord("kb.vision", logging.WARNING, "f", 1, "careful", (), None))
        self.assertIn("WARNING vision", warn)

    def test_library_modules_log_to_named_loggers(self):
        with self.assertLogs("kb.usage", level="WARNING") as cm:
            logging.getLogger("kb.usage").warning("could not save token usage: %s", "boom")
        self.assertIn("boom", cm.output[0])


if __name__ == "__main__":
    unittest.main()
