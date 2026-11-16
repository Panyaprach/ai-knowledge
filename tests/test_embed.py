import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

import kb.embed as E


class RateLimitError(Exception):
    pass


class Fake:
    def __init__(self, dim=4, fail_times=0, error=None):
        self.dim, self.fail_times, self.error, self.calls = dim, fail_times, error, []
        self.embeddings = NS(create=self.create)

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        if self.fail_times:
            self.fail_times -= 1
            raise RateLimitError("slow down")
        data = [NS(index=i, embedding=[float(i + 1)] * self.dim) for i, _ in enumerate(kw["input"])]
        return NS(data=list(reversed(data)), usage=NS(prompt_tokens=len(kw["input"]) * 3, total_tokens=len(kw["input"]) * 3))


class Remote(unittest.TestCase):
    def run_embed(self, fake, texts, kind="doc", **over):
        cfg = dict(EMBED_BASE_URL="http://x/v1", EMBED_DIM=4, EMBED_QUERY_PREFIX="", EMBED_DOC_PREFIX="",
                   EMBED_SEND_DIMENSIONS=False)
        cfg.update(over)
        with patch.multiple(E, _client=lambda: fake, record=lambda *a, **k: None, **cfg), patch.object(E.time, "sleep"):
            return E.embed(texts, kind)

    def test_batches_orders_and_normalises(self):
        fake = Fake()
        out = self.run_embed(fake, [f"t{i}" for i in range(70)])
        self.assertEqual([len(c["input"]) for c in fake.calls], [32, 32, 6])
        self.assertEqual(out.shape, (70, 4))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)

    def test_prefixes_by_kind(self):
        fake = Fake()
        self.run_embed(fake, ["a"], "query", EMBED_QUERY_PREFIX="q: ", EMBED_DOC_PREFIX="d: ")
        self.run_embed(fake, ["a"], "doc", EMBED_QUERY_PREFIX="q: ", EMBED_DOC_PREFIX="d: ")
        self.assertEqual([c["input"] for c in fake.calls], [["q: a"], ["d: a"]])

    def test_dimensions_parameter_only_when_enabled(self):
        fake = Fake()
        self.run_embed(fake, ["a"])
        self.run_embed(fake, ["a"], EMBED_SEND_DIMENSIONS=True)
        self.assertNotIn("dimensions", fake.calls[0])
        self.assertEqual(fake.calls[1]["dimensions"], 4)

    def test_wrong_dimension_is_explained(self):
        with self.assertRaises(E.EmbeddingError) as cm:
            self.run_embed(Fake(dim=8), ["a"])
        self.assertIn("EMBED_DIM=4", str(cm.exception))

    def test_retries_rate_limits_but_not_other_errors(self):
        fake = Fake(fail_times=2)
        self.assertEqual(self.run_embed(fake, ["a"]).shape, (1, 4))
        self.assertEqual(len(fake.calls), 3)
        bad = Fake(error=ValueError("bad key"))
        with self.assertRaises(E.EmbeddingError):
            self.run_embed(bad, ["a"])
        self.assertEqual(len(bad.calls), 1)

    def test_backend_description(self):
        with patch.multiple(E, EMBED_BASE_URL="http://localhost:11434/v1", EMBED_MODEL="nomic-embed-text"):
            self.assertEqual(E.describe_backend(), "nomic-embed-text via localhost:11434")
        with patch.multiple(E, EMBED_BASE_URL="", EMBED_MODEL="bge"):
            self.assertEqual(E.describe_backend(), "bge, local")


if __name__ == "__main__":
    unittest.main()
