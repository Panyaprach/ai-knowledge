import unittest
from kb.quality import looks_degenerate

LOOP = "```mermaid\nerDiagram\n    actor |--| film_actor\n" + "".join(
    f"    film |--| {t}\n" for _ in range(12) for t in ("film_actor", "film_text", "film_category"))
GOOD = "## Tables / entities\n" + "\n".join(f"- table_{i}" for i in range(23)) + "\n## Relationships\n" + "\n".join(
    f"table_{i} --fk_{i}--> table_{i + 1}" for i in range(20))


class Quality(unittest.TestCase):
    def test_repeated_lines_are_degenerate(self):
        self.assertTrue(looks_degenerate(LOOP))

    def test_repeated_sequence_on_one_line_is_degenerate(self):
        self.assertTrue(looks_degenerate("film |--| film_actor film |--| film_text " * 40))

    def test_real_description_is_fine(self):
        self.assertFalse(looks_degenerate(GOOD))
        self.assertFalse(looks_degenerate("short answer"))


if __name__ == "__main__":
    unittest.main()
