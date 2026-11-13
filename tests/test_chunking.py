import unittest
from kb.chunking import clean_pages, section_chunks
from kb.sqlguard import extract_sql


def page(n, *body):
    return (n, ["Sakila Sample Database", *body, f"{n}Sakila Sample Database"])


class Chunking(unittest.TestCase):
    pages = [
        (1, ["Table of Contents", "5.1.6 The customer Table ........ 7", "Sakila Sample Database"]),
        page(2, "5.1.6 The customer Table", "The customer table contains a list of all customers.", "• active: whether the customer is active."),
        page(3, "5.1.7 The film Table", "The film table lists all films in stock at the stores.", "1 row in set (0.00 sec)"),
        page(4, "5.1.8 The film_actor Table", "The film_actor table links films and actors (many-to-many)."),
    ]

    def test_headers_footers_and_toc_removed(self):
        cleaned = clean_pages(self.pages)
        text = "\n".join(s for _, lines in cleaned for _, s in lines)
        self.assertNotIn("Sakila Sample Database", text)
        self.assertNotIn("......", text)

    def test_one_chunk_per_section_with_label(self):
        ch = section_chunks(self.pages, "Sakila Sample Database")
        labels = [c["section"] for c in ch]
        self.assertIn("5.1.6 The customer Table", labels)
        self.assertIn("5.1.8 The film_actor Table", labels)
        film = next(c for c in ch if "film Table" in c["section"])
        self.assertIn("1 row in set", film["text"])        # not mistaken for a heading
        self.assertEqual(film["page"], 3)

    def test_extract_sql(self):
        self.assertEqual(extract_sql("```sql\nSELECT 1\n```"), "SELECT 1")
        self.assertEqual(extract_sql("SELECT 2"), "SELECT 2")
        self.assertIsNone(extract_sql("The film_actor table links them."))


if __name__ == "__main__":
    unittest.main()
