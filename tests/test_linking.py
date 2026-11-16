import unittest
from kb.linking import find_named_tables

TABLES = ["actor", "film", "film_actor", "film_category", "category", "rental", "inventory", "customer", "store"]


class Linking(unittest.TestCase):
    def test_finds_named_tables_in_order(self):
        self.assertEqual(find_named_tables("What are the foreign keys of the rental table?", TABLES), ["rental"])
        self.assertEqual(find_named_tables("How many films are in each category?", TABLES), ["film", "category"])
        self.assertEqual(find_named_tables("top Customers by rentals", TABLES), ["customer", "rental"])

    def test_underscore_names_do_not_match_their_prefix(self):
        self.assertEqual(find_named_tables("explain film_actor", TABLES), ["film_actor"])
        self.assertEqual(find_named_tables("explain film_actors please", TABLES), ["film_actor"])

    def test_no_false_hits_inside_other_words(self):
        self.assertEqual(find_named_tables("restore the inventories", TABLES), [])
        self.assertEqual(find_named_tables("filmography and storefront", TABLES), [])


if __name__ == "__main__":
    unittest.main()
