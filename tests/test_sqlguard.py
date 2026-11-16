import importlib.util
import unittest

from kb.sqlguard import validate, UnsafeSQL

HAS_SQLGLOT = importlib.util.find_spec("sqlglot") is not None
TABLES = {"film", "customer", "category", "film_category", "payment", "rental", "inventory", "actor"}


@unittest.skipUnless(HAS_SQLGLOT, "pip install sqlglot to run these tests")
class SqlGuard(unittest.TestCase):
    def ok(self, sql, **kw):
        kw.setdefault("allowed_tables", TABLES)
        kw.setdefault("database", "sakila")
        return validate(sql, **kw)

    def bad(self, sql, **kw):
        kw.setdefault("allowed_tables", TABLES)
        kw.setdefault("database", "sakila")
        with self.assertRaises(UnsafeSQL, msg=sql):
            validate(sql, **kw)

    def test_allows_reads(self):
        self.assertTrue(validate("SELECT 1;").upper().startswith("SELECT 1"))
        self.ok("select c.last_update from customer c where c.first_name = 'DELETE'")
        self.ok("WITH t AS (SELECT film_id FROM film) SELECT * FROM t")
        self.ok("SELECT title FROM film UNION SELECT name FROM category")
        self.ok("SELECT c.name, COUNT(*) FROM film_category fc JOIN category c ON c.category_id = fc.category_id GROUP BY c.name")
        self.ok("SELECT * FROM sakila.film")

    def test_limit_is_added_and_capped(self):
        self.assertIn("LIMIT 100", self.ok("SELECT * FROM film", max_rows=100).upper())
        out = self.ok("SELECT * FROM film LIMIT 5", max_rows=100).upper()
        self.assertIn("LIMIT 5", out)
        self.assertNotIn("LIMIT 100", out)
        self.assertIn("LIMIT 100", self.ok("SELECT * FROM film LIMIT 100000", max_rows=100).upper())

    def test_blocks_writes_and_admin(self):
        for q in ["DROP TABLE film", "DELETE FROM film", "UPDATE film SET title='x'",
                  "INSERT INTO film(title) VALUES ('x')", "SELECT 1; DROP TABLE film", "",
                  "CALL film_in_stock(1,1,@c)", "show tables", "SELECT * FROM film FOR UPDATE",
                  "SELECT * FROM film INTO OUTFILE '/tmp/x'"]:
            self.bad(q)

    def test_blocks_dangerous_functions_and_variables(self):
        for q in ["SELECT SLEEP(10)", "SELECT BENCHMARK(1000000, MD5('a'))", "SELECT LOAD_FILE('/etc/passwd')",
                  "SELECT @@version", "SELECT @a := 1"]:
            self.bad(q)

    def test_blocks_comments(self):
        self.bad("SELECT * FROM film -- hi")
        self.bad("SELECT /*! 50000 1 */ 1")

    def test_blocks_other_databases_and_unknown_tables(self):
        self.bad("SELECT * FROM information_schema.tables")
        self.bad("SELECT * FROM mysql.user")
        self.bad("SELECT * FROM other.film")
        with self.assertRaises(UnsafeSQL) as cm:
            validate("SELECT * FROM rentals", allowed_tables=TABLES, database="sakila")
        self.assertIn("Available tables", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
