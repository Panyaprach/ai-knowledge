import unittest
from kb.sqlguard import validate, UnsafeSQL


class SqlGuard(unittest.TestCase):
    def test_allows_select_and_with(self):
        self.assertEqual(validate("SELECT 1;"), "SELECT 1")
        validate("WITH t AS (SELECT 1 a) SELECT * FROM t")
        validate("select c.last_update from customer c where c.first_name = 'DELETE'")  # keyword only inside a string / identifier

    def test_blocks_writes_and_tricks(self):
        for bad in ["DROP TABLE film", "DELETE FROM film", "UPDATE film SET title='x'",
                    "SELECT 1; DROP TABLE film", "SELECT * FROM film INTO OUTFILE '/tmp/x'",
                    "SELECT SLEEP(10)", "SELECT * FROM film -- hi", "SELECT /* x */ 1",
                    "CALL film_in_stock(1,1,@c)", "SELECT * FROM film FOR UPDATE", "", "show tables"]:
            with self.assertRaises(UnsafeSQL, msg=bad):
                validate(bad)


if __name__ == "__main__":
    unittest.main()
