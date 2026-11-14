"""Guard for LLM-written SQL, built on a real parser (sqlglot) instead of keyword matching.

validate() parses the query with the MySQL dialect and then checks the syntax tree:
  * exactly one statement, and it is a SELECT (or UNION/INTERSECT/EXCEPT of SELECTs, CTEs allowed)
  * no write/DDL/admin nodes anywhere in the tree (INSERT, UPDATE, DELETE, DROP, ..., SELECT ... INTO, FOR UPDATE)
  * no dangerous functions (SLEEP, BENCHMARK, LOAD_FILE, GET_LOCK, ...) and no @user / @@system variables
  * every table is in the allow-list (the tables of the Sakila schema) and in the right database,
    so information_schema / mysql.* are unreachable even if the DB user could read them
  * a LIMIT is always present and capped
The query that is returned is regenerated from the tree with comments removed, so nothing hidden in a comment
(e.g. MySQL's executable /*! ... */ comments) can reach the server. The DB session is also forced READ ONLY.
"""
import re


class UnsafeSQL(ValueError):
    pass


# Node classes that must never appear. Looked up by name so different sqlglot versions are tolerated.
_FORBIDDEN_NODE_NAMES = [
    "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "AlterTable", "TruncateTable",
    "Command", "Set", "Use", "Pragma", "Transaction", "Commit", "Rollback", "LoadData", "Into", "Lock",
    "Parameter", "SessionParameter", "Grant", "Kill",
]
BLOCKED_FUNCTIONS = {
    "SLEEP", "BENCHMARK", "LOAD_FILE", "GET_LOCK", "RELEASE_LOCK", "RELEASE_ALL_LOCKS", "IS_FREE_LOCK",
    "IS_USED_LOCK", "MASTER_POS_WAIT", "SYS_EXEC", "SYS_EVAL", "UPDATEXML", "EXTRACTVALUE",
}


def validate(sql: str, allowed_tables: set[str] | None = None, max_rows: int | None = None,
             database: str | None = None, dialect: str = "mysql") -> str:
    """Return a safe, comment-free, LIMIT-capped version of `sql`, or raise UnsafeSQL."""
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError

    text = (sql or "").strip()
    if not text:
        raise UnsafeSQL("empty query")
    if "/*" in text or "--" in text:
        raise UnsafeSQL("comments are not allowed")

    try:
        statements = [s for s in sqlglot.parse(text, read=dialect) if s is not None]
    except SqlglotError as e:
        raise UnsafeSQL(f"cannot parse as {dialect} SQL: {str(e).splitlines()[0][:200]}") from None
    if len(statements) != 1:
        raise UnsafeSQL("exactly one statement is allowed")
    tree = statements[0]

    set_op = getattr(exp, "SetOperation", exp.Union)
    if not isinstance(tree, (exp.Select, set_op)):
        raise UnsafeSQL("only SELECT queries are allowed")

    forbidden = tuple(c for c in (getattr(exp, n, None) for n in _FORBIDDEN_NODE_NAMES) if isinstance(c, type))
    for node in tree.find_all(exp.Expression):
        if isinstance(node, forbidden):
            raise UnsafeSQL(f"{type(node).__name__.upper()} is not allowed in a read-only query")
        if isinstance(node, exp.Func):
            name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            if str(name).upper() in BLOCKED_FUNCTIONS:
                raise UnsafeSQL(f"function {str(name).upper()} is not allowed")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables} if allowed_tables is not None else None
    for table in tree.find_all(exp.Table):
        name = table.name
        if not name:
            raise UnsafeSQL("table-valued expressions are not allowed")
        if table.db and database and table.db.lower() != database.lower():
            raise UnsafeSQL(f"only database '{database}' may be queried (found '{table.db}')")
        if table.db and not database:
            raise UnsafeSQL("schema-qualified tables are not allowed")
        if allowed is not None and not table.db and name.lower() not in cte_names and name.lower() not in allowed:
            known = ", ".join(sorted(allowed)[:40])
            raise UnsafeSQL(f"unknown table '{name}'. Available tables: {known}")
        if allowed is not None and table.db and name.lower() not in allowed:
            known = ", ".join(sorted(allowed)[:40])
            raise UnsafeSQL(f"unknown table '{name}'. Available tables: {known}")

    if max_rows:
        limit = tree.args.get("limit")
        try:
            current = int(limit.expression.name) if limit is not None else None
        except (ValueError, AttributeError):
            current = None
        if current is None or current > max_rows:
            tree = tree.limit(max_rows)

    return tree.sql(dialect=dialect, comments=False)


def extract_sql(text: str) -> str | None:
    m = re.search(r"```(?:sql|mysql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    return text.strip() if re.match(r"(?is)^\s*(select|with)\b", text) else None
