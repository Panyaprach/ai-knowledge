"""MySQL access: schema introspection -> 'schema cards', and a guarded read-only query runner."""
import datetime, decimal, time
from functools import lru_cache
import pymysql
from .config import (MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MAX_ROWS)
from .sqlguard import validate


def mysql_conn(timeout: int = 15):
    return pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD,
                           database=MYSQL_DATABASE, charset="utf8mb4",
                           connect_timeout=timeout, read_timeout=30)


def wait_for_mysql(timeout: int = 120) -> None:
    end, err = time.time() + timeout, None
    while time.time() < end:
        try:
            mysql_conn(3).close()
            return
        except Exception as e:
            err = e
            time.sleep(3)
    raise RuntimeError(f"MySQL not reachable at {MYSQL_HOST}:{MYSQL_PORT}: {err}")


def _q(cur, sql, args=()):
    cur.execute(sql, args)
    return cur.fetchall()


def build_cards() -> list[dict]:
    """One text card per table/view: columns, types (incl. enum values), keys, FK in/out, approx rows."""
    db = MYSQL_DATABASE
    conn = mysql_conn()
    try:
        with conn.cursor() as cur:
            tables = _q(cur, "SELECT TABLE_NAME t, TABLE_TYPE ty, TABLE_ROWS r FROM information_schema.TABLES "
                             "WHERE TABLE_SCHEMA=%s ORDER BY TABLE_NAME", (db,))
            cols = _q(cur, "SELECT TABLE_NAME t, COLUMN_NAME c, COLUMN_TYPE ty, IS_NULLABLE n, COLUMN_KEY k, "
                           "EXTRA e FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s "
                           "ORDER BY TABLE_NAME, ORDINAL_POSITION", (db,))
            fks = _q(cur, "SELECT TABLE_NAME t, COLUMN_NAME c, REFERENCED_TABLE_NAME rt, REFERENCED_COLUMN_NAME rc "
                          "FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA=%s "
                          "AND REFERENCED_TABLE_NAME IS NOT NULL ORDER BY TABLE_NAME, ORDINAL_POSITION", (db,))
    finally:
        conn.close()

    out_fk, in_fk = {}, {}
    for t, c, rt, rc in fks:
        out_fk.setdefault(t, []).append(f"{c} -> {rt}.{rc}")
        in_fk.setdefault(rt, []).append(f"{t}.{c}")
    by_table = {}
    for t, c, ty, n, k, e in cols:
        flags = [f for f in ("NOT NULL" if n == "NO" else "NULL",
                             {"PRI": "PK", "UNI": "UNIQUE", "MUL": "indexed"}.get(k, ""), e or "") if f]
        by_table.setdefault(t, []).append(f"- {c} {ty} {' '.join(flags)}")

    cards = []
    for t, ty, rows in tables:
        kind = "VIEW" if ty == "VIEW" else "TABLE"
        neighbors = sorted({x.split(" -> ")[1].split(".")[0] for x in out_fk.get(t, [])} |
                           {x.split(".")[0] for x in in_fk.get(t, [])})
        text = [f"[MySQL {kind.lower()} {db}.{t}]" + (f" ~{rows} rows" if kind == "TABLE" and rows else ""),
                "Columns:", *by_table.get(t, [])]
        if out_fk.get(t):
            text.append("Foreign keys: " + "; ".join(out_fk[t]))
        if in_fk.get(t):
            text.append("Referenced by: " + ", ".join(in_fk[t]))
        cards.append({"table": t, "kind": kind, "text": "\n".join(text), "neighbors": neighbors})
    return cards


def _jsonable(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        return "<binary>"
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    return v


@lru_cache(maxsize=1)
def table_names() -> frozenset:
    """Tables and views of the configured database: the only names a generated query may reference."""
    conn = mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s", (MYSQL_DATABASE,))
            return frozenset(r[0].lower() for r in cur.fetchall())
    finally:
        conn.close()


def run_select(sql: str, max_rows: int = MAX_ROWS):
    """Validate, then run in a READ ONLY session. Returns (columns, rows, truncated, executed_sql)."""
    sql = validate(sql, allowed_tables=table_names(), max_rows=max_rows + 1, database=MYSQL_DATABASE)
    conn = mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            cur.execute("SET SESSION MAX_EXECUTION_TIME=10000")
            cur.execute(sql)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(max_rows + 1)
    finally:
        conn.close()
    return columns, [[_jsonable(v) for v in r] for r in rows[:max_rows]], len(rows) > max_rows, sql
