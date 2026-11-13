"""Defence-in-depth for LLM-written SQL. (The DB session is also forced READ ONLY.)"""
import re


class UnsafeSQL(ValueError):
    pass


FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|call|lock|load|outfile|dumpfile|"
    r"into|sleep|benchmark|load_file|get_lock|kill|shutdown|rename|handler|prepare|execute)\b", re.I)
STRING = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.|\"\")*\"")


def validate(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise UnsafeSQL("empty query")
    if ";" in STRING.sub("''", s):
        raise UnsafeSQL("multiple statements are not allowed")
    bare = STRING.sub("''", s)
    if "--" in bare or "/*" in bare or "#" in bare:
        raise UnsafeSQL("comments are not allowed")
    if not re.match(r"(?is)^(select|with)\b", bare):
        raise UnsafeSQL("only SELECT queries are allowed")
    m = FORBIDDEN.search(bare)
    if m:
        raise UnsafeSQL(f"forbidden keyword: {m.group(1).upper()}")
    return s


def extract_sql(text: str) -> str | None:
    m = re.search(r"```(?:sql|mysql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    return text.strip() if re.match(r"(?is)^\s*(select|with)\b", text) else None
