"""A strict guard for SQL written by an LLM: one read-only SELECT on the `sales` table, nothing else.

Why it exists
-------------
The analyst may use SQL when no fixed tool fits. The SQL text comes from an LLM that read a user's
question (which could be hostile), so it is treated as untrusted. This guard READS the text; it never
trusts it. If anything is unclear the query is REJECTED, with a message that tells the LLM what to fix.

What is allowed
---------------
  - exactly ONE statement that starts with SELECT (no WITH, no trailing second statement);
  - FROM the table `sales` only, or a derived table  FROM (SELECT ... FROM sales ...) AS t;
  - only functions from a fixed list (sum, count, round, date_trunc, strftime, ...);
  - normal clauses: WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, window functions (OVER), CASE, UNION.

What is rejected
----------------
  - comments, a second statement, unknown characters, unterminated quotes;
  - any word that changes things or reaches outside: INSERT UPDATE DELETE DROP ALTER CREATE ATTACH COPY
    PRAGMA SET INSTALL LOAD CALL EXPORT IMPORT ... (also INTO, RETURNING);
  - JOIN of any kind and comma-joins (the data is one table; this also stops "cross join" blow-ups);
  - subqueries inside expressions (WHERE x IN (SELECT ...), scalar subqueries), which can run once per row;
  - any function not on the list: this blocks file readers like read_csv(...) even when written as
    "read_csv"(...), and engine internals such as duckdb_settings();
  - text that looks like a file path or URL (a '/', a '\\', or a file extension inside a quoted string),
    which blocks FROM 'data.csv' style file reads;
  - any table other than `sales` (information_schema, other tables, schema-qualified names).

Rows are capped by wrapping the query:  SELECT * FROM (<query>) AS _q LIMIT <max_rows + 1>.

Honest limits
-------------
  - This is a text guard, not a full SQL parser. It is deliberately strict, so some valid queries are
    refused (the LLM is told why and may rewrite). It was tested on SQLite-style text and the cases
    in tests/test_analyst.py; the DuckDB-specific behaviour of an allowed query was not run by me.
  - There is no query timeout. Cost is limited by structure instead: one table, no joins, no
    subqueries inside expressions, a row cap on the output.
  - Even a read-only SELECT can reveal any row of the loaded file. That is what the analyst is for;
    it never touches anything outside that one file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_SQL_CHARS = 2000
TABLE_NAME = "sales"


class SqlGuardError(ValueError):
    """The SQL was rejected. The message says why and is safe to show."""


_TOKEN_RE = re.compile(r"""
    (?P<comment>--[^\n]*|/\*.*?\*/)
  | (?P<string>'(?:[^']|'')*')
  | (?P<qident>"(?:[^"]|"")*")
  | (?P<number>\d+(?:\.\d+)?)
  | (?P<word>[A-Za-z_][A-Za-z_0-9$]*)
  | (?P<punct>::|<>|<=|>=|!=|\|\||[(),;.*+\-/%<>=\[\]])
  | (?P<ws>\s+)
  | (?P<other>.)
""", re.X | re.S)

FORBIDDEN_WORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "attach", "detach", "copy", "export", "import",
    "install", "load", "pragma", "call", "set", "reset", "truncate", "vacuum", "checkpoint", "use", "begin",
    "commit", "rollback", "grant", "revoke", "merge", "replace_into", "into", "returning", "explain", "describe",
    "show", "summarize", "unpivot", "pivot", "recursive", "lateral", "macro", "sequence", "view", "table_function",
    "force", "analyze", "execute", "prepare", "deallocate", "comment", "transaction",
}

ALLOWED_FUNCTIONS = {
    # aggregates
    "sum", "count", "avg", "mean", "min", "max", "median", "mode", "stddev", "stddev_pop", "stddev_samp",
    "variance", "var_pop", "var_samp", "quantile", "quantile_cont", "quantile_disc", "approx_count_distinct",
    "any_value", "first", "last", "string_agg", "bool_and", "bool_or", "count_if",
    # maths
    "round", "abs", "ceil", "ceiling", "floor", "sqrt", "ln", "log", "log10", "exp", "power", "pow", "sign", "mod",
    "coalesce", "nullif", "ifnull", "greatest", "least",
    # casting and types
    "cast", "try_cast", "decimal", "numeric", "varchar", "char", "double", "float", "integer", "int", "bigint", "date",
    "timestamp",
    # dates
    "date_trunc", "date_part", "extract", "year", "month", "quarter", "day", "week", "dayofweek", "dayofmonth",
    "dayofyear", "monthname", "dayname", "strftime", "make_date", "date_diff", "datediff",
    # text
    "lower", "upper", "trim", "ltrim", "rtrim", "length", "len", "substr", "substring", "left", "right", "concat",
    "replace", "starts_with", "contains",
    # windows
    "row_number", "rank", "dense_rank", "ntile", "lag", "lead", "first_value", "last_value",
}

# words that may legitimately be followed by "(" without being a function call
SAFE_BEFORE_PAREN = {
    "in", "not", "and", "or", "exists", "any", "all", "some", "over", "filter", "from", "as", "on", "when", "then",
    "else", "case", "between", "like", "ilike", "select", "where", "having", "by", "limit", "offset", "union",
    "intersect", "except", "distinct", "values", "partition", "order", "group", "interval", "rows", "range",
    "using", "within", "is", "end", "asc", "desc", "nulls",
}

TERMINATORS = {"where", "group", "order", "having", "limit", "offset", "union", "intersect", "except",
               "window", "qualify"}
FROM_INSIDE_FUNCTIONS = {"extract", "substring", "trim", "position", "overlay"}   # EXTRACT(year FROM d)
FILE_LIKE = re.compile(r"[/\\]|://|\.(csv|tsv|parquet|json|jsonl|ndjson|db|duckdb|sqlite|txt|xlsx?|gz|zip)\b", re.I)


@dataclass
class _Tok:
    kind: str
    text: str


def _tokenize(sql: str) -> list[_Tok]:
    out = []
    for m in _TOKEN_RE.finditer(sql):
        kind = m.lastgroup
        if kind == "ws":
            continue
        out.append(_Tok(kind, m.group()))
    return out


def check_select(sql: str) -> list[str]:
    """Return the problems with this SQL (empty list = acceptable)."""
    problems: list[str] = []
    if not isinstance(sql, str) or not sql.strip():
        return ["The SQL is empty."]
    if len(sql) > MAX_SQL_CHARS:
        return [f"The SQL is too long ({len(sql)} characters; the limit is {MAX_SQL_CHARS})."]
    toks = _tokenize(sql)

    for t in toks:
        if t.kind == "comment":
            problems.append("Comments (-- or /* */) are not allowed.")
            break
    for t in toks:
        if t.kind == "other":
            problems.append(f"The character {t.text!r} is not allowed (or a quote is not closed).")
            break

    # exactly one statement: a single trailing ';' is tolerated, any other ';' is not
    body = list(toks)
    while body and body[-1].kind == "punct" and body[-1].text == ";":
        body.pop()
        break
    if any(t.kind == "punct" and t.text == ";" for t in body):
        problems.append("Only one statement is allowed (no ';' in the middle).")

    sig = [t for t in body if t.kind not in ("comment",)]
    if not sig or not (sig[0].kind == "word" and sig[0].text.lower() == "select"):
        problems.append("The query must start with SELECT (WITH and other statements are not allowed).")

    low = [(t.kind, t.text.lower() if t.kind == "word" else t.text) for t in sig]
    for kind, txt in low:
        if kind == "word" and txt in FORBIDDEN_WORDS:
            problems.append(f"The word {txt.upper()} is not allowed: only read-only SELECT queries are permitted.")
            break
        if kind == "word" and txt == "join":
            problems.append("JOIN is not allowed: the data is a single table called sales.")
            break

    for t in sig:
        if t.kind == "string":
            inner = t.text[1:-1].replace("''", "'")
            if FILE_LIKE.search(inner):
                problems.append("Text that looks like a file path or URL is not allowed inside quotes.")
                break

    # functions: every name directly followed by "(" must be on the list
    for i, (kind, txt) in enumerate(low[:-1]):
        nxt_kind, nxt_txt = low[i + 1]
        if nxt_kind == "punct" and nxt_txt == "(":
            if kind == "word":
                name = txt
            elif kind == "qident":
                name = txt[1:-1].replace('""', '"').lower()
            else:
                continue
            if name in SAFE_BEFORE_PAREN and kind == "word":
                continue
            if name not in ALLOWED_FUNCTIONS:
                problems.append(f"The function {name}(...) is not allowed. Allowed functions include "
                                f"sum, count, avg, min, max, round, coalesce, date_trunc, strftime, lower, upper.")
                break

    # subqueries only as derived tables: FROM ( SELECT ... )
    for i, (kind, txt) in enumerate(low[:-1]):
        if kind == "punct" and txt == "(" and low[i + 1] == ("word", "select"):
            prev = low[i - 1] if i > 0 else None
            if prev != ("word", "from"):
                problems.append("A subquery is only allowed directly after FROM, as in FROM (SELECT ...) AS t.")
                break

    # FROM rules (table name, no comma-joins), tracking parentheses
    stack: list[str] = []                       # "fn" when the paren belongs to extract/substring/trim...
    in_from: dict[int, bool] = {}
    for i, (kind, txt) in enumerate(low):
        depth = len(stack)
        if kind == "punct" and txt == "(":
            prev = low[i - 1] if i > 0 else (None, None)
            stack.append("fn" if prev[0] == "word" and prev[1] in FROM_INSIDE_FUNCTIONS else "x")
            continue
        if kind == "punct" and txt == ")":
            in_from.pop(depth, None)
            if stack:
                stack.pop()
            continue
        if kind == "word" and txt == "from":
            if stack and stack[-1] == "fn":
                continue
            nxt = low[i + 1] if i + 1 < len(low) else (None, None)
            ok = (nxt == ("word", TABLE_NAME) or
                  (nxt[0] == "qident" and nxt[1][1:-1].lower() == TABLE_NAME) or
                  nxt == ("punct", "("))
            if not ok:
                shown = nxt[1] if nxt[1] is not None else "nothing"
                problems.append(f"Only the table '{TABLE_NAME}' can be queried (found FROM {shown}).")
                break
            if nxt[0] != "punct":
                after = low[i + 2] if i + 2 < len(low) else (None, None)
                if after == ("punct", "."):
                    problems.append(f"Use the plain table name '{TABLE_NAME}' without a schema prefix.")
                    break
            in_from[depth] = True
            continue
        if in_from.get(depth):
            if kind == "word" and txt in TERMINATORS:
                in_from[depth] = False
            elif kind == "punct" and txt == ",":
                problems.append("Comma-joins are not allowed: the data is a single table.")
                break

    return list(dict.fromkeys(problems))


def clean_select(sql: str, max_rows: int = 200) -> tuple[str, str]:
    """Validate and return (the query without a trailing ';', the row-capped query to run).
    Raises SqlGuardError with ALL the problems if it is not acceptable."""
    problems = check_select(sql)
    if problems:
        raise SqlGuardError(" ".join(problems))
    core = sql.strip()
    while core.endswith(";"):
        core = core[:-1].rstrip()
    return core, f"SELECT * FROM ({core}) AS _q LIMIT {int(max_rows) + 1}"
