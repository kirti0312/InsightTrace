"""Load a CSV/Excel sales file into DuckDB and work out what each column means.

What this file does, in plain words:
  1. Reads the file (handles odd encodings, ; or tab separators, Excel sheets).
  2. Guesses which column is the date, revenue, profit, quantity, discount,
     order id, and which columns are "dimensions" (Region, Product, Customer...).
  3. Cleans the values (dates, "$1,200.50" style numbers).
  4. Stores a tidy table called `sales` in an in-memory DuckDB database.

Everything the rest of the engine does starts from the `Dataset` object that
`load_dataset()` returns.

Try it from the project folder:
    python -m engine.ingest data/Sample_-_Superstore.csv
"""
from __future__ import annotations

import csv
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

TABLE = "sales"
ROLE_COLUMNS = ("date", "revenue", "profit", "quantity", "discount", "order_id")


class IngestError(Exception):
    """Raised when a file cannot be used. The message is meant for the user."""


def q(name: str) -> str:
    """Quote a column name for SQL. Column names come from uploaded files, so
    we ALWAYS quote them (this also blocks SQL injection through headers)."""
    return '"' + str(name).replace('"', '""') + '"'


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


# ----------------------------------------------------------------------------
# 1. Reading files
# ----------------------------------------------------------------------------
def _get_bytes(source, filename: str | None):
    """Return (raw_bytes, name) for a path, bytes, or an uploaded-file object."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise IngestError(f"File not found: {path}")
        return path.read_bytes(), filename or path.name
    name = filename or getattr(source, "name", None) or "uploaded_file"
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), name
    if hasattr(source, "getvalue"):
        return source.getvalue(), name
    if hasattr(source, "read"):
        return source.read(), name
    raise IngestError("Unsupported file input.")


def _decode(raw: bytes):
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise IngestError("Could not read the text encoding of this file.")


def _sniff_delimiter(text: str) -> str:
    sample = text[:20000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_table(source, filename: str | None = None, sheet=None):
    """Read CSV/TSV/Excel into a DataFrame. Returns (df, info_notes)."""
    raw, name = _get_bytes(source, filename)
    if not raw:
        raise IngestError("The file is empty.")
    ext = Path(name).suffix.lower()
    notes: list[str] = []

    if ext == ".xls":
        raise IngestError("Old .xls files are not supported. In Excel choose "
                          "File > Save As > .xlsx (or CSV) and upload that.")
    is_excel = ext in (".xlsx", ".xlsm") or (ext not in (".csv", ".tsv", ".txt")
                                             and raw[:2] == b"PK")
    if is_excel:
        try:
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=sheet if sheet is not None else None)
        except Exception as exc:  # noqa: BLE001
            raise IngestError(f"Could not read the Excel file: {exc}") from exc
        if isinstance(sheets, pd.DataFrame):
            df = sheets
        else:
            # several sheets: use the biggest one
            best = max(sheets, key=lambda k: len(sheets[k]))
            df = sheets[best]
            if len(sheets) > 1:
                notes.append(f"Excel file has {len(sheets)} sheets; used '{best}' "
                             f"(the biggest).")
    else:
        text, enc = _decode(raw)
        delim = _sniff_delimiter(text)
        try:
            df = pd.read_csv(io.StringIO(text), sep=delim)
        except Exception as exc:  # noqa: BLE001
            raise IngestError(f"Could not read the file as CSV: {exc}") from exc
        if enc != "utf-8-sig":
            notes.append(f"File is not UTF-8; read it as {enc}.")
        if delim != ",":
            notes.append(f"Separator detected: {delim!r}.")

    # tidy: drop empty columns/rows and blank header names
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [not c.lower().startswith("unnamed:") or df[c].notna().any()
                    for c in df.columns]]
    df = df.dropna(how="all")
    if df.empty or len(df.columns) < 2:
        raise IngestError("The file has no usable rows or columns.")
    return df.reset_index(drop=True), notes


# ----------------------------------------------------------------------------
# 2. Cleaning values
# ----------------------------------------------------------------------------
_DATE_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d",
    "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%B %d, %Y", "%d %B %Y",
    "%m/%d/%y", "%d/%m/%y",
]
_DATE_REGEX = re.compile(
    r"^\s*(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}[ -][A-Za-z]{3,9}[ -]\d{2,4}|"
    r"[A-Za-z]{3,9} \d{1,2}, \d{4})")
_DATE_NAME_TOKENS = {"date", "time", "day", "month", "period", "timestamp",
                     "created", "placed", "ordered"}


def _sample_strings(series: pd.Series, n: int = 2000) -> pd.Series:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return s.head(n)


def _looks_like_date(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    s = _sample_strings(series, 200)
    if s.empty:
        return False
    hit = s.str.match(_DATE_REGEX).mean()
    name_hint = bool(set(_norm(name).split()) & _DATE_NAME_TOKENS)
    return hit >= 0.9 or (name_hint and hit >= 0.5)


def parse_dates(series: pd.Series):
    """Turn a column into real dates. Returns (dates, description, warnings)."""
    warnings: list[str] = []
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.normalize(), "already dates", warnings

    if pd.api.types.is_numeric_dtype(series):  # Excel serial numbers
        nonnull = series.dropna()
        if len(nonnull) and nonnull.between(20000, 60000).all():
            out = pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")
            return out.dt.normalize(), "Excel serial numbers", warnings
        raise IngestError("Date column holds numbers that are not dates.")

    sample = _sample_strings(series)
    if sample.empty:
        raise IngestError("The date column is empty.")

    scores = []
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        scores.append((parsed.notna().mean(), fmt))
    best_ratio = max(s for s, _ in scores)
    if best_ratio >= 0.9:
        best_fmts = [f for s, f in scores if s == best_ratio]
        fmt = best_fmts[0]  # list order prefers month/day (US) over day/month
        if "%m/%d/%Y" in best_fmts and "%d/%m/%Y" in best_fmts:
            warnings.append("Dates like 03/04/2016 are ambiguous (March 4 or "
                            "April 3?). Assumed month/day. If that is wrong, "
                            "tell me and we can switch it.")
        full = pd.to_datetime(series.astype(str).str.strip(), format=fmt, errors="coerce")
        return full.dt.normalize(), f"format {fmt}", warnings

    mixed = pd.to_datetime(sample, format="mixed", errors="coerce")
    if mixed.notna().mean() >= 0.9:
        full = pd.to_datetime(series.astype(str).str.strip(), format="mixed", errors="coerce")
        warnings.append("Date formats are mixed in this column; parsed them one by one.")
        return full.dt.normalize(), "mixed formats", warnings

    raise IngestError("Could not understand the dates in the date column.")


def to_numeric(series: pd.Series):
    """Numbers, even when stored as text like '$1,200.50' or '(45.00)'.
    Returns (numbers, share_of_values_that_worked)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce"), 1.0
    s = series.astype(str).str.strip()
    s = s.where(series.notna())
    neg = s.str.match(r"^\(.*\)$", na=False)
    s = s.str.replace(r"[()$€£₹¥%\s]", "", regex=True).str.replace(",", "", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    out = out.where(~neg, -out)
    nonnull = series.notna().sum()
    ratio = out.notna().sum() / nonnull if nonnull else 0.0
    return out, float(ratio)


# ----------------------------------------------------------------------------
# 3. Working out what each column means
# ----------------------------------------------------------------------------
_SYNONYMS = {
    "date": ["order date", "invoice date", "transaction date", "sale date",
             "sales date", "purchase date", "date", "created at", "timestamp",
             "order placed", "ordered at"],
    "revenue": ["sales", "revenue", "sales amount", "net sales", "total sales",
                "amount", "turnover", "gmv", "order total", "line total", "total"],
    "profit": ["profit", "net profit", "gross profit", "profit amount"],
    "quantity": ["quantity", "qty", "units", "units sold", "order quantity"],
    "discount": ["discount", "discount rate", "discount pct", "discount percent", "disc"],
    "order_id": ["order id", "order number", "order no", "invoice number",
                 "invoice no", "invoice", "transaction id", "bill no", "receipt", "order"],
}
_EXCLUDE = {
    "profit": {"margin", "ratio", "pct", "percent", "rate"},
    "revenue": {"id", "code", "date", "profit", "margin", "tax", "discount", "cost", "ship", "shipping"},
    "quantity": {"id", "code", "date"},
    "discount": {"id", "code", "date"},
    "date": {"ship", "shipping", "delivery", "due", "birth", "expiry", "expire"},
    "order_id": {"date", "priority"},
}


def _pick(role: str, candidates: list[str]) -> str | None:
    """Choose the best column for a role by name. Exact name matches first,
    then 'contains' matches (fewest extra words wins)."""
    banned = _EXCLUDE.get(role, set())
    pool = [(c, _norm(c)) for c in candidates if not (set(_norm(c).split()) & banned)]
    for syn in _SYNONYMS[role]:
        for c, n in pool:
            if n == syn:
                return c
    for syn in _SYNONYMS[role]:
        syn_tokens = set(syn.split())
        hits = [(len(n.split()) - len(syn_tokens), c) for c, n in pool
                if syn_tokens <= set(n.split())]
        if hits:
            return min(hits)[1]
    return None


@dataclass
class Schema:
    date: str
    revenue: str
    profit: str | None = None
    quantity: str | None = None
    discount: str | None = None
    order_id: str | None = None
    dimensions: list[str] = field(default_factory=list)
    cardinality: dict[str, int] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)  # dropped column -> kept column
    date_format: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                (*ROLE_COLUMNS, "dimensions", "cardinality", "aliases", "date_format")}


def _find_dimensions(df: pd.DataFrame, used: set[str]):
    n_rows = len(df)
    dims, card = [], {}
    id_hints = {"id", "code", "store", "branch", "zone", "group", "type", "class", "channel"}
    for col in df.columns:
        if col in used:
            continue
        s = df[col]
        if _looks_like_date(s, col) or pd.api.types.is_bool_dtype(s):
            continue
        nun = int(s.nunique(dropna=True))
        if nun < 2:
            continue
        if pd.api.types.is_numeric_dtype(s):
            # only small integer-coded categories like "Store ID" count as dimensions
            is_int = pd.api.types.is_integer_dtype(s)
            if not (is_int and nun <= 200 and (set(_norm(col).split()) & id_hints)):
                continue
        else:
            if nun > 20000 or (n_rows > 100 and nun / n_rows > 0.6):
                continue  # looks like a unique key or free text
        dims.append(col)
        card[col] = nun

    # drop duplicate pairs such as "Customer ID" + "Customer Name" (1-to-1)
    aliases: dict[str, str] = {}
    generic = {"id", "code", "name", "number", "no", "key"}
    for a in list(dims):
        for b in list(dims):
            if a >= b or a in aliases or b in aliases or card[a] != card[b]:
                continue
            # Only merge pairs that look like the same thing ("Customer ID" and
            # "Customer Name" share the word "customer") and only when there are
            # enough rows for a 1-to-1 match to be more than a coincidence.
            shared = (set(_norm(a).split()) & set(_norm(b).split())) - generic
            if not shared or n_rows < 30:
                continue
            ab = df.groupby(a, dropna=False)[b].nunique().max()
            ba = df.groupby(b, dropna=False)[a].nunique().max()
            if ab == 1 and ba == 1:
                id_like = lambda c: bool(set(_norm(c).split()) & {"id", "code", "number", "no"})  # noqa: E731
                drop, keep = (a, b) if id_like(a) and not id_like(b) else (b, a)
                aliases[drop] = keep
    dims = [d for d in dims if d not in aliases]
    dims.sort(key=lambda c: (card[c], c))
    return dims, card, aliases


def build_schema(df: pd.DataFrame, overrides: dict | None = None):
    """Guess the schema. `overrides` lets the user force a role, e.g.
    {"revenue": "Net Amount", "profit": None}. Returns (Schema, warnings)."""
    overrides = overrides or {}
    cols = list(df.columns)
    warnings: list[str] = []

    for role, value in overrides.items():
        if role not in ROLE_COLUMNS:
            raise IngestError(f"Unknown role '{role}'.")
        if value is not None and value not in cols:
            raise IngestError(f"Column '{value}' is not in the file.")

    def pick(role, pool):
        if role in overrides:
            return overrides[role]
        return _pick(role, pool)

    # date
    date_pool = [c for c in cols if _looks_like_date(df[c], c)
                 or (pd.api.types.is_numeric_dtype(df[c]) and _norm(c) in _SYNONYMS["date"])]
    date_col = pick("date", date_pool)
    if date_col is None and date_pool:
        banned = _EXCLUDE["date"]
        good = [c for c in date_pool if not (set(_norm(c).split()) & banned)]
        date_col = (good or date_pool)[0]
        warnings.append(f"Used '{date_col}' as the date column (best guess).")
    if date_col is None:
        raise IngestError("No date column found. InsightTrace needs a column with "
                          "dates (like 'Order Date') to track changes over time.")

    # numeric measures
    numeric_pool = []
    for c in cols:
        if c == date_col:
            continue
        _, ratio = to_numeric(df[c])
        if ratio >= 0.9 and not (set(_norm(c).split()) & {"id"}):
            numeric_pool.append(c)

    revenue = pick("revenue", numeric_pool)
    if revenue is None:
        raise IngestError("No sales/revenue column found. Rename the money column "
                          "to something like 'Sales' or 'Revenue', or pick it manually.")
    profit = pick("profit", [c for c in numeric_pool if c != revenue])
    quantity = pick("quantity", [c for c in numeric_pool if c not in (revenue, profit)])
    discount = pick("discount", [c for c in numeric_pool if c not in (revenue, profit, quantity)])
    order_id = pick("order_id", [c for c in cols if c not in (date_col, revenue)])

    used = {c for c in (date_col, revenue, profit, quantity, discount, order_id) if c}
    dims, card, aliases = _find_dimensions(df, used)
    if not dims:
        warnings.append("No category columns (like Region or Product) found, so "
                        "InsightTrace can detect changes but not explain them.")
    return Schema(date_col, revenue, profit, quantity, discount, order_id,
                  dims, card, aliases), warnings


# ----------------------------------------------------------------------------
# 4. Loading into DuckDB
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    con: duckdb.DuckDBPyConnection
    schema: Schema
    name: str
    n_rows: int
    min_date: pd.Timestamp
    max_date: pd.Timestamp
    warnings: list[str]

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        """Run SQL. A fresh cursor per call keeps this safe inside Streamlit."""
        cur = self.con.cursor()
        try:
            return cur.execute(sql, params or []).df()
        finally:
            cur.close()


def _prepare_frame(df: pd.DataFrame, schema: Schema):
    """Build the tidy table: standard column names for roles, original names
    for dimensions."""
    warnings: list[str] = []
    dates, fmt, w = parse_dates(df[schema.date])
    warnings += w
    schema.date_format = fmt

    out = pd.DataFrame({"order_date": dates})
    out["revenue"], _ = to_numeric(df[schema.revenue])
    for role in ("profit", "quantity", "discount"):
        col = getattr(schema, role)
        if col:
            out[role], _ = to_numeric(df[col])
    if schema.order_id:
        out["order_id"] = df[schema.order_id].astype(str).astype(object)

    reserved = set(out.columns)
    renamed = {}
    for d in schema.dimensions:
        target = d if d not in reserved else f"{d} (category)"
        renamed[d] = target
        out[target] = df[d].astype("object").where(df[d].notna(), "(missing)").astype(str).astype(object)
    schema.dimensions = [renamed[d] for d in schema.dimensions]
    schema.cardinality = {renamed.get(k, k): v for k, v in schema.cardinality.items()}

    before = len(out)
    out = out[out["order_date"].notna() & out["revenue"].notna()]
    dropped = before - len(out)
    if dropped:
        warnings.append(f"Skipped {dropped:,} row{'s' if dropped != 1 else ''} with a missing date or revenue "
                        f"({dropped / before:.1%} of the file).")
    if out.empty:
        raise IngestError("No valid rows left after cleaning dates and revenue.")
    if schema.discount and out["discount"].dropna().max() > 1.5:
        warnings.append("Discount values look like percentages (e.g. 15 instead of 0.15).")
    return out.reset_index(drop=True), warnings


def load_dataset(source, filename: str | None = None, overrides: dict | None = None,
                 sheet=None) -> Dataset:
    """Main entry point: file in, ready-to-query Dataset out."""
    raw_df, notes = read_table(source, filename, sheet)
    _, name = _get_bytes(source, filename) if not isinstance(source, (str, Path)) else (None, Path(source).name)
    schema, warnings = build_schema(raw_df, overrides)
    tidy, w2 = _prepare_frame(raw_df, schema)

    con = duckdb.connect(":memory:")
    con.register("_tidy", tidy)
    con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM _tidy")
    con.unregister("_tidy")
    con.execute(f"ALTER TABLE {TABLE} ALTER order_date TYPE DATE")

    return Dataset(
        con=con, schema=schema, name=filename or name, n_rows=len(tidy),
        min_date=pd.Timestamp(tidy["order_date"].min()),
        max_date=pd.Timestamp(tidy["order_date"].max()),
        warnings=notes + warnings + w2,
    )


def describe(ds: Dataset) -> str:
    s = ds.schema
    lines = [
        f"File: {ds.name}",
        f"Rows: {ds.n_rows:,}   Dates: {ds.min_date.date()} to {ds.max_date.date()}",
        f"Date column: {s.date}  ({s.date_format})",
        f"Revenue: {s.revenue}",
        f"Profit: {s.profit}   Quantity: {s.quantity}   Discount: {s.discount}   Order id: {s.order_id}",
        "Dimensions (smallest to largest): "
        + ", ".join(f"{d} ({s.cardinality[d]})" for d in s.dimensions),
    ]
    if s.aliases:
        lines.append("Merged duplicates: " + ", ".join(f"{k} -> {v}" for k, v in s.aliases.items()))
    for w in ds.warnings:
        lines.append(f"Note: {w}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m engine.ingest <path to csv or xlsx>")
        sys.exit(1)
    try:
        print(describe(load_dataset(sys.argv[1])))
    except IngestError as err:
        print(f"Could not load the file: {err}")
        sys.exit(1)
