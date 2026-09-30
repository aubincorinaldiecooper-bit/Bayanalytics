"""Tables in the pages a web search returned: daily price history and reported figures.

Data comes from web search only (see the repository's CLAUDE.md). This module reads the body
the fetcher already retrieved for a search hit and nothing else: HTML ``<table>`` elements, a
CSV or JSON response (when the search result URL itself is one) and simple JSON tables embedded
in the page (``application/ld+json`` and ``application/json`` script blocks). It never follows
a link found inside a page ("Download CSV" and the like): nothing here produces a URL, and the
provider refuses any URL its own search did not return.

Three layers:

* :func:`html_tables`, :func:`json_tables` and :func:`page_tables` turn a page into bounded
  :class:`RawTable` objects: header cells, rows of cell text, and the caption plus the text just
  before the table (where units such as "(in millions)" are written);
* :func:`price_candidates` and :func:`figure_candidates` are the deterministic filter: tables
  that *could* be a daily price history (a date column with daily rows and numeric columns) or
  reported figures (dated periods and lines of numbers), with :func:`shortlist` naming the
  lines worth offering for one metric. Which table, which column is the close and which line
  is which metric is decided by Laya among these options (``research.selection``), never here;
* :func:`read_price_history` and :func:`read_figures` read what was chosen: dates, numbers,
  units, currency, the ``as_of`` cut and the plausibility checks.

Nothing is inferred or filled in. A period needs an end date written on the page (a fiscal
label such as "Q3 2026" alone does not fix its dates, see ``normalization.periods``); a price
series needs at least ``MIN_PRICE_ROWS`` valid daily rows; a value that fails a plausibility
check is dropped, never repaired. Tables a browser builds with JavaScript are not in the body
and are therefore not seen.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from itertools import pairwise
from statistics import median
from typing import Any

from bayanalytics.normalization.units import (
    CURRENCY_SYMBOL_TO_CODE,
    ParsedNumber,
    try_parse_number,
)
from bayanalytics.schemas.evidence import PricePoint

log = logging.getLogger(__name__)

MAX_TABLES = 24
MAX_TABLE_ROWS = 4000
MAX_TABLE_COLS = 40
MAX_CELL_CHARS = 200
MAX_CONTEXT_CHARS = 400
MAX_JSON_SCRIPT_CHARS = 1_000_000
MAX_JSON_NODES = 50_000
MAX_JSON_DEPTH = 8
MIN_PRICE_ROWS = 20
MAX_DAILY_MEDIAN_GAP_DAYS = 5
MIN_FIGURE_YEAR = 1970
MAX_MONEY = 1e15  # a larger amount is a scale error (e.g. full units tagged "in millions")
MAX_SHARES = 1e12
MAX_ABS_EPS = 1e5

QUARTER = "fiscal_quarter"
YEAR = "fiscal_year"
INSTANT = "instant"

PER_SHARE_METRICS = frozenset({"eps_diluted", "eps_basic"})
SHARE_METRICS = frozenset({"shares_outstanding"})

_WS = re.compile(r"\s+")
_MINUS = "\u2212\u2013\u2014"


# --------------------------------------------------------------------------------------
# raw tables
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class RawTable:
    """One table as the page wrote it: header cells, rows of cell text and nearby text."""

    columns: list[str]
    rows: list[list[str]]
    context: str = ""
    origin: str = "html"  # html | csv | json

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "context": self.context,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RawTable | None:
        if not isinstance(data, dict):
            return None
        columns, rows = data.get("columns"), data.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            return None
        return _bounded(
            [_cell(c) for c in columns],
            [[_cell(c) for c in row] for row in rows if isinstance(row, list)],
            str(data.get("context") or ""),
            str(data.get("origin") or "html"),
        )


def _cell(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value.is_integer() and abs(value) < 1e18:
            text = str(int(value))
        else:
            text = f"{value:.8f}".rstrip("0").rstrip(".")
    elif isinstance(value, (dict, list)):
        return ""
    else:
        text = str(value)
    return _WS.sub(" ", text).strip()[:MAX_CELL_CHARS]


def _bounded(
    columns: list[str], rows: list[list[str]], context: str, origin: str
) -> RawTable | None:
    """Cap columns, rows and context; pad rows to one width. ``None`` for fewer than two
    columns or no rows."""
    rows = rows[:MAX_TABLE_ROWS]
    width = min(MAX_TABLE_COLS, max([len(columns), *(len(r) for r in rows)] or [0]))
    if width < 2 or not rows:
        return None
    columns = (columns + [""] * width)[:width]
    rows = [(row + [""] * width)[:width] for row in rows]
    return RawTable(columns, rows, _WS.sub(" ", context).strip()[:MAX_CONTEXT_CHARS], origin)


class _OpenTable:
    """A ``<table>`` being parsed (cells expanded by ``colspan``)."""

    def __init__(self, context: str) -> None:
        self.context = context
        self.rows: list[tuple[list[str], bool]] = []  # (cells, is_header_row)
        self.row: list[tuple[str, bool, int]] | None = None
        self.cell: list[str] | None = None
        self.cell_header = False
        self.cell_span = 1
        self.caption: list[str] = []
        self.in_caption = False
        self.in_thead = False

    def start_row(self) -> None:
        self.end_row()
        self.row = []

    def start_cell(self, header: bool, span: int) -> None:
        if self.row is None:
            self.row = []
        self.end_cell()
        self.cell = []
        self.cell_header = header
        self.cell_span = span

    def end_cell(self) -> None:
        if self.cell is not None and self.row is not None:
            self.row.append((_cell("".join(self.cell)), self.cell_header, self.cell_span))
        self.cell = None

    def end_row(self) -> None:
        self.end_cell()
        row, self.row = self.row, None
        if not row or len(self.rows) >= MAX_TABLE_ROWS + 8:
            return
        cells: list[str] = []
        for text, _header, span in row:
            cells.extend([text] * span)
            if len(cells) >= MAX_TABLE_COLS:
                break
        filled = [(text, header) for text, header, _span in row if text]
        header_row = self.in_thead or (bool(filled) and all(header for _t, header in filled))
        self.rows.append((cells[:MAX_TABLE_COLS], header_row))

    def text(self, data: str) -> None:
        if self.in_caption:
            self.caption.append(data)
        elif self.cell is not None:
            self.cell.append(data)

    def finish(self) -> RawTable | None:
        self.end_row()
        if len(self.rows) < 2:
            return None
        count = 0
        while count < min(4, len(self.rows) - 1) and self.rows[count][1]:
            count += 1
        count = max(count, 1)
        header_rows = [cells for cells, _ in self.rows[:count]]
        body = [cells for cells, _ in self.rows[count:]]
        width = min(MAX_TABLE_COLS, max(len(c) for c in header_rows + body))
        columns: list[str] = []
        for j in range(width):
            parts: list[str] = []
            for cells in header_rows:
                if j < len(cells) and cells[j] and cells[j] not in parts:
                    parts.append(cells[j])
            columns.append(" ".join(parts))
        caption = _cell(" ".join(self.caption))
        context = " | ".join(part for part in (caption, self.context) if part)
        return _bounded(columns, body, context, "html")


_SKIPPED_TAGS = frozenset({"script", "style", "template", "svg"})


class _TableParser(HTMLParser):
    """Collect ``<table>`` elements and JSON script blocks; iterative, no DOM kept."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[RawTable] = []
        self.json_blobs: list[str] = []
        self._open: list[_OpenTable] = []
        self._recent = ""
        self._skip = 0
        self._json: list[str] | None = None
        self._json_chars = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            kind = (dict(attrs).get("type") or "").lower()
            if "json" in kind and self._skip == 0:
                self._json, self._json_chars = [], 0
            else:
                self._skip += 1
            return
        if tag in _SKIPPED_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "table":
            self._open.append(_OpenTable(self._recent[-MAX_CONTEXT_CHARS:]))
            return
        if not self._open:
            if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "section"):
                self._remember(" | ")
            return
        table = self._open[-1]
        if tag == "caption":
            table.in_caption = True
        elif tag == "thead":
            table.in_thead = True
        elif tag in ("tbody", "tfoot"):
            table.end_row()
            table.in_thead = False
        elif tag == "tr":
            table.start_row()
        elif tag in ("td", "th"):
            try:
                span = int(dict(attrs).get("colspan") or 1)
            except ValueError:
                span = 1
            table.start_cell(tag == "th", max(1, min(span, MAX_TABLE_COLS)))
        elif tag == "br":
            table.text(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            if self._json is not None:
                blob, self._json = "".join(self._json), None
                if blob.strip() and len(self.json_blobs) < MAX_TABLES:
                    self.json_blobs.append(blob)
            elif self._skip:
                self._skip -= 1
            return
        if tag in _SKIPPED_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip or not self._open:
            return
        table = self._open[-1]
        if tag == "table":
            self._open.pop()
            finished = table.finish()
            if finished is not None and len(self.tables) < MAX_TABLES:
                self.tables.append(finished)
        elif tag == "caption":
            table.in_caption = False
        elif tag == "thead":
            table.end_row()
            table.in_thead = False
        elif tag == "tr":
            table.end_row()
        elif tag in ("td", "th"):
            table.end_cell()

    def handle_data(self, data: str) -> None:
        if self._json is not None:
            if self._json_chars <= MAX_JSON_SCRIPT_CHARS:
                self._json.append(data)
                self._json_chars += len(data)
            return
        if self._skip:
            return
        if self._open:
            self._open[-1].text(data)
        else:
            self._remember(data)

    def _remember(self, text: str) -> None:
        self._recent = (self._recent + text)[-2 * MAX_CONTEXT_CHARS :]

    def close(self) -> None:
        super().close()
        while self._open:
            finished = self._open.pop().finish()
            if finished is not None and len(self.tables) < MAX_TABLES:
                self.tables.append(finished)


def html_tables(html: str) -> list[RawTable]:
    """The page's ``<table>`` elements and the tables in its JSON script blocks (bounded)."""
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    tables = list(parser.tables)
    for blob in parser.json_blobs:
        if len(tables) >= MAX_TABLES or len(blob) > MAX_JSON_SCRIPT_CHARS:
            break
        try:
            data = json.loads(blob)
        except (ValueError, RecursionError):
            continue
        tables.extend(json_tables(data)[: MAX_TABLES - len(tables)])
    return tables


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float))


def _records_table(items: list[Any], key: str) -> RawTable | None:
    """A list of objects sharing keys (``[{"date": ..., "close": ...}, ...]``)."""
    records = [item for item in items[:MAX_TABLE_ROWS] if isinstance(item, dict)]
    if len(records) < 2 or len(records) < len(items[:MAX_TABLE_ROWS]) // 2:
        return None
    columns: list[str] = []
    for record in records[:5]:
        for name, value in record.items():
            if isinstance(name, str) and _scalar(value) and name not in columns:
                columns.append(name)
    if len(columns) < 2:
        return None
    rows = [[_cell(record.get(name)) for name in columns] for record in records]
    return _bounded(columns[:MAX_TABLE_COLS], rows, key, "json")


def _columnar_table(node: dict[str, Any], key: str) -> RawTable | None:
    """``{"columns": [...], "data": [[...], ...]}`` or ``{"date": [...], "close": [...]}``."""
    columns, data = node.get("columns"), node.get("data")
    if (
        isinstance(columns, list)
        and isinstance(data, list)
        and len(data) >= 2
        and all(isinstance(c, str) for c in columns)
        and all(isinstance(row, list) for row in data[:MAX_TABLE_ROWS])
    ):
        rows = [[_cell(v) for v in row] for row in data[:MAX_TABLE_ROWS]]
        return _bounded([_cell(c) for c in columns], rows, key, "json")
    arrays = {
        name: value
        for name, value in node.items()
        if isinstance(name, str)
        and isinstance(value, list)
        and len(value) >= 2
        and all(_scalar(v) for v in value[:MAX_TABLE_ROWS])
    }
    lengths = {len(v) for v in arrays.values()}
    if len(arrays) < 2 or len(lengths) != 1:
        return None
    names = list(arrays)[:MAX_TABLE_COLS]
    length = min(lengths.pop(), MAX_TABLE_ROWS)
    rows = [[_cell(arrays[name][i]) for name in names] for i in range(length)]
    return _bounded(names, rows, key, "json")


def json_tables(data: Any) -> list[RawTable]:
    """Record lists and column arrays found in a JSON document (bounded walk, no recursion)."""
    found: list[RawTable] = []
    stack: list[tuple[Any, int, str]] = [(data, 0, "")]
    nodes = 0
    while stack and len(found) < MAX_TABLES and nodes < MAX_JSON_NODES:
        node, depth, key = stack.pop()
        nodes += 1
        if isinstance(node, list):
            table = _records_table(node, key)
            if table is not None:
                found.append(table)
                continue
            if depth < MAX_JSON_DEPTH:
                stack.extend((item, depth + 1, key) for item in reversed(node[:200]))
        elif isinstance(node, dict):
            table = _columnar_table(node, key)
            if table is not None:
                found.append(table)
                continue
            if depth < MAX_JSON_DEPTH:
                items = list(node.items())[:200]
                stack.extend((value, depth + 1, str(name)) for name, value in reversed(items))
    return found


def csv_table(columns: list[Any], rows: list[Any]) -> RawTable | None:
    return _bounded(
        [_cell(c) for c in columns],
        [[_cell(c) for c in row] for row in rows[:MAX_TABLE_ROWS] if isinstance(row, list)],
        "",
        "csv",
    )


def page_tables(extraction_method: str, structured: dict[str, Any] | None) -> list[RawTable]:
    """The raw tables of one extracted page (``EvidenceRecord.extraction_method`` /
    ``.structured``): a CSV response is one table, a JSON response is walked for tables, an
    HTML page carries the tables ``extract_html`` collected."""
    structured = structured or {}
    if extraction_method == "csv":
        table = csv_table(structured.get("columns") or [], structured.get("rows") or [])
        return [table] if table is not None else []
    if extraction_method == "json":
        if structured.get("parse_error"):
            return []
        data = structured.get("items") if set(structured) == {"items"} else structured
        return json_tables(data)[:MAX_TABLES]
    tables = structured.get("tables")
    if not isinstance(tables, list):
        return []
    parsed = (RawTable.from_dict(item) for item in tables[:MAX_TABLES])
    return [table for table in parsed if table is not None]


# --------------------------------------------------------------------------------------
# cells: dates, numbers, headers
# --------------------------------------------------------------------------------------

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DAY_YEAR_SEP = r"(?:\s*,\s*|\s+|\s*-\s*)"
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})([-/.])(\d{1,2})\2(\d{1,2})(?!\d)")
_NAME_MDY = re.compile(
    rf"(?<![a-z]){_MONTH}\s*-?\s*(\d{{1,2}})(?:st|nd|rd|th)?{_DAY_YEAR_SEP}'?(\d{{4}}|\d{{2}})(?!\d)",
    re.I,
)
_NAME_DMY = re.compile(
    rf"(?<![\d.])(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+|\s*-\s*|\s*\.\s*)?{_MONTH}{_DAY_YEAR_SEP}'?(\d{{4}}|\d{{2}})(?!\d)",
    re.I,
)
_NUMERIC_DATE = re.compile(r"(?<![\d.])(\d{1,2})([/.-])(\d{1,2})\2(\d{4}|\d{2})(?![\d.])")
_COMPACT_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_EPOCH = re.compile(r"^(\d{10}|\d{13})(?:\.\d+)?$")


def _year(text: str) -> int:
    year = int(text)
    if year < 100:
        year += 2000 if year < 70 else 1900
    return year


def _safe_date(year: int, month: int, day: int) -> date | None:
    if not 1900 <= year <= 2100:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def date_order(texts: list[str]) -> str | None:
    """How the column writes numeric dates: ``"mdy"``, ``"dmy"``, ``"conflict"`` (both seen) or
    ``None`` (no numeric date, or every one ambiguous)."""
    mdy = dmy = False
    for text in texts:
        if _ISO_DATE.search(text) or _NAME_MDY.search(text) or _NAME_DMY.search(text):
            continue
        match = _NUMERIC_DATE.search(text)
        if match is None:
            continue
        first, second = int(match.group(1)), int(match.group(3))
        if first > 12 >= second:
            dmy = True
        elif second > 12 >= first:
            mdy = True
    if mdy and dmy:
        return "conflict"
    return "mdy" if mdy else "dmy" if dmy else None


def parse_table_date(text: str, order: str | None = None, *, epoch: bool = False) -> date | None:
    """A calendar date written in a cell (ISO, month names, numeric, compact ``YYYYMMDD``;
    epoch seconds or milliseconds only when ``epoch``, i.e. in a column headed as a date).

    Numeric day/month order comes from ``order`` (see :func:`date_order`); without it a date is
    read only when it is unambiguous (a part above 12, or both parts equal). Two-digit years are
    19xx from 70 and 20xx below; years outside 1900-2100 are not dates. ``None`` when nothing
    parses: a date is never guessed.
    """
    text = (text or "").strip()
    if not text or len(text) > 80:
        return None
    match = _COMPACT_DATE.match(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _EPOCH.match(text)
    if match and epoch:
        seconds = float(text) / (1000 if len(match.group(1)) == 13 else 1)
        try:
            parsed = datetime.fromtimestamp(seconds, tz=UTC).date()
        except (OverflowError, OSError, ValueError):
            return None
        return parsed if 1970 <= parsed.year <= 2100 else None
    match = _ISO_DATE.search(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(3)), int(match.group(4)))
    match = _NAME_MDY.search(text)
    if match:
        month = _MONTHS[match.group(1).lower()[:3]]
        return _safe_date(_year(match.group(3)), month, int(match.group(2)))
    match = _NAME_DMY.search(text)
    if match:
        month = _MONTHS[match.group(2).lower()[:3]]
        return _safe_date(_year(match.group(3)), month, int(match.group(1)))
    match = _NUMERIC_DATE.search(text)
    if match:
        first, second, year = int(match.group(1)), int(match.group(3)), _year(match.group(4))
        if order == "conflict":
            return None
        if order == "dmy" or (order is None and first > 12 >= second):
            return _safe_date(year, second, first)
        if order == "mdy" or (order is None and second > 12 >= first):
            return _safe_date(year, first, second)
        if first == second:
            return _safe_date(year, first, second)
    return None


_MISSING = frozenset({"", "-", "--", "n/a", "na", "nm", "n.m.", "none", "null", "nan"})
_FOOTNOTE = re.compile(r"(?:\s*\(\d{1,2}\)|\s*\[\d{1,2}\]|[*\u2020\u2021]+)$")


def parse_cell_number(text: str) -> ParsedNumber | None:
    """A number written in a cell (``$1,234.5``, ``(1,234)``, ``45.1M``, ``12%``), or ``None``."""
    text = (text or "").strip().replace("\u00a0", " ").replace("\u2009", "")
    if text.lower() in _MISSING or all(ch in _MINUS for ch in text):
        return None
    text = _FOOTNOTE.sub("", text).strip()
    parsed = try_parse_number(text)
    if parsed is None or not math.isfinite(parsed.value):
        return None
    return parsed


def _split_camel(text: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)


def norm_header(text: str) -> str:
    """Lower-case words of a header or row label (``"Close*"`` -> ``"close"``,
    ``"adjClose"`` -> ``"adj close"``, ``"EPS (Diluted)"`` -> ``"eps diluted"``)."""
    text = _split_camel(text or "").lower().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9%]+", " ", text).split())


_CURRENCY_RE = re.compile(
    r"(US\$|HK\$|NZ\$|CA\$|AU\$|C\$|A\$|S\$|R\$|CN\u00a5|\bUSD\b|\bEUR\b|\bGBP\b|\bJPY\b|\bCHF\b"
    r"|\bCAD\b|\bAUD\b|\bHKD\b|\bCNY\b|\bINR\b|\bKRW\b|\bSEK\b|\bNOK\b|\bDKK\b|\bSGD\b|\bTWD\b"
    r"|\bBRL\b|\bMXN\b|\bZAR\b|\bNZD\b|\$|\u20ac|\u00a3|\u00a5|\u20b9|\u20a9)"
)


def _currencies(texts: list[str]) -> list[str]:
    found: list[str] = []
    for text in texts:
        for match in _CURRENCY_RE.finditer(text):
            code = CURRENCY_SYMBOL_TO_CODE.get(match.group(1), match.group(1))
            if code not in found:
                found.append(code)
    return found


def _table_currency(cells: list[str], context: list[str]) -> tuple[str | None, bool]:
    """(currency, stated): the one currency the cells (else the headers and context) name;
    ``(None, True)`` when they name several; ``("USD", False)`` when none is written."""
    for texts in (cells, context):
        found = _currencies(texts)
        if len(found) == 1:
            return found[0], True
        if len(found) > 1:
            return None, True
    return "USD", False


# --------------------------------------------------------------------------------------
# price history
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PriceTable:
    """A daily price history read from one table, oldest first, on or before ``as_of``."""

    points: tuple[PricePoint, ...]
    currency: str
    currency_stated: bool
    after_as_of: int = 0
    rejected_rows: int = 0
    origin: str = "html"


@dataclass(frozen=True, slots=True)
class PriceCandidate:
    """A table that could be a daily price history: a date column with at least
    ``MIN_PRICE_ROWS`` daily rows on or before ``as_of`` and the columns holding positive
    numbers on those rows (one of which Laya may name as the closing price)."""

    index: int
    table: RawTable
    date_col: int
    order: str | None
    value_cols: tuple[int, ...]
    rows: int

    def headers(self) -> str:
        return " | ".join(cell or "?" for cell in self.table.columns)


_NOT_AMOUNTS = frozenset({"percent", "ratio", "bp", "days", "shares"})


def _is_date_header(name: str) -> bool:
    words = name.split()
    return "date" in words or "time" in words or "timestamp" in words or name == "day"


def _positive(text: str) -> float | None:
    parsed = parse_cell_number(text)
    if parsed is None or parsed.unit in _NOT_AMOUNTS or parsed.value <= 0:
        return None
    return parsed.value


def _dated_rows(
    table: RawTable, col: int, order: str | None, epoch: bool
) -> list[tuple[int, date]]:
    out: list[tuple[int, date]] = []
    for i, row in enumerate(table.rows):
        day = parse_table_date(row[col], order, epoch=epoch)
        if day is not None:
            out.append((i, day))
    return out


def price_candidates(tables: list[RawTable], as_of: date) -> list[PriceCandidate]:
    """Tables that could hold a daily price history, in page order (deterministic filter;
    which one is the price history, and which column the close, is Laya's choice)."""
    found: list[PriceCandidate] = []
    for index, table in enumerate(tables):
        norm = [norm_header(c) for c in table.columns]
        named = [j for j, name in enumerate(norm) if name and _is_date_header(name)]
        columns = named or [j for j in range(min(2, len(table.columns)))]
        best: tuple[int, str | None, list[tuple[int, date]]] | None = None
        for j in columns:
            order = date_order([row[j] for row in table.rows])
            dated = _dated_rows(table, j, order, epoch=j in named)
            if len(dated) >= MIN_PRICE_ROWS and (best is None or len(dated) > len(best[2])):
                best = (j, order, dated)
        if best is None:
            continue
        date_col, order, dated = best
        kept = [(i, day) for i, day in dated if day <= as_of]
        days = sorted({day for _, day in kept})
        if len(days) < MIN_PRICE_ROWS:
            continue
        gaps = [(b - a).days for a, b in pairwise(days)]
        if median(gaps) > MAX_DAILY_MEDIAN_GAP_DAYS:
            continue
        value_cols = tuple(
            j
            for j in range(len(table.columns))
            if j != date_col
            and sum(1 for i, _ in kept if _positive(table.rows[i][j]) is not None)
            >= 0.8 * len(kept)
        )
        if value_cols:
            found.append(PriceCandidate(index, table, date_col, order, value_cols, len(kept)))
    return found


def read_price_history(candidate: PriceCandidate, close_col: int, as_of: date) -> PriceTable | None:
    """Read the candidate as a daily price history with ``close_col`` as the close.

    Open, High, Low and Volume are read when a column is headed so. A row needs its date and a
    positive close; rows dated after ``as_of`` are dropped (no look-ahead), a row whose close
    lies outside its own low-high range is rejected, and a repeated date keeps its first row.
    ``None`` below ``MIN_PRICE_ROWS`` rows, when the dates are not daily, or when the table
    names several currencies. Volume accepts thousands separators and K/M/B suffixes.
    """
    table = candidate.table
    if close_col == candidate.date_col or not 0 <= close_col < len(table.columns):
        return None
    norm = [norm_header(c) for c in table.columns]
    extra: dict[str, int] = {}
    for key, words in (
        ("open", ("open",)),
        ("high", ("high",)),
        ("low", ("low",)),
        ("volume", ("volume", "vol")),
    ):
        for j, name in enumerate(norm):
            if j not in (close_col, candidate.date_col) and name and name.split()[0] in words:
                extra[key] = j
                break
    points: dict[date, PricePoint] = {}
    after = rejected = 0
    epoch = _is_date_header(norm[candidate.date_col])
    for row in table.rows:
        day = parse_table_date(row[candidate.date_col], candidate.order, epoch=epoch)
        close = _positive(row[close_col])
        if day is None or close is None:
            rejected += 1
            continue
        if day > as_of:
            after += 1
            continue
        high = _positive(row[extra["high"]]) if "high" in extra else None
        low = _positive(row[extra["low"]]) if "low" in extra else None
        if (
            high is not None
            and low is not None
            and (low > high or close > high * 1.01 or close < low * 0.99)
        ):
            rejected += 1
            continue
        volume: float | None = None
        if "volume" in extra:
            parsed = parse_cell_number(row[extra["volume"]])
            if parsed is not None and parsed.value >= 0 and parsed.unit == "number":
                volume = parsed.value
        if day in points:
            continue
        points[day] = PricePoint(
            date=day,
            open=_positive(row[extra["open"]]) if "open" in extra else None,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
    if len(points) < MIN_PRICE_ROWS:
        return None
    ordered = sorted(points.values(), key=lambda p: p.date)
    gaps = [(b.date - a.date).days for a, b in pairwise(ordered)]
    if median(gaps) > MAX_DAILY_MEDIAN_GAP_DAYS:
        return None
    currency, stated = _table_currency(
        [row[close_col] for row in table.rows[:40]], [table.columns[close_col], table.context]
    )
    if currency is None:
        return None
    return PriceTable(
        points=tuple(ordered),
        currency=currency,
        currency_stated=stated,
        after_as_of=after,
        rejected_rows=rejected,
        origin=table.origin,
    )


# --------------------------------------------------------------------------------------
# reported figures
# --------------------------------------------------------------------------------------

FIGURE_METRICS: tuple[str, ...] = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "eps_diluted",
    "eps_basic",
    "operating_cash_flow",
    "capex",
    "free_cash_flow",
    "shares_outstanding",
)

# Exact labels that name a metric (after ``norm_header``). A shortlist ranks them first.
_METRIC_LABELS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "revenue",
        "revenues",
        "total revenue",
        "total revenues",
        "net revenue",
        "net revenues",
        "total net revenue",
        "total net revenues",
        "net sales",
        "total net sales",
        "sales",
        "total sales",
        "operating revenue",
        "total operating revenue",
    ),
    "gross_profit": ("gross profit", "total gross profit", "gross income", "gross margin"),
    "operating_income": (
        "operating income",
        "operating income loss",
        "total operating income",
        "income from operations",
        "income loss from operations",
        "operating profit",
    ),
    "net_income": (
        "net income",
        "net income loss",
        "net earnings",
        "net earnings loss",
        "net profit",
        "net income common stockholders",
        "net income attributable to common stockholders",
    ),
    "eps_diluted": (
        "diluted eps",
        "eps diluted",
        "diluted earnings per share",
        "earnings per share diluted",
        "diluted net income per share",
        "net income per share diluted",
        "eps",
        "earnings per share",
    ),
    "eps_basic": (
        "basic eps",
        "eps basic",
        "basic earnings per share",
        "earnings per share basic",
        "basic net income per share",
        "net income per share basic",
    ),
    "operating_cash_flow": (
        "operating cash flow",
        "cash flow from operations",
        "cash from operations",
        "cash provided by operating activities",
        "net cash provided by operating activities",
        "net cash from operating activities",
        "cash generated by operating activities",
    ),
    "capex": (
        "capital expenditure",
        "capital expenditures",
        "capex",
        "purchases of property and equipment",
        "payments for acquisition of property plant and equipment",
    ),
    "free_cash_flow": ("free cash flow", "fcf"),
    "shares_outstanding": (
        "shares outstanding",
        "diluted shares outstanding",
        "shares outstanding diluted",
        "basic shares outstanding",
        "shares outstanding basic",
    ),
}
# Every keyword group must match for a label to be offered for the metric.
_METRIC_KEYWORDS: dict[str, tuple[frozenset[str], ...]] = {
    "revenue": (frozenset({"revenue", "revenues", "sales", "turnover"}),),
    "gross_profit": (frozenset({"gross"}), frozenset({"profit", "income", "margin"})),
    "operating_income": (
        frozenset({"operating", "operations"}),
        frozenset({"income", "profit", "earnings", "loss"}),
    ),
    "net_income": (frozenset({"net"}), frozenset({"income", "earnings", "profit"})),
    "eps_diluted": (frozenset({"diluted"}), frozenset({"eps", "share"})),
    "eps_basic": (frozenset({"basic"}), frozenset({"eps", "share"})),
    "operating_cash_flow": (frozenset({"operating", "operations"}), frozenset({"cash"})),
    "capex": (frozenset({"capital", "capex", "property"}),),
    "free_cash_flow": (frozenset({"free", "fcf"}),),
    "shares_outstanding": (frozenset({"shares"}), frozenset({"outstanding"})),
}
_EXCLUDED_WORDS = frozenset(
    {
        "adjusted",
        "adj",
        "non",
        "pro",
        "normalized",
        "growth",
        "yoy",
        "qoq",
        "change",
        "ratio",
        "ttm",
        "employee",
        "estimate",
        "estimates",
        "forecast",
        "guidance",
        "consensus",
        "dividend",
        "dividends",
        "cost",
        "costs",
        "expense",
        "expenses",
        "before",
        "proceeds",
        "sale",
        "%",
    }
)
_PLAIN_EPS = frozenset({"eps", "earnings per share"})
_SECTION_QUALIFIED = frozenset({"basic", "diluted", "products", "services", "total", "other"})
_UNIT_GROUP = re.compile(r"[(\[]([^)\]]{0,48})[)\]]")
_ROW_SCALE = (
    re.compile(
        r"^\s*(?:in\s+)?(?:(?:usd|us\$|\$|eur|\u20ac|gbp|\u00a3)\s*)?"
        r"(thousands|millions|billions|thousand|million|billion)\s*(?:usd)?\s*$",
        re.I,
    ),
    # Single letters count only in upper case: "(b)" is a footnote, "(B)" or "($M)" a unit.
    re.compile(
        r"^\s*(?:in\s+)?(?:(?:USD|US\$|\$|EUR|\u20ac|GBP|\u00a3)\s*)?(K|M|MM|MN|B|BN|000s?|'000s?)\s*$"
    ),
)
_SCALES = {
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
    "000": 1e3,
    "000s": 1e3,
    "'000": 1e3,
    "'000s": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "m": 1e6,
    "mm": 1e6,
    "mn": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "b": 1e9,
    "bn": 1e9,
}
_TABLE_SCALE = (
    re.compile(
        r"\bin\s+(?:(?:u\.?s\.?\s*)?(?:usd|us\$|\$|dollars|eur|euros?|\u20ac|gbp|\u00a3)\s+)?"
        r"(thousands|millions|billions)\b",
        re.I,
    ),
    re.compile(
        r"\b(thousands|millions|billions)\s+of\s+(?:u\.?s\.?\s+)?(?:dollars|usd|euros?)\b", re.I
    ),
    re.compile(
        r"(?:\busd|us\$|\$|\beur|\u20ac|\bgbp|\u00a3)\s*(thousands|millions|billions)\b", re.I
    ),
    re.compile(r"(?:\busd|\beur|\bgbp)\s+(k|m|mm|mn|b|bn)\b", re.I),
    re.compile(r"(?<![a-z0-9])\$\s*(k|m|mm|mn|b|bn)(?![a-z0-9])", re.I),
    re.compile(r"\(\s*(?:in\s+)?(K|M|MM|MN|B|BN|000s?|'000s?)\s*\)"),
)
_SHARE_SCALE = re.compile(
    r"\bshares?\b[^.;|]{0,80}?\b(?:in\s+)?(thousands|millions|billions)\b", re.I
)
_PERIOD_ROW = re.compile(
    r"(?:(?:fiscal )?(?:period|quarter|year) )?(?:end|ending|ended|end date)|date|report date|"
    r"period|fiscal date ending"
)
_SKIP_PERIOD = re.compile(
    r"\b(ttm|ltm|trailing|ytd|year to date|nine months|six months|9 months|6 months|26 weeks|"
    r"39 weeks|estimates?|est|forecast|projected|guidance|consensus|current)\b|\b\d{4}e\b|\(e\)"
)
_QUARTER_LABELS = (
    re.compile(r"\bq([1-4])\s*[- ]?\s*(?:fy|f)?\s*'?(\d{4}|\d{2})\b"),
    re.compile(r"\b(?:fy\s*)?(\d{4})\s*[- ]?\s*q([1-4])\b"),
    re.compile(r"\b([1-4])q\s*(?:fy)?\s*'?(\d{4}|\d{2})\b"),
)
_FY_LABEL = re.compile(r"\b(?:fy|fiscal\s+year|fiscal)\s*'?(\d{4}|\d{2})\b")
_QUARTER_WORDS = re.compile(
    r"\b(three months|3 months|13 weeks|14 weeks|quarter|quarterly|quarters)\b"
)
_YEAR_WORDS = re.compile(
    r"\b(twelve months|12 months|52 weeks|53 weeks|year ended|year ending|fiscal year|annual|"
    r"annually|yearly)\b"
)
_PERIOD_GAPS = {QUARTER: (80, 100), YEAR: (350, 380)}


@dataclass(frozen=True, slots=True)
class Figure:
    """One reported value from one page: a metric for one period, in base units."""

    metric: str
    value: float
    unit: str  # "<CCY>" for money, "<CCY>/shares" per share, "shares"
    currency: str | None
    end: date
    kind: str  # fiscal_quarter | fiscal_year | instant
    period_label: str | None  # the page's own label ("Q3 2026", "FY2025"), when it gave one
    label: str  # the page's row or column label ("Total net sales")
    raw_value: str
    origin: str


@dataclass
class FigureSet:
    """The figures read from one page and what was dropped on the way."""

    figures: list[Figure] = field(default_factory=list)
    after_as_of: int = 0
    implausible: dict[str, int] = field(default_factory=dict)
    currency_stated: bool = True

    def notes(self, as_of: date) -> list[str]:
        notes: list[str] = []
        if self.after_as_of:
            notes.append(
                f"{self.after_as_of} value(s) for periods ending after {as_of.isoformat()} "
                "dropped (no look-ahead)"
            )
        if self.implausible:
            detail = "; ".join(f"{reason}: {n}" for reason, n in sorted(self.implausible.items()))
            notes.append(
                f"{sum(self.implausible.values())} implausible value(s) dropped ({detail})"
            )
        if self.figures and not self.currency_stated:
            notes.append("the table states no currency; USD assumed")
        return notes


@dataclass(frozen=True, slots=True)
class PeriodCell:
    """What a header (or first-column) cell says about its period."""

    end: date | None
    kind: str | None
    label: str | None
    skip: bool


@dataclass(frozen=True, slots=True)
class FigureLine:
    """A row (periods as columns) or a column (periods as rows) holding numbers."""

    key: str
    label: str
    index: int
    scale: float | None


@dataclass(frozen=True, slots=True)
class FigureCandidate:
    """A table that could report figures: dated periods and numeric lines. Which table holds
    the quarterly results, and which line is which metric, is Laya's choice."""

    index: int
    table: RawTable
    across: bool  # True: periods are columns and lines are rows
    periods: dict[int, PeriodCell]
    kinds: dict[int, str | None]
    lines: tuple[FigureLine, ...]
    currency: str
    currency_stated: bool
    scales: tuple[float, float | None]
    undated: int

    def describe(self) -> str:
        dated = sorted(
            (cell for cell in self.periods.values() if cell.end is not None),
            key=lambda c: c.end,  # type: ignore[arg-type,return-value]
            reverse=True,
        )
        labels = ", ".join(c.label or c.end.isoformat() for c in dated[:2])  # type: ignore[union-attr]
        return f"{len(dated)} periods ({labels}), {len(self.lines)} rows"

    def line(self, key: str) -> FigureLine | None:
        return next((line for line in self.lines if line.key == key), None)


def _period_cell(text: str, order: str | None, *, epoch: bool = False) -> PeriodCell:
    low = " ".join((text or "").lower().split())
    skip = bool(_SKIP_PERIOD.search(low))
    kind: str | None = None
    label: str | None = None
    for index, pattern in enumerate(_QUARTER_LABELS):
        match = pattern.search(low)
        if match:
            quarter, year = (match.group(2), match.group(1)) if index == 1 else match.groups()
            kind, label = QUARTER, f"Q{quarter} {_year(year)}"
            break
    if kind is None:
        match = _FY_LABEL.search(low)
        if match:
            kind, label = YEAR, f"FY{_year(match.group(1))}"
    if kind is None and _QUARTER_WORDS.search(low) and not _YEAR_WORDS.search(low):
        kind = QUARTER
    elif kind is None and _YEAR_WORDS.search(low) and not _QUARTER_WORDS.search(low):
        kind = YEAR
    return PeriodCell(parse_table_date(text, order, epoch=epoch), kind, label, skip)


def _context_kind(text: str) -> str | None:
    low = text.lower()
    quarter, year = bool(_QUARTER_WORDS.search(low)), bool(_YEAR_WORDS.search(low))
    if quarter and not year:
        return QUARTER
    if year and not quarter:
        return YEAR
    return None


def _spacing_kind(ends: list[date]) -> str | None:
    """Quarter or year when every gap between consecutive distinct ends says so."""
    ordered = sorted(set(ends))
    if len(ordered) < 2:
        return None
    gaps = [(b - a).days for a, b in pairwise(ordered)]
    for kind, (low, high) in _PERIOD_GAPS.items():
        if all(low <= gap <= high for gap in gaps):
            return kind
    return None


def _resolve_kinds(cells: dict[int, PeriodCell], context: str) -> dict[int, str | None]:
    """Each dated period's kind: its own label or words, else the table's context, else the
    spacing of the undetermined periods' end dates; ``None`` when nothing decides it."""
    kinds = {key: cell.kind for key, cell in cells.items()}
    open_keys = [key for key, kind in kinds.items() if kind is None]
    if open_keys:
        ends = [cells[key].end for key in open_keys if cells[key].end is not None]
        fallback = _context_kind(context) or _spacing_kind(ends)  # type: ignore[arg-type]
        for key in open_keys:
            kinds[key] = fallback
    return kinds


def _scale_word(text: str) -> float | None:
    return _SCALES.get(text.lower())


def _line_label(text: str) -> tuple[str, float | None]:
    """(label without unit groups, the unit scale a group named) for a row or column label."""
    scale: float | None = None

    def unit_group(match: re.Match[str]) -> str:
        nonlocal scale
        inner = match.group(1)
        for pattern in _ROW_SCALE:
            found = pattern.match(inner)
            if found:
                scale = _scale_word(found.group(1))
                return " "
        return f" ({inner}) "

    cleaned = _WS.sub(" ", _UNIT_GROUP.sub(unit_group, text or "")).strip()
    return cleaned, scale


def _label_words(label: str) -> list[str]:
    return [w for w in norm_header(_split_camel(label)).split() if not re.fullmatch(r"\d{1,2}", w)]


def shortlist(candidate: FigureCandidate, metric: str, limit: int) -> list[FigureLine]:
    """The candidate's lines worth offering for ``metric``, best first: exact labels, then
    labels that carry every keyword group the metric needs; never an adjusted, growth, ratio,
    cost or estimate line (and no per-share line for an amount)."""
    exact = set(_METRIC_LABELS[metric])
    groups = _METRIC_KEYWORDS[metric]
    scored: list[tuple[int, int, FigureLine]] = []
    for position, line in enumerate(candidate.lines):
        words = _label_words(line.label)
        name = " ".join(words)
        present = set(words)
        if name in exact:
            score = 8 if name in _PLAIN_EPS else 10 + (1 if words[:1] == ["total"] else 0)
        elif all(group & present for group in groups) and not present & _EXCLUDED_WORDS:
            score = 5
        else:
            continue
        amount = metric not in PER_SHARE_METRICS and metric not in SHARE_METRICS
        if amount and {"per", "share"} <= present:
            continue
        if metric in PER_SHARE_METRICS and "shares" in present:
            continue  # a share count ("Shares used in computing earnings per share")
        scored.append((-score, position, line))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [line for _score, _position, line in scored[:limit]]


def _table_scale(table: RawTable) -> tuple[float, float | None]:
    """(money scale, share scale or ``None``) from the caption, nearby text and headers."""
    text = " ".join([table.context, *table.columns[:3]])
    best: tuple[int, float] | None = None
    for pattern in _TABLE_SCALE:
        match = pattern.search(text)
        if match:
            value = _scale_word(match.group(1))
            if value is not None and (best is None or match.start() < best[0]):
                best = (match.start(), value)
    shares = _SHARE_SCALE.search(text)
    return (best[1] if best else 1.0), (_scale_word(shares.group(1)) if shares else None)


def _numeric(text: str) -> bool:
    parsed = parse_cell_number(text)
    return parsed is not None and parsed.unit not in ("percent", "ratio")


def _across_candidate(
    table: RawTable, order: str | None
) -> tuple[dict[int, PeriodCell], list[FigureLine], int] | None:
    """Periods as columns: header cells (or a "Period ending" row) date the columns and each
    row label is a line."""
    width = len(table.columns)
    cells = {j: _period_cell(table.columns[j], order) for j in range(1, width)}
    period_row: int | None = None
    for i, row in enumerate(table.rows[:6]):
        if _PERIOD_ROW.fullmatch(norm_header(row[0])):
            dates = {j: parse_table_date(row[j], order) for j in range(1, width)}
            if sum(1 for d in dates.values() if d) >= max(1, (width - 1) // 2):
                cells = {
                    j: PeriodCell(cell.end or dates[j], cell.kind, cell.label, cell.skip)
                    for j, cell in cells.items()
                }
                period_row = i
                break
    undated = sum(1 for c in cells.values() if c.end is None and c.label and not c.skip)
    dated = {j: c for j, c in cells.items() if c.end is not None and not c.skip}
    if not dated:
        return None
    lines: list[FigureLine] = []
    section = ""
    for i, row in enumerate(table.rows):
        if i == period_row or not row[0]:
            continue
        label, scale = _line_label(row[0])
        if not any(_numeric(row[j]) for j in dated):
            section = label
            continue
        if norm_header(label) in _SECTION_QUALIFIED and section:
            label = f"{section.rstrip(':')}: {label}"
        lines.append(FigureLine(f"l{len(lines) + 1}", label[:MAX_CELL_CHARS], i, scale))
    return dated, lines, undated


def _down_candidate(
    table: RawTable, order: str | None
) -> tuple[dict[int, PeriodCell], list[FigureLine], int] | None:
    """Periods as rows: one column dates the rows (a label or date column) and every other
    column with numbers is a line."""
    norm = [norm_header(c) for c in table.columns]
    best: tuple[int, int] | None = None
    for j in range(len(table.columns)):
        epoch = _is_date_header(norm[j])
        count = sum(
            1
            for row in table.rows[:200]
            if (cell := _period_cell(row[j], order, epoch=epoch)).end or cell.label
        )
        if count >= 2 and (best is None or count > best[1]):
            best = (j, count)
    if best is None:
        return None
    period_col = best[0]
    epoch = _is_date_header(norm[period_col])
    date_cols = [
        j for j in range(len(table.columns)) if j != period_col and _is_date_header(norm[j])
    ]
    cells: dict[int, PeriodCell] = {}
    undated = 0
    for i, row in enumerate(table.rows):
        cell = _period_cell(row[period_col], order, epoch=epoch)
        if cell.end is None:
            extra = next((d for j in date_cols if (d := parse_table_date(row[j], order))), None)
            cell = PeriodCell(extra, cell.kind, cell.label, cell.skip)
        if cell.end is None:
            undated += 1 if cell.label and not cell.skip else 0
            continue
        if not cell.skip:
            cells[i] = cell
    if not cells:
        return None
    lines: list[FigureLine] = []
    for j, header in enumerate(table.columns):
        if j == period_col or j in date_cols or not header:
            continue
        if any(_numeric(table.rows[i][j]) for i in cells):
            label, scale = _line_label(header)
            lines.append(FigureLine(f"l{len(lines) + 1}", label[:MAX_CELL_CHARS], j, scale))
    return cells, lines, undated


def figure_candidates(tables: list[RawTable]) -> list[FigureCandidate]:
    """Tables that could report figures (at least one period with an end date written on the
    page and a kind, quarter or fiscal year, and at least one line of numbers), in page order.

    A period's kind comes from its own label ("Q3 2026", "FY2025", "Three Months Ended"), else
    the table's caption and nearby text ("Quarterly"), else the spacing of the period ends
    (80-100 days apart: quarters). A label without a date ("Q3 2026" alone) does not fix the
    period, so it is counted as undated and not read. Trailing, year-to-date and estimate
    periods are skipped.
    """
    found: list[FigureCandidate] = []
    for index, table in enumerate(tables):
        order = date_order(
            [
                *table.columns,
                *(row[0] for row in table.rows),
                *(c for r in table.rows[:6] for c in r),
            ]
        )
        best: tuple[bool, dict[int, PeriodCell], list[FigureLine], int] | None = None
        for across, built in (
            (True, _across_candidate(table, order)),
            (False, _down_candidate(table, order)),
        ):
            if built is None or not built[1]:
                continue
            size = len(built[0]) * len(built[1])
            if best is None or size > len(best[1]) * len(best[2]):
                best = (across, built[0], built[1], built[2])
        if best is None:
            continue
        across, periods, lines, undated = best
        context = " ".join([table.context, table.columns[0] if across else " ".join(table.columns)])
        kinds = _resolve_kinds(periods, context)
        if not any(kinds.values()):
            continue
        if across:
            values = [table.rows[line.index][j] for line in lines for j in periods]
        else:
            values = [table.rows[i][line.index] for i in periods for line in lines]
        currency, stated = _table_currency(values[:400], [*table.columns, table.context])
        if currency is None:
            continue
        found.append(
            FigureCandidate(
                index=index,
                table=table,
                across=across,
                periods=periods,
                kinds=kinds,
                lines=tuple(lines),
                currency=currency,
                currency_stated=stated,
                scales=_table_scale(table),
                undated=undated,
            )
        )
    return found


def read_figures(
    candidate: FigureCandidate, chosen: dict[str, FigureLine], as_of: date
) -> FigureSet:
    """Read the chosen line of each metric for every dated period of the candidate.

    Money is scaled by the unit written in the line's label, else the caption or the text
    before the table ("(in millions)", "$M", "USD thousands"), unless the cell carries its own
    suffix ("94.9B"); per-share values are never scaled; share counts use a share-specific
    unit when the page states one. Periods ending after ``as_of`` are dropped, then values no
    statement can report (see ``_plausible``).
    """
    result = FigureSet(currency_stated=candidate.currency_stated)
    table = candidate.table
    money_scale, share_scale = candidate.scales
    collected: list[Figure] = []
    for metric, line in chosen.items():
        for key, period in candidate.periods.items():
            cell = table.rows[line.index][key] if candidate.across else table.rows[key][line.index]
            parsed = parse_cell_number(cell)
            if parsed is None or parsed.unit in ("percent", "ratio", "bp", "days"):
                continue
            kind = INSTANT if metric in SHARE_METRICS else candidate.kinds.get(key)
            if kind is None or period.end is None:
                continue
            if period.end > as_of:
                result.after_as_of += 1
                continue
            value = parsed.value
            if metric in PER_SHARE_METRICS:
                unit = f"{parsed.currency or candidate.currency}/shares"
            elif metric in SHARE_METRICS:
                if not parsed.scale_applied:
                    value *= line.scale or share_scale or money_scale
                unit = "shares"
            else:
                if not parsed.scale_applied:
                    value *= line.scale or money_scale
                unit = parsed.currency or candidate.currency
            collected.append(
                Figure(
                    metric=metric,
                    value=value,
                    unit=unit,
                    currency=None if unit == "shares" else unit.split("/")[0],
                    end=period.end,
                    kind=kind,
                    period_label=period.label,
                    label=line.label,
                    raw_value=cell,
                    origin=table.origin,
                )
            )
    result.figures, result.implausible = _plausible(collected)
    result.figures.sort(key=lambda f: (f.metric, f.kind, f.end))
    return result


def _plausible(figures: list[Figure]) -> tuple[list[Figure], dict[str, int]]:
    """Drop values no statement can report: non-finite, out of range, or margins outside
    0-100 % against the same table's revenue for the same period. A "gross margin" line must
    be an amount of at least 1 % of revenue (a percentage written without its sign is not)."""
    kept: list[Figure] = []
    dropped: dict[str, int] = {}
    revenue = {(f.kind, f.end): f.value for f in figures if f.metric == "revenue"}
    for figure in figures:
        value = figure.value
        base = revenue.get((figure.kind, figure.end))
        reason: str | None = None
        if not math.isfinite(value) or figure.end.year < MIN_FIGURE_YEAR:
            reason = "unusable value or date"
        elif figure.metric in PER_SHARE_METRICS:
            reason = "per-share value out of range" if abs(value) > MAX_ABS_EPS else None
        elif figure.metric in SHARE_METRICS:
            reason = "share count out of range" if not 0 < value <= MAX_SHARES else None
        elif abs(value) > MAX_MONEY:
            reason = "amount out of range"
        elif figure.metric == "revenue" and value < 0:
            reason = "negative revenue"
        elif figure.metric == "gross_profit":
            margin_line = "margin" in _label_words(figure.label)
            if base is not None and base > 0 and not 0 <= value / base <= 1:
                reason = "gross margin outside 0-100%"
            elif margin_line and (base is None or base <= 0 or value / base < 0.01):
                reason = "gross margin line is not an amount"
        elif (
            figure.metric in ("operating_income", "net_income", "free_cash_flow")
            and base is not None
            and base > 0
            and value / base > 1
        ):
            reason = "margin above 100%"
        if reason is None:
            kept.append(figure)
        else:
            dropped[reason] = dropped.get(reason, 0) + 1
    return kept, dropped
