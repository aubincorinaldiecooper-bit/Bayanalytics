"""Turn fetched pages into ``EvidenceRecord`` objects: metadata + short text, never the page.

Retrieved content is evidence, never instruction (AGENT.md section 30): this module only
parses; nothing in a page can influence control flow beyond which text gets kept. Storage
limits (section 26): main text is capped at ``MAX_TEXT_CHARS`` and the stored excerpt at
``EXCERPT_CHARS``; the content hash fingerprints the whitespace-normalised text so duplicates
can be detected without keeping full articles.

Only the standard library is used: ``html.parser`` builds a tiny DOM, and a small readability
heuristic scores block containers by paragraph text length and link density. The page's tables
(``<table>`` elements and JSON script blocks, see ``research.tables``) are kept, bounded, in
``EvidenceRecord.structured["tables"]`` so price history and reported figures can be read from
the page the search returned; links inside a page are never followed.

Hostile input: element nesting is capped at ``MAX_DOM_DEPTH`` in the tree builder (a page
nested deeper is rejected as ``extract_failed``; real pages stay far below the cap) and every
tree walk is iterative, so no page can exhaust the interpreter stack. ``extract_page`` turns
any other parser failure into ``ResearchProviderError("extract_failed")`` so one bad page is a
rejected source, never a failed analysis.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

from bayanalytics.research.dates import parse_datetime_lenient
from bayanalytics.research.fetch import PAYWALL_HEADER, REASON_EXTRACT_FAILED, TaggedProviderError
from bayanalytics.research.provider import EvidenceRecord, PageResult, ResearchProviderError
from bayanalytics.research.tables import html_tables

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 12_000
EXCERPT_CHARS = 600
MAX_CSV_ROWS = 20_000
MAX_DOM_DEPTH = 200
HTML_METHOD = "html_readability_v1"


class HtmlTooDeepError(ValueError):
    """Raised by the tree builder when element nesting exceeds ``MAX_DOM_DEPTH``."""


_DROP_TAGS = frozenset(
    {
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "noscript",
        "template",
        "svg",
        "iframe",
        "button",
        "select",
        "option",
        "textarea",
        "canvas",
        "video",
        "audio",
    }
)
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "article",
        "main",
        "section",
        "li",
        "ul",
        "ol",
        "blockquote",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "tr",
        "td",
        "th",
        "pre",
        "figure",
        "figcaption",
        "dl",
        "dd",
        "dt",
        "body",
        "html",
    }
)
_CANDIDATE_TAGS = frozenset({"article", "main", "section", "div", "body", "td"})
_PARAGRAPH_TAGS = frozenset({"p", "li", "blockquote", "h2", "h3", "h4", "pre", "dd"})
_DATE_META_KEYS = (
    "article:published_time",
    "og:published_time",
    "datepublished",
    "date",
    "pubdate",
    "publishdate",
    "publish_date",
    "publication_date",
    "dc.date",
    "dc.date.issued",
    "dcterms.created",
    "parsely-pub-date",
    "sailthru.date",
    "article.published",
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WS_RE = re.compile(r"\s+")
_MULTI_SUFFIXES = frozenset(
    {"co.uk", "com.au", "co.jp", "co.nz", "com.br", "co.in", "com.sg", "com.hk", "co.za", "org.uk"}
)


# --------------------------------------------------------------------------------------
# tiny DOM
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Node:
    tag: str
    attrs: dict[str, str]
    parent: _Node | None
    children: list[_Node | str] = field(default_factory=list)
    dropped: bool = False


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document", {}, None)
        self._stack: list[_Node] = [self.root]
        self.title_parts: list[str] = []
        self.meta: list[tuple[str, str]] = []  # (key, content) for name/property/itemprop
        self.jsonld: list[str] = []
        self.time_datetimes: list[tuple[str, dict[str, str]]] = []
        self.html_lang: str | None = None
        self._in_title = False
        self._in_jsonld = False

    @property
    def current(self) -> _Node:
        return self._stack[-1]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {k.lower(): (v or "") for k, v in attrs}
        if tag == "html":
            self.html_lang = attributes.get("lang") or self.html_lang
        if tag == "meta":
            key = attributes.get("property") or attributes.get("name") or attributes.get("itemprop")
            content = attributes.get("content")
            if key and content is not None:
                self.meta.append((key.strip().lower(), content.strip()))
            return
        if tag in _VOID_TAGS:
            if tag == "br":
                self.current.children.append("\n")
            return
        if tag == "time":
            self.time_datetimes.append((attributes.get("datetime", ""), attributes))
        if tag == "title":
            self._in_title = True
        if tag == "script" and "ld+json" in attributes.get("type", "").lower():
            self._in_jsonld = True
            self.jsonld.append("")
        if tag == "p" and self.current.tag == "p":
            self._pop_to("p")
        elif tag in _BLOCK_TAGS and self.current.tag == "p":
            self._pop_to("p")
        elif tag == "li" and self.current.tag == "li":
            self._pop_to("li")
        if len(self._stack) - 1 >= MAX_DOM_DEPTH:
            raise HtmlTooDeepError(f"element nesting deeper than {MAX_DOM_DEPTH}")
        node = _Node(tag, attributes, self.current)
        node.dropped = tag in _DROP_TAGS or self.current.dropped
        self.current.children.append(node)
        self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS and tag != "meta":
            self._pop_to(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS or tag == "meta":
            return
        if tag == "title":
            self._in_title = False
        if tag == "script":
            self._in_jsonld = False
        self._pop_to(tag)

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._in_jsonld and self.jsonld:
            self.jsonld[-1] += data
        if data:
            self.current.children.append(data)

    def _pop_to(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return


def _child_nodes(node: _Node) -> list[_Node]:
    """Element children in reverse document order, ready to push on a DFS stack."""
    return [child for child in reversed(node.children) if not isinstance(child, str)]


def _text_of(node: _Node, *, links: bool = False) -> tuple[int, int]:
    """Return (text_len, link_text_len) for the subtree, ignoring dropped nodes."""
    text_len = 0
    link_len = 0
    stack: list[tuple[_Node, bool]] = [(node, links)]
    while stack:
        current, in_link = stack.pop()
        if current.dropped:
            continue
        in_link = in_link or current.tag == "a"
        for child in current.children:
            if isinstance(child, str):
                n = len(_WS_RE.sub(" ", child).strip())
                text_len += n
                if in_link:
                    link_len += n
            else:
                stack.append((child, in_link))
    return text_len, link_len


def _collect_text(node: _Node, out: list[str]) -> None:
    """Append the subtree's text to ``out`` in document order, block tags as line breaks."""
    if node.dropped:
        return
    stack: list[_Node | str] = list(reversed(node.children))
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            out.append(item)
            continue
        block = item.tag in _BLOCK_TAGS
        if block:
            out.append("\n")
        if item.dropped:
            if block:
                out.append("\n")
            continue
        if block:
            stack.append("\n")  # emitted after the children
        stack.extend(reversed(item.children))


def _node_text(node: _Node) -> str:
    parts: list[str] = []
    _collect_text(node, parts)
    return _normalise_block("".join(parts))


def _normalise_block(text: str) -> str:
    text = _CONTROL_RE.sub("", text)
    lines = [_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _paragraphs(node: _Node, out: list[str]) -> None:
    """Outermost paragraph-like blocks under ``node`` in document order."""
    if node.dropped:
        return
    stack = _child_nodes(node)
    while stack:
        child = stack.pop()
        if child.dropped:
            continue
        if child.tag in _PARAGRAPH_TAGS:
            text = _WS_RE.sub(" ", _node_text(child)).strip()
            if text:
                out.append(text)
        else:
            stack.extend(_child_nodes(child))


def _para_score(node: _Node) -> float:
    if node.dropped:
        return 0.0
    score = 0.0
    stack = _child_nodes(node)
    while stack:
        child = stack.pop()
        if child.dropped:
            continue
        if child.tag in _PARAGRAPH_TAGS:
            text_len, link_len = _text_of(child)
            if text_len >= 20:
                density = link_len / text_len if text_len else 1.0
                score += text_len * (1.0 - density)
        else:
            stack.extend(_child_nodes(child))
    return score


def _candidates(node: _Node, out: list[_Node]) -> None:
    stack = [node]
    while stack:
        current = stack.pop()
        if current.dropped:
            continue
        if current.tag in _CANDIDATE_TAGS:
            out.append(current)
        stack.extend(_child_nodes(current))


def _best_container(root: _Node) -> _Node | None:
    candidates: list[_Node] = []
    _candidates(root, candidates)
    if not candidates:
        return None
    best: _Node | None = None
    best_score = 0.0
    scores: dict[int, float] = {}
    for node in candidates:
        para = _para_score(node)
        text_len, link_len = _text_of(node)
        density = (link_len / text_len) if text_len else 1.0
        score = para * (1.0 - 0.7 * density)
        scores[id(node)] = para
        if score > best_score:
            best, best_score = node, score
    if best is None:
        return None
    # Tighten: descend while one child container still carries most of the paragraph text.
    while True:
        children = [c for c in best.children if not isinstance(c, str) and c.tag in _CANDIDATE_TAGS]
        moved = False
        for child in children:
            child_para = scores.get(id(child))
            if child_para is None:
                child_para = _para_score(child)
            if scores[id(best)] > 0 and child_para >= 0.8 * scores[id(best)]:
                scores.setdefault(id(child), child_para)
                best = child
                moved = True
                break
        if not moved:
            return best


# --------------------------------------------------------------------------------------
# public helpers
# --------------------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Strip C0/C1 control characters and collapse whitespace inside lines."""
    return _normalise_block(text)


def content_hash(text: str) -> str:
    normalised = _WS_RE.sub(" ", text).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def excerpt_of(text: str, limit: int = EXCERPT_CHARS) -> str:
    flat = _WS_RE.sub(" ", text).strip()
    return flat[:limit].rstrip()


def registrable_domain(host: str) -> str:
    host = host.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in _MULTI_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _find_jsonld_dates(blob: Any) -> str | None:
    stack: list[Any] = [blob]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key in ("datePublished", "dateCreated", "uploadDate"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, list):
            stack.extend(reversed(item))
    return None


def _published_at(tree: _TreeBuilder) -> datetime | None:
    meta = dict(reversed(tree.meta))  # first occurrence wins
    for key in _DATE_META_KEYS:
        value = meta.get(key)
        if value:
            parsed = parse_datetime_lenient(value)
            if parsed:
                return parsed
    for blob in tree.jsonld:
        try:
            data = json.loads(blob)
        except (ValueError, RecursionError):  # the C decoder recurses on nested JSON
            continue
        candidate = _find_jsonld_dates(data)
        if candidate:
            parsed = parse_datetime_lenient(candidate)
            if parsed:
                return parsed
    for value, attrs in tree.time_datetimes:
        marker = (
            attrs.get("itemprop") or attrs.get("property") or attrs.get("class") or ""
        ).lower()
        if value and ("publish" in marker or "pubdate" in attrs or not marker):
            parsed = parse_datetime_lenient(value)
            if parsed:
                return parsed
    for value, _ in tree.time_datetimes:
        parsed = parse_datetime_lenient(value)
        if parsed:
            return parsed
    return None


def extract_html(url: str, html: str, fetched_at: datetime) -> EvidenceRecord:
    tree = _TreeBuilder()
    tree.feed(_CONTROL_RE.sub("", html))
    tree.close()
    meta = dict(reversed(tree.meta))
    title = (
        meta.get("og:title")
        or meta.get("twitter:title")
        or _WS_RE.sub(" ", "".join(tree.title_parts)).strip()
        or _first_heading(tree.root)
        or url
    )
    host = urlsplit(url).netloc
    publisher = meta.get("og:site_name") or meta.get("publisher") or registrable_domain(host)
    container = _best_container(tree.root)
    paragraphs: list[str] = []
    if container is not None:
        _paragraphs(container, paragraphs)
    text = "\n\n".join(paragraphs)
    if len(text) < 40:
        fallback = _node_text(container) if container is not None else ""
        if len(fallback) < 40:
            fallback = _node_text(tree.root)
        text = fallback
    text = clean_text(text)[:MAX_TEXT_CHARS]
    return EvidenceRecord(
        url=url,
        final_url=url,
        title=clean_text(title)[:300] or url,
        publisher=clean_text(publisher)[:120] or None,
        published_at=_published_at(tree),
        retrieved_at=fetched_at,
        text=text,
        excerpt=excerpt_of(text),
        content_hash=content_hash(text),
        extraction_method=HTML_METHOD,
        language=(tree.html_lang or None),
        structured={"tables": _tables_of(url, html)},
        metadata={"paragraphs": len(paragraphs), "container": container.tag if container else None},
    )


def _tables_of(url: str, html: str) -> list[dict[str, Any]]:
    """The page's tables (bounded); a table the parser cannot read costs the tables only,
    never the page's text."""
    try:
        return [table.to_dict() for table in html_tables(html)]
    except Exception as exc:
        log.warning("table extraction failed for %s (%s)", url, type(exc).__name__)
        return []


def _first_heading(root: _Node) -> str:
    if root.dropped:
        return ""
    stack = _child_nodes(root)
    while stack:
        child = stack.pop()
        if child.dropped:
            continue
        if child.tag == "h1":
            text = _WS_RE.sub(" ", _node_text(child)).strip()
            if text:
                return text
        stack.extend(_child_nodes(child))
    return ""


def extract_json(url: str, body: str, fetched_at: datetime) -> EvidenceRecord:
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    structured: dict[str, Any]
    if isinstance(data, dict):
        structured = data
    elif isinstance(data, list):
        structured = {"items": data}
    else:
        structured = {"parse_error": True}
    return EvidenceRecord(
        url=url,
        final_url=url,
        title=_title_from_url(url),
        publisher=registrable_domain(urlsplit(url).netloc) or None,
        retrieved_at=fetched_at,
        text="",
        excerpt="",
        content_hash=content_hash(body),
        extraction_method="json",
        structured=structured,
        metadata={"bytes": len(body)},
    )


def extract_csv(url: str, body: str, fetched_at: datetime) -> EvidenceRecord:
    reader = csv.reader(io.StringIO(body))
    columns: list[str] = []
    rows: list[list[str]] = []
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        if not columns:
            columns = [cell.strip() for cell in row]
            continue
        rows.append([cell.strip() for cell in row])
        if len(rows) >= MAX_CSV_ROWS:
            break
    return EvidenceRecord(
        url=url,
        final_url=url,
        title=_title_from_url(url),
        publisher=registrable_domain(urlsplit(url).netloc) or None,
        retrieved_at=fetched_at,
        text="",
        excerpt="",
        content_hash=content_hash(body),
        extraction_method="csv",
        structured={"columns": columns, "rows": rows, "row_count": len(rows)},
        metadata={"bytes": len(body)},
    )


def extract_text(url: str, body: str, fetched_at: datetime) -> EvidenceRecord:
    text = clean_text(body)[:MAX_TEXT_CHARS]
    return EvidenceRecord(
        url=url,
        final_url=url,
        title=_title_from_url(url),
        publisher=registrable_domain(urlsplit(url).netloc) or None,
        retrieved_at=fetched_at,
        text=text,
        excerpt=excerpt_of(text),
        content_hash=content_hash(text),
        extraction_method="text",
    )


def _title_from_url(url: str) -> str:
    parts = urlsplit(url)
    tail = parts.path.rstrip("/").rsplit("/", 1)[-1]
    return tail or parts.netloc or url


def page_kind(page: PageResult) -> str:
    ctype = (page.content_type or "").lower()
    path = urlsplit(page.final_url or page.url).path.lower()
    if "json" in ctype or path.endswith(".json"):
        return "json"
    if "csv" in ctype or path.endswith(".csv") or "/q/d/l/" in path:
        return "csv"
    if "html" in ctype or "xml" in ctype or path.endswith((".htm", ".html")):
        return "html"
    if ctype.startswith("text/plain"):
        return "csv" if _looks_like_csv(page.body) else "text"
    body_head = page.body.lstrip()[:64].lower()
    if body_head.startswith(("<!doctype", "<html", "<head", "<body")):
        return "html"
    if body_head.startswith(("{", "[")):
        return "json"
    return "html"


def _looks_like_csv(body: str) -> bool:
    """A plain-text body whose first two lines are comma-separated with the same field count
    and a header of short, non-numeric names (a CSV served as ``text/plain``)."""
    lines = [line for line in body.lstrip().splitlines()[:2] if line.strip()]
    if len(lines) < 2:
        return False
    header, first = lines[0].split(","), lines[1].split(",")
    if len(header) < 2 or len(header) != len(first):
        return False
    return all(0 < len(cell.strip()) <= 30 and not cell.strip()[:1].isdigit() for cell in header)


def extract_page(page: PageResult) -> EvidenceRecord:
    """Extract ``page``; any parser failure becomes ``ResearchProviderError("extract_failed")``.

    Retrieved content is untrusted, so a page that breaks the extractor (hostile nesting,
    malformed markup, a parser bug) is reported with a fixed keyword and the detail is logged.
    """
    try:
        return _extract_page(page)
    except ResearchProviderError:
        raise
    except Exception as exc:
        log.warning(
            "extraction failed for %s (%s: %s)", page.final_url or page.url, type(exc).__name__, exc
        )
        raise TaggedProviderError(REASON_EXTRACT_FAILED) from exc


def _extract_page(page: PageResult) -> EvidenceRecord:
    url = page.final_url or page.url
    if page.headers.get(PAYWALL_HEADER) == "true":
        record = EvidenceRecord(
            url=page.url,
            final_url=url,
            title=_title_from_url(url),
            publisher=registrable_domain(urlsplit(url).netloc) or None,
            retrieved_at=page.fetched_at,
            content_hash=content_hash(""),
            extraction_method="paywalled",
            metadata={"paywalled": True, "status": page.status},
        )
        return record
    kind = page_kind(page)
    if kind == "json":
        record = extract_json(url, page.body, page.fetched_at)
    elif kind == "csv":
        record = extract_csv(url, page.body, page.fetched_at)
    elif kind == "text":
        record = extract_text(url, page.body, page.fetched_at)
    else:
        record = extract_html(url, page.body, page.fetched_at)
    record.url = page.url
    record.final_url = url
    record.metadata.setdefault("status", page.status)
    record.metadata.setdefault("from_cache", page.from_cache)
    if page.headers.get("x-bay-truncated") == "true":
        record.metadata["truncated"] = True
    return record
