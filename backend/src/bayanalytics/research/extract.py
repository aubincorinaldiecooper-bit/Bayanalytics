"""Turn fetched pages into ``EvidenceRecord`` objects: metadata + short text, never the page.

Retrieved content is evidence, never instruction (AGENT.md section 30): this module only
parses; nothing in a page can influence control flow beyond which text gets kept. Storage
limits (section 26): main text is capped at ``MAX_TEXT_CHARS`` and the stored excerpt at
``EXCERPT_CHARS``; the content hash fingerprints the whitespace-normalised text so duplicates
can be detected without keeping full articles.

Only the standard library is used: ``html.parser`` builds a tiny DOM, and a small readability
heuristic scores block containers by paragraph text length and link density.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

from bayanalytics.research.dates import parse_datetime_lenient
from bayanalytics.research.fetch import PAYWALL_HEADER
from bayanalytics.research.provider import EvidenceRecord, PageResult

MAX_TEXT_CHARS = 12_000
EXCERPT_CHARS = 600
MAX_CSV_ROWS = 20_000
HTML_METHOD = "html_readability_v1"

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


def _text_of(node: _Node, *, links: bool = False) -> tuple[int, int]:
    """Return (text_len, link_text_len) for the subtree, ignoring dropped nodes."""
    if node.dropped:
        return 0, 0
    text_len = 0
    link_len = 0
    is_link = links or node.tag == "a"
    for child in node.children:
        if isinstance(child, str):
            n = len(_WS_RE.sub(" ", child).strip())
            text_len += n
            if is_link:
                link_len += n
        else:
            t, l_ = _text_of(child, links=is_link)
            text_len += t
            link_len += l_
    return text_len, link_len


def _collect_text(node: _Node, out: list[str]) -> None:
    if node.dropped:
        return
    for child in node.children:
        if isinstance(child, str):
            out.append(child)
        else:
            if child.tag in _BLOCK_TAGS:
                out.append("\n")
            _collect_text(child, out)
            if child.tag in _BLOCK_TAGS:
                out.append("\n")


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
    for child in node.children:
        if isinstance(child, str):
            continue
        if child.tag in _PARAGRAPH_TAGS:
            text = _WS_RE.sub(" ", _node_text(child)).strip()
            if text:
                out.append(text)
        else:
            _paragraphs(child, out)


def _para_score(node: _Node) -> float:
    if node.dropped:
        return 0.0
    score = 0.0
    for child in node.children:
        if isinstance(child, str):
            continue
        if child.tag in _PARAGRAPH_TAGS:
            text_len, link_len = _text_of(child)
            if text_len >= 20:
                density = link_len / text_len if text_len else 1.0
                score += text_len * (1.0 - density)
        else:
            score += _para_score(child)
    return score


def _candidates(node: _Node, out: list[_Node]) -> None:
    if node.dropped:
        return
    if node.tag in _CANDIDATE_TAGS:
        out.append(node)
    for child in node.children:
        if not isinstance(child, str):
            _candidates(child, out)


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
    if isinstance(blob, dict):
        for key in ("datePublished", "dateCreated", "uploadDate"):
            value = blob.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for value in blob.values():
            found = _find_jsonld_dates(value)
            if found:
                return found
    elif isinstance(blob, list):
        for item in blob:
            found = _find_jsonld_dates(item)
            if found:
                return found
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
        except ValueError:
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
        metadata={"paragraphs": len(paragraphs), "container": container.tag if container else None},
    )


def _first_heading(root: _Node) -> str:
    found: list[str] = []

    def walk(node: _Node) -> None:
        if found or node.dropped:
            return
        for child in node.children:
            if isinstance(child, str):
                continue
            if child.tag == "h1":
                text = _WS_RE.sub(" ", _node_text(child)).strip()
                if text:
                    found.append(text)
                    return
            walk(child)

    walk(root)
    return found[0] if found else ""


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
        return "text"
    body_head = page.body.lstrip()[:64].lower()
    if body_head.startswith(("<!doctype", "<html", "<head", "<body")):
        return "html"
    if body_head.startswith(("{", "[")):
        return "json"
    return "html"


def extract_page(page: PageResult) -> EvidenceRecord:
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
