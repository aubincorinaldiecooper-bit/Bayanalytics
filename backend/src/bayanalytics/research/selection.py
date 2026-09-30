"""Laya's bounded choices while research runs (stages ``source_selection`` and
``data_identification``).

Laya never produces a value. Every question offers options that exist in this run: the hits a
search returned, the tables of a page that the deterministic filter in ``research.tables`` found
(``price_candidates`` / ``figure_candidates``), the columns of the chosen price table and the
lines of the chosen figures table (``tables.shortlist``). Deterministic code then reads what was
chosen (units, ``as_of`` cut, plausibility). A declined choice (``none``) or one below
``DATA_CHOICE_MIN_CONFIDENCE`` means the table or line is skipped, never guessed.

* ``order_hits``: which hits of a search most likely hold the intent's topic; the answer's
  probabilities order the hits the runner opens within its fetch budget. Engine order is the
  fallback when Laya is not available (no wrapper, or ``LAYA_INFERENCE_FAILED``).
* ``identify_page_data``: which table is the daily price history and which the quarterly
  results; which column is the close; which line is each metric.

Every question is asked with ``laya.started`` / ``laya.decision`` / ``laya.completed`` events
and its decisions are returned so the analysis records them with the others. A question whose
options do not fit Laya's head is asked again with half the options (down to two).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.laya.schemas import (
    CLOSE_COLUMN_KEY,
    FIGURES_TABLE_KEY,
    MAX_DYNAMIC_OPTIONS,
    NONE_OPTION,
    OPEN_ORDER_KEY,
    PRICE_TABLE_KEY,
    STAGE_DATA_IDENTIFICATION,
    STAGE_SOURCE_SELECTION,
    close_column_questions,
    figure_line_questions,
    line_key,
    result_order_questions,
    table_choice_questions,
)
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.research.market import (
    BROAD_MARKET,
    COMPANY,
    figures_preview,
    mentions_broad_market,
    mentions_symbol,
    price_preview,
)
from bayanalytics.research.provider import EvidenceRecord, SearchResult
from bayanalytics.research.sources import domain_of
from bayanalytics.research.tables import (
    FIGURE_METRICS,
    FigureCandidate,
    FigureLine,
    FigureSet,
    PriceCandidate,
    PriceTable,
    figure_candidates,
    page_tables,
    price_candidates,
    read_figures,
    read_price_history,
    shortlist,
)
from bayanalytics.schemas.common import ErrorCode, stable_id
from bayanalytics.schemas.decisions import ChoiceAnswer, LayaDecision, LayaQuestion, LayaQuestionSet

log = logging.getLogger(__name__)

DATA_CHOICE_MIN_CONFIDENCE = 0.5
"""Minimum Laya confidence (top choice probability) to read a table, a column or a line."""
MAX_LINE_OPTIONS = 6
MAX_TABLE_OPTIONS = 6


async def ask_shrinking(
    laya: LayaFinanceWrapper,
    stage: str,
    state: dict[str, Any],
    build: Callable[[int], dict[str, LayaQuestion]],
    count: int,
    segment_id: str | None,
    ctx: AnalysisContext,
) -> list[LayaDecision] | None:
    """Ask ``build(n)`` with ``n = count`` options, with ``laya.started`` / ``laya.decision`` /
    ``laya.completed`` events; ``n`` is halved (down to two) while the question does not fit
    Laya's head (``invalid_questions``). ``None`` when Laya could not answer
    (``LAYA_INFERENCE_FAILED``); any other failure (a cancellation) propagates."""
    n = count
    while True:
        try:
            questions = build(n)
        except ValueError:
            return None
        await ctx.event("laya.started", stage=stage, questions=len(questions))
        try:
            decisions = await laya.ask(
                LayaQuestionSet(
                    stage=stage, state=state, questions=questions, segment_id=segment_id
                ),
                ctx,
            )
        except AnalysisError as exc:
            if exc.code != ErrorCode.LAYA_INFERENCE_FAILED:
                raise
            await ctx.event("laya.completed", stage=stage, decisions=0)
            reason = (exc.details or {}).get("reason")
            if reason == "invalid_questions" and n > 2:
                n = max(2, n // 2)
                continue
            log.info("laya unavailable for %s (%s)", stage, reason)
            return None
        for decision in decisions:
            await ctx.event("laya.decision", **decision.event_view())
        await ctx.event("laya.completed", stage=stage, decisions=len(decisions))
        return decisions


def chosen_option(decisions: list[LayaDecision] | None, key: str) -> tuple[str | None, float]:
    """(chosen option, confidence) of the decision ``key``; ``None`` when missing or ``none``."""
    for decision in decisions or []:
        if decision.decision_type == key and isinstance(decision.answer, ChoiceAnswer):
            if decision.answer.choice == NONE_OPTION:
                return None, decision.confidence
            return decision.answer.choice, decision.confidence
    return None, 0.0


# --------------------------------------------------------------------------------------
# which hits to open first
# --------------------------------------------------------------------------------------


async def order_hits(
    laya: LayaFinanceWrapper | None,
    hits: list[SearchResult],
    *,
    symbol: str,
    topic: str,
    query: str,
    ctx: AnalysisContext,
) -> tuple[list[SearchResult], list[LayaDecision]]:
    """The hits in the order to open them: by Laya's probability that each holds ``topic``
    (the first ``MAX_DYNAMIC_OPTIONS`` hits are offered; the rest follow in engine order).
    Engine order when there is no choice to make or Laya is not available."""
    if laya is None or len(hits) < 2:
        return list(hits), []
    offered = hits[:MAX_DYNAMIC_OPTIONS]
    options = [
        (f"r{i + 1}", f"{hit.title} ({domain_of(hit.url)})") for i, hit in enumerate(offered)
    ]
    state = {
        "instrument": symbol,
        "question": topic,
        "query": query,
        "hits": [
            {
                "option": key,
                "site": domain_of(hit.url),
                "title": hit.title[:120],
                "published": hit.published_at.date().isoformat() if hit.published_at else None,
            }
            for (key, _), hit in zip(options, offered, strict=True)
        ],
    }
    decisions = await ask_shrinking(
        laya,
        STAGE_SOURCE_SELECTION,
        state,
        lambda n: result_order_questions(topic, options[:n]),
        len(options),
        stable_id("search", query),
        ctx,
    )
    answer = next(
        (
            d.answer
            for d in decisions or []
            if d.decision_type == OPEN_ORDER_KEY and isinstance(d.answer, ChoiceAnswer)
        ),
        None,
    )
    if answer is None:
        return list(hits), list(decisions or [])
    probability = {
        index: answer.probabilities.get(key, 0.0) for index, (key, _) in enumerate(options)
    }
    ranked = sorted(range(len(offered)), key=lambda i: (-probability[i], i))
    ordered = [offered[i] for i in ranked] + list(hits[len(offered) :])
    return ordered, list(decisions or [])


# --------------------------------------------------------------------------------------
# what a page holds
# --------------------------------------------------------------------------------------


@dataclass
class PageData:
    """What one page yielded after Laya's choices and the deterministic reading."""

    prices: PriceTable | None = None
    figures: FigureSet | None = None
    decisions: list[LayaDecision] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_prices(self) -> bool:
        return self.prices is not None

    @property
    def has_figures(self) -> bool:
        return self.figures is not None and bool(self.figures.figures)

    @property
    def has_data(self) -> bool:
        return self.has_prices or self.has_figures

    def preview(self) -> dict[str, Any] | None:
        if self.prices is not None:
            return price_preview(self.prices)
        if self.figures is not None:
            return figures_preview(self.figures)
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "price_points": len(self.prices.points) if self.prices else 0,
            "figures": len(self.figures.figures) if self.figures else 0,
        }


def _table_option(candidate: PriceCandidate | FigureCandidate) -> str:
    """How a table is offered: a price table by its header row and row count, a figures table
    by its first line labels and period count."""
    if isinstance(candidate, PriceCandidate):
        return f"{candidate.headers()} ({candidate.rows} rows)"
    labels = ", ".join(line.label for line in candidate.lines[:3])
    return f"{labels}; {len(candidate.periods)} periods"


def _table_state(candidate: PriceCandidate | FigureCandidate) -> str:
    if isinstance(candidate, PriceCandidate):
        return f"{candidate.headers()[:160]} ({candidate.rows} rows)"
    labels = ", ".join(line.label for line in candidate.lines[:8])
    return f"{candidate.describe()}: {labels}"[:230]


async def identify_page_data(
    laya: LayaFinanceWrapper | None,
    record: EvidenceRecord,
    *,
    source_id: str,
    symbol: str,
    role: str,
    as_of: date,
    ctx: AnalysisContext,
) -> PageData:
    """Ask Laya which of the page's candidate tables holds the ``role``'s data and read it.

    ``role`` is ``company`` (price history and quarterly figures of ``symbol``; the page must
    name the ticker) or ``broad_market`` (the S&P 500 price history only; the page must name
    the index). No candidates, no Laya, a declined or unsure choice: no data from the page.
    """
    page = PageData()
    tables = page_tables(record.extraction_method, record.structured)
    if not tables:
        return page
    if role == BROAD_MARKET:
        if not mentions_broad_market(record, tables):
            return page
        prices = price_candidates(tables, as_of)[:MAX_TABLE_OPTIONS]
        figures: list[FigureCandidate] = []
        subject = "S&P 500"
    else:
        if not mentions_symbol(symbol, record, tables):
            return page
        prices = price_candidates(tables, as_of)[:MAX_TABLE_OPTIONS]
        figures = [c for c in figure_candidates(tables) if c.index not in {p.index for p in prices}]
        figures = figures[:MAX_TABLE_OPTIONS]
        subject = symbol
    if not prices and not figures:
        return page
    if laya is None:
        page.notes.append("tables not read: no Laya decision available")
        return page
    price_options = [(f"t{c.index + 1}", _table_option(c)) for c in prices]
    figure_options = [(f"t{c.index + 1}", _table_option(c)) for c in figures]
    state = {
        "instrument": subject,
        "page": record.title[:160],
        "site": domain_of(record.final_url or record.url),
        "tables": [
            {"option": f"t{c.index + 1}", "table": _table_state(c)} for c in [*prices, *figures]
        ],
    }
    decisions = await ask_shrinking(
        laya,
        STAGE_DATA_IDENTIFICATION,
        state,
        lambda n: table_choice_questions(price_options[:n], figure_options[:n]),
        max(len(price_options), len(figure_options)),
        source_id,
        ctx,
    )
    page.decisions.extend(decisions or [])
    if decisions is None:
        page.notes.append("tables not read: Laya did not answer")
        return page
    by_key = {f"t{c.index + 1}": c for c in prices}
    chosen, confidence = chosen_option(decisions, PRICE_TABLE_KEY)
    if chosen in by_key and confidence >= DATA_CHOICE_MIN_CONFIDENCE:
        page.prices = await _read_prices(laya, by_key[chosen], subject, source_id, as_of, page, ctx)
    figure_by_key = {f"t{c.index + 1}": c for c in figures}
    chosen, confidence = chosen_option(decisions, FIGURES_TABLE_KEY)
    if chosen in figure_by_key and confidence >= DATA_CHOICE_MIN_CONFIDENCE:
        page.figures = await _read_figures(
            laya, figure_by_key[chosen], subject, source_id, as_of, page, ctx
        )
    return page


async def _read_prices(
    laya: LayaFinanceWrapper,
    candidate: PriceCandidate,
    subject: str,
    source_id: str,
    as_of: date,
    page: PageData,
    ctx: AnalysisContext,
) -> PriceTable | None:
    table = candidate.table
    columns = [(f"c{j + 1}", table.columns[j] or f"column {j + 1}") for j in candidate.value_cols][
        :MAX_DYNAMIC_OPTIONS
    ]
    sample = next((row for row in table.rows if row[candidate.date_col]), table.rows[0])
    state = {
        "instrument": subject,
        "table": candidate.headers()[:200],
        "columns": [
            {"option": key, "header": header, "example": sample[int(key[1:]) - 1]}
            for key, header in columns
        ],
    }
    decisions = await ask_shrinking(
        laya,
        STAGE_DATA_IDENTIFICATION,
        state,
        lambda n: close_column_questions(columns[:n]),
        len(columns),
        f"{source_id}:t{candidate.index + 1}",
        ctx,
    )
    page.decisions.extend(decisions or [])
    chosen, confidence = chosen_option(decisions, CLOSE_COLUMN_KEY)
    if chosen is None or confidence < DATA_CHOICE_MIN_CONFIDENCE:
        page.notes.append("price table skipped: no confident closing-price column")
        return None
    result = read_price_history(candidate, int(chosen[1:]) - 1, as_of)
    if result is None:
        page.notes.append("price table skipped: fewer than 20 valid daily rows")
    return result


async def _read_figures(
    laya: LayaFinanceWrapper,
    candidate: FigureCandidate,
    subject: str,
    source_id: str,
    as_of: date,
    page: PageData,
    ctx: AnalysisContext,
) -> FigureSet | None:
    offered: dict[str, list[FigureLine]] = {}
    for metric in FIGURE_METRICS:
        lines = shortlist(candidate, metric, MAX_LINE_OPTIONS)
        if lines:
            offered[metric] = lines
    if not offered:
        page.notes.append("figures table skipped: no line matches a reported figure")
        return None
    union: dict[str, str] = {}
    for lines in offered.values():
        for line in lines:
            union.setdefault(line.key, line.label)
    state = {
        "instrument": subject,
        "table": candidate.describe(),
        "lines": [{"option": key, "label": label[:80]} for key, label in union.items()],
    }
    decisions = await ask_shrinking(
        laya,
        STAGE_DATA_IDENTIFICATION,
        state,
        lambda n: figure_line_questions(
            {m: [(line.key, line.label) for line in lines[:n]] for m, lines in offered.items()},
            across=candidate.across,
        ),
        MAX_LINE_OPTIONS,
        f"{source_id}:t{candidate.index + 1}:lines",
        ctx,
    )
    page.decisions.extend(decisions or [])
    if decisions is None:
        return None
    picks: dict[str, tuple[FigureLine, float]] = {}
    for metric in offered:
        key, confidence = chosen_option(decisions, line_key(metric))
        line = candidate.line(key) if key else None
        if line is None or confidence < DATA_CHOICE_MIN_CONFIDENCE:
            continue
        clash = next((m for m, (held, _) in picks.items() if held.key == line.key), None)
        if clash is not None:
            # One line cannot be two metrics: keep the more confident reading, drop the other.
            if confidence <= picks[clash][1]:
                continue
            del picks[clash]
        picks[metric] = (line, confidence)
    if not picks:
        page.notes.append("figures table skipped: no confident line choice")
        return None
    return read_figures(candidate, {m: line for m, (line, _) in picks.items()}, as_of)


def role_for(intent: str) -> str:
    """Pages found for the sector-benchmark intent are read for the broad market's prices;
    every other page for the company's prices and figures."""
    return BROAD_MARKET if intent == "retrieve_sector_benchmark" else COMPANY
