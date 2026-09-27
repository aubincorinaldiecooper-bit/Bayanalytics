"""SEC EDGAR structured access: ticker directory, submissions and XBRL company facts.

Everything goes through the shared ``Fetcher`` so the SEC user agent, the 10 req/s fair-access
limit, robots and the disk cache apply. Filings are never stored whole: the runner keeps the
filing index metadata and, budget allowing, a 600-character excerpt of the primary document.

XBRL rows follow the contract agreed with the normalization module::

    {concept, metric, value, unit, start, end, fy, fp, form, filed, accn, frame,
     source_id, basis, currency}

``fp`` is EDGAR's fiscal period of the *filing* ("FY" for a 10-K row even when the row's
start..end spans one quarter); durations must be derived from ``start``/``end``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from importlib import resources
from typing import Any

from pydantic import BaseModel, Field

from bayanalytics.config import Settings
from bayanalytics.research.dates import ensure_utc, parse_date_lenient
from bayanalytics.research.extract import EXCERPT_CHARS, excerpt_of, extract_page
from bayanalytics.research.fetch import DEFAULT_TTL_S, EDGAR_TICKERS_TTL_S, PageFetcher
from bayanalytics.research.provider import ResearchProviderError
from bayanalytics.research.sources import classify_freshness
from bayanalytics.schemas.common import new_id, utcnow
from bayanalytics.schemas.evidence import SourceRecord

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"
FACT_FORMS: frozenset[str] = frozenset({"10-K", "10-Q", "20-F", "40-F", "10-K/A", "10-Q/A"})
DEFAULT_FILING_FORMS: tuple[str, ...] = ("10-K", "10-Q", "8-K")

# Concept -> canonical metric. Order matters: for each metric the first concept present in the
# company's facts wins and the later fallbacks are ignored (no double counting).
CONCEPT_MAP: tuple[tuple[str, str], ...] = (
    ("us-gaap:Revenues", "revenue"),
    ("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "revenue"),
    ("us-gaap:SalesRevenueNet", "revenue"),
    ("us-gaap:GrossProfit", "gross_profit"),
    ("us-gaap:OperatingIncomeLoss", "operating_income"),
    ("us-gaap:NetIncomeLoss", "net_income"),
    ("us-gaap:EarningsPerShareDiluted", "eps_diluted"),
    ("us-gaap:EarningsPerShareBasic", "eps_basic"),
    ("us-gaap:NetCashProvidedByUsedInOperatingActivities", "operating_cash_flow"),
    ("us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", "capex"),
    ("dei:EntityCommonStockSharesOutstanding", "shares_outstanding"),
    ("us-gaap:CommonStockSharesOutstanding", "shares_outstanding"),
    ("us-gaap:CashAndCashEquivalentsAtCarryingValue", "cash_and_equivalents"),
    ("us-gaap:LongTermDebt", "total_debt"),
    ("us-gaap:LongTermDebtNoncurrent", "long_term_debt_noncurrent"),
    ("us-gaap:LongTermDebtCurrent", "long_term_debt_current"),
    ("us-gaap:StockholdersEquity", "stockholders_equity"),
    ("us-gaap:Assets", "total_assets"),
    ("us-gaap:ResearchAndDevelopmentExpense", "research_and_development"),
    ("us-gaap:DepreciationDepletionAndAmortization", "depreciation_amortization"),
    ("us-gaap:DepreciationAndAmortization", "depreciation_amortization"),
)
UNIT_MAP: dict[str, str] = {
    "USD": "USD",
    "USD/shares": "USD/shares",
    "shares": "shares",
    "pure": "pure",
}
_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")


# --------------------------------------------------------------------------------------
# seed
# --------------------------------------------------------------------------------------


def load_company_tickers_seed() -> list[dict[str, Any]]:
    """The packaged EDGAR ticker seed (``research/data/company_tickers_seed.json``)."""
    path = resources.files("bayanalytics.research.data").joinpath("company_tickers_seed.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    companies = payload.get("companies") if isinstance(payload, dict) else None
    return [dict(row) for row in companies or [] if isinstance(row, dict)]


def parse_company_tickers(payload: Any) -> list[dict[str, Any]]:
    """EDGAR's ``{"0": {cik_str, ticker, title}, ...}`` (or a list) -> list of rows."""
    if isinstance(payload, dict) and "companies" in payload:
        payload = payload["companies"]
    rows: list[dict[str, Any]] = []
    items: list[Any]
    if isinstance(payload, dict):
        items = [payload[k] for k in sorted(payload, key=lambda k: int(k) if k.isdigit() else 0)]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ResearchProviderError("company_tickers payload has an unexpected shape")
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            cik = int(item["cik_str"])
            ticker = str(item["ticker"]).upper()
            title = str(item["title"])
        except (KeyError, TypeError, ValueError):
            continue
        row: dict[str, Any] = {"cik_str": cik, "ticker": ticker, "title": title}
        for extra in ("exchange", "aliases"):
            if extra in item:
                row[extra] = item[extra]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------


class Filing(BaseModel):
    form: str
    filing_date: date
    report_date: date | None = None
    accession: str
    primary_document: str | None = None
    primary_doc_description: str | None = None
    url: str
    index_url: str
    cik: int

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")


class EdgarSubmissions(BaseModel):
    cik: int
    name: str
    tickers: list[str] = Field(default_factory=list)
    exchanges: list[str] = Field(default_factory=list)
    sic: str | None = None
    sic_description: str | None = None
    fiscal_year_end: str | None = None  # "MMDD"
    state_of_incorporation: str | None = None
    former_names: list[dict[str, Any]] = Field(default_factory=list)  # {name, from, to}
    filings: list[Filing] = Field(default_factory=list)
    url: str
    retrieved_at: datetime
    from_cache: bool = False

    @property
    def name_history(self) -> list[str]:
        return [str(item.get("name")) for item in self.former_names if item.get("name")]

    def latest_filings(
        self,
        forms: tuple[str, ...] = DEFAULT_FILING_FORMS,
        limit: int = 6,
        as_of: datetime | date | None = None,
    ) -> list[Filing]:
        cutoff = _as_date(as_of)
        wanted = set(forms) if forms else None
        rows = [
            f
            for f in self.filings
            if (wanted is None or f.form in wanted) and (cutoff is None or f.filing_date <= cutoff)
        ]
        rows.sort(key=lambda f: (f.filing_date, f.accession), reverse=True)
        return rows[:limit]


class CompanyFacts(BaseModel):
    cik: int
    entity_name: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    source: SourceRecord
    rows_filtered_after_as_of: int = 0
    concepts_used: dict[str, str] = Field(default_factory=dict)  # metric -> concept

    def __len__(self) -> int:
        return len(self.rows)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _as_date(value: datetime | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value).date()
    return value


def normalize_cik(cik: int | str) -> int:
    text = str(cik).strip().upper().removeprefix("CIK")
    try:
        return int(text)
    except ValueError as exc:
        raise ResearchProviderError(f"invalid CIK: {cik!r}") from exc


def filing_urls(cik: int, accession: str, primary_document: str | None) -> tuple[str, str]:
    nodash = accession.replace("-", "")
    index_url = ARCHIVE_URL.format(
        cik=cik, accession_nodash=nodash, document=f"{accession}-index.htm"
    )
    if primary_document:
        url = ARCHIVE_URL.format(cik=cik, accession_nodash=nodash, document=primary_document)
    else:
        url = index_url
    return url, index_url


def fiscal_period_label(
    report_date: date | None, fiscal_year_end: str | None, form: str | None = None
) -> str | None:
    """Best-effort fiscal label: "FY2025" for annual forms, "Q3 FY2025" for 10-Qs."""
    if report_date is None:
        return None
    form = form or ""
    if not fiscal_year_end or len(fiscal_year_end) != 4 or not fiscal_year_end.isdigit():
        return report_date.isoformat()
    fye_month = int(fiscal_year_end[:2])
    if fye_month < 1 or fye_month > 12:
        return report_date.isoformat()
    # Fiscal year = calendar year in which the fiscal year ends.
    fy = report_date.year if report_date.month <= fye_month else report_date.year + 1
    if form.startswith(("10-K", "20-F", "40-F")):
        return f"FY{fy}"
    if form.startswith("10-Q"):
        months_into_year = (report_date.month - fye_month - 1) % 12 + 1
        quarter = (months_into_year - 1) // 3 + 1
        return f"Q{quarter} FY{fy}"
    return report_date.isoformat()


def parse_submissions(
    payload: dict[str, Any], url: str, retrieved_at: datetime
) -> EdgarSubmissions:
    try:
        cik = normalize_cik(payload.get("cik") or 0)
    except ResearchProviderError:
        cik = 0
    recent = (payload.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []
    accessions = recent.get("accessionNumber") or []
    primary_docs = recent.get("primaryDocument") or []
    primary_desc = recent.get("primaryDocDescription") or []
    filings: list[Filing] = []
    for index, accession in enumerate(accessions):
        if not isinstance(accession, str) or not _ACCESSION_RE.match(accession):
            continue
        filing_date = parse_date_lenient(_at(filing_dates, index))
        if filing_date is None:
            continue
        primary_document = _at(primary_docs, index) or None
        doc_url, index_url = filing_urls(cik, accession, primary_document)
        filings.append(
            Filing(
                form=str(_at(forms, index) or ""),
                filing_date=filing_date,
                report_date=parse_date_lenient(_at(report_dates, index)),
                accession=accession,
                primary_document=primary_document,
                primary_doc_description=_at(primary_desc, index) or None,
                url=doc_url,
                index_url=index_url,
                cik=cik,
            )
        )
    former = []
    for item in payload.get("formerNames") or []:
        if isinstance(item, dict) and item.get("name"):
            former.append(
                {"name": item.get("name"), "from": item.get("from"), "to": item.get("to")}
            )
    sic = payload.get("sic")
    return EdgarSubmissions(
        cik=cik,
        name=str(payload.get("name") or ""),
        tickers=[str(t) for t in payload.get("tickers") or []],
        exchanges=[str(e) for e in payload.get("exchanges") or []],
        sic=str(sic) if sic not in (None, "") else None,
        sic_description=payload.get("sicDescription") or None,
        fiscal_year_end=payload.get("fiscalYearEnd") or None,
        state_of_incorporation=payload.get("stateOfIncorporation") or None,
        former_names=former,
        filings=filings,
        url=url,
        retrieved_at=retrieved_at,
    )


def _at(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def parse_company_facts(
    payload: dict[str, Any],
    *,
    source_id: str,
    as_of: datetime | date | None = None,
) -> tuple[list[dict[str, Any]], int, dict[str, str]]:
    """Return ``(rows, filtered_after_as_of, concepts_used)`` in the XBRL row contract."""
    facts = payload.get("facts") or {}
    cutoff = _as_date(as_of)
    rows: list[dict[str, Any]] = []
    filtered = 0
    concepts_used: dict[str, str] = {}
    for concept, metric in CONCEPT_MAP:
        if metric in concepts_used:
            continue
        taxonomy, _, name = concept.partition(":")
        entry = (facts.get(taxonomy) or {}).get(name)
        if not isinstance(entry, dict):
            continue
        units = entry.get("units") or {}
        concept_rows: list[dict[str, Any]] = []
        for raw_unit, items in units.items():
            unit = UNIT_MAP.get(raw_unit)
            if unit is None or not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                form = item.get("form")
                if form not in FACT_FORMS:
                    continue
                filed = parse_date_lenient(item.get("filed"))
                end = parse_date_lenient(item.get("end"))
                value = item.get("val")
                if filed is None or end is None or not isinstance(value, int | float):
                    continue
                if cutoff is not None and filed > cutoff:
                    filtered += 1
                    continue
                start = parse_date_lenient(item.get("start"))
                fy = item.get("fy")
                fp = item.get("fp")
                concept_rows.append(
                    {
                        "concept": concept,
                        "metric": metric,
                        "value": float(value),
                        "unit": unit,
                        "start": start.isoformat() if start else None,
                        "end": end.isoformat(),
                        "fy": int(fy) if isinstance(fy, int | float) else None,
                        "fp": str(fp) if fp else None,
                        "form": str(form),
                        "filed": filed.isoformat(),
                        "accn": str(item.get("accn") or ""),
                        "frame": str(item["frame"]) if item.get("frame") else None,
                        "source_id": source_id,
                        "basis": "gaap",
                        "currency": "USD" if unit in ("USD", "USD/shares") else None,
                    }
                )
        if concept_rows:
            concepts_used[metric] = concept
            rows.extend(concept_rows)
    rows.sort(key=lambda r: (r["metric"], r["end"], r["start"] or "", r["filed"]))
    return rows, filtered, concepts_used


# --------------------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------------------


class EdgarClient:
    def __init__(self, fetcher: PageFetcher, settings: Settings) -> None:
        self._fetcher = fetcher
        self._settings = settings
        self.seed_fallback = False

    async def company_tickers(self) -> list[dict[str, Any]]:
        """Live EDGAR ticker directory (24 h cache); falls back to the packaged seed."""
        try:
            page = await self._fetcher.open(
                COMPANY_TICKERS_URL, ttl_s=EDGAR_TICKERS_TTL_S, accept="application/json"
            )
            rows = parse_company_tickers(json.loads(page.body))
            if not rows:
                raise ResearchProviderError("company_tickers.json contained no companies")
            self.seed_fallback = False
            return rows
        except (ResearchProviderError, ValueError):
            self.seed_fallback = True
            return load_company_tickers_seed()

    async def submissions(self, cik: int | str) -> EdgarSubmissions:
        number = normalize_cik(cik)
        url = SUBMISSIONS_URL.format(cik=number)
        page = await self._fetcher.open(url, ttl_s=DEFAULT_TTL_S, accept="application/json")
        try:
            payload = json.loads(page.body)
        except ValueError as exc:
            raise ResearchProviderError(f"submissions for CIK {number} is not JSON") from exc
        if not isinstance(payload, dict):
            raise ResearchProviderError(f"submissions for CIK {number} has an unexpected shape")
        parsed = parse_submissions(payload, url, page.fetched_at)
        parsed.from_cache = page.from_cache
        if parsed.cik == 0:
            parsed.cik = number
            for filing in parsed.filings:
                filing.cik = number
                filing.url, filing.index_url = filing_urls(
                    number, filing.accession, filing.primary_document
                )
        return parsed

    async def company_facts(
        self,
        cik: int | str,
        *,
        as_of: datetime | date | None = None,
        symbol: str | None = None,
        intent: str | None = "retrieve_earnings_history",
    ) -> CompanyFacts:
        number = normalize_cik(cik)
        url = COMPANY_FACTS_URL.format(cik=number)
        page = await self._fetcher.open(url, ttl_s=DEFAULT_TTL_S, accept="application/json")
        try:
            payload = json.loads(page.body)
        except ValueError as exc:
            raise ResearchProviderError(f"companyfacts for CIK {number} is not JSON") from exc
        if not isinstance(payload, dict):
            raise ResearchProviderError(f"companyfacts for CIK {number} has an unexpected shape")
        source_id = new_id("src")
        rows, filtered, used = parse_company_facts(payload, source_id=source_id, as_of=as_of)
        latest_filed = max((r["filed"] for r in rows), default=None)
        published_at = (
            datetime.fromisoformat(latest_filed).replace(tzinfo=UTC) if latest_filed else None
        )
        reference = ensure_utc(as_of) if isinstance(as_of, datetime) else utcnow()
        source = SourceRecord(
            source_id=source_id,
            url=url,
            title="SEC XBRL company facts",
            publisher="SEC EDGAR",
            source_type="regulatory_filing",
            published_at=published_at,
            retrieved_at=page.fetched_at,
            symbol=symbol,
            excerpt=f"{len(rows)} XBRL facts across {len(used)} metrics for CIK {number}",
            content_hash=None,
            extraction_method="json",
            freshness=classify_freshness(published_at, reference),
            redistribution="allowed",
            terms_note="US government work; SEC fair-access policy applies to retrieval",
            research_intent=intent,
            metadata={
                "cik": number,
                "rows": len(rows),
                "rows_filtered_after_as_of": filtered,
                "concepts_used": used,
                "from_cache": page.from_cache,
            },
        )
        return CompanyFacts(
            cik=number,
            entity_name=payload.get("entityName") or None,
            rows=rows,
            source=source,
            rows_filtered_after_as_of=filtered,
            concepts_used=used,
        )

    def filing_source_record(
        self,
        filing: Filing,
        *,
        symbol: str | None,
        as_of: datetime,
        fiscal_year_end: str | None = None,
        intent: str | None = "retrieve_latest_filing",
        excerpt: str = "",
    ) -> SourceRecord:
        """Provenance for a filing index entry; the full document is never stored."""
        published = datetime(
            filing.filing_date.year,
            filing.filing_date.month,
            filing.filing_date.day,
            tzinfo=UTC,
        )
        return SourceRecord(
            source_id=new_id("src"),
            url=filing.index_url,
            title=f"{filing.form} filed {filing.filing_date.isoformat()}",
            publisher="SEC EDGAR",
            source_type="regulatory_filing",
            published_at=published,
            retrieved_at=utcnow(),
            symbol=symbol,
            fiscal_period=fiscal_period_label(filing.report_date, fiscal_year_end, filing.form),
            excerpt=excerpt[:EXCERPT_CHARS],
            content_hash=None,
            extraction_method="edgar_submissions",
            freshness=classify_freshness(published, as_of),
            redistribution="allowed",
            terms_note="US government work; SEC fair-access policy applies to retrieval",
            research_intent=intent,
            metadata={
                "form": filing.form,
                "accession": filing.accession,
                "report_date": filing.report_date.isoformat() if filing.report_date else None,
                "primary_document_url": filing.url,
                "primary_doc_description": filing.primary_doc_description,
                "cik": filing.cik,
            },
        )

    async def fetch_filing_excerpt(self, filing: Filing) -> str:
        """Fetch the primary document and return only its lead excerpt (<= 600 chars)."""
        page = await self._fetcher.open(filing.url, ttl_s=DEFAULT_TTL_S)
        record = extract_page(page)
        return excerpt_of(record.text)
