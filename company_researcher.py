"""
Company Researcher — JobHunter AI Crew.

Searches five targeted platforms (website, LinkedIn, GitHub, Glassdoor, Twitter/news)
via SerpAPI and produces a PM-interview-ready company brief where every claim
has a cited source URL — and unciteable claims are suppressed, not invented.

CARDINAL SIN: any claim not backed by a retrieved source URL.
Claude's training data about a company is explicitly excluded as a source.
It may be stale, wrong, or refer to a different company with the same name.

ARCHITECTURE:
  - 5-6 SerpAPI queries (code-controlled, one per platform)
  - 7-day local JSON cache in company_cache/ directory
  - One LLM synthesis call (messages.parse, CompanyBriefLLM schema)
  - Final CompanyBrief assembled from LLM output + deterministic fields

TRIGGER: Apply and Conditional Apply decisions only. Orchestrator enforces this.

Setup:  pip install "anthropic>=0.40" pydantic>=2 requests
        export ANTHROPIC_API_KEY=...
        export SERPAPI_KEY=...

Usage:
    from company_researcher import research_company
    from job_analyst import parse_job

    job = parse_job(jd_text, job_url)
    brief = research_company(
        company_name=job.company_name.value,
        decision_level="apply",
        product_name=job.job_title.value,
        job_record=job,
    )
    print(brief.model_dump_json(indent=2))
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Union

import requests
from pydantic import BaseModel, Field
from anthropic import Anthropic

try:
    from job_analyst import OkRecord
except ImportError:
    OkRecord = None  # type: ignore — optional dependency

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
MAX_TOKENS_SYNTHESIS = 6000
CACHE_TTL_DAYS = 7
CACHE_DIR = "company_cache"
SERPAPI_RESULTS_PER_QUERY = 5

_client = Anthropic()


# ---------------------------------------------------------------------------
# LLM response schema — what the synthesis call produces
# Zero Optional / union types throughout
# ---------------------------------------------------------------------------

class KeyClaim(BaseModel):
    claim: str
    source: str = ""
    date: str = ""


class BriefSection(BaseModel):
    paragraph: str       # "" when stated:false
    key_claims: list[KeyClaim]
    stated: bool         # false = no citeable sources found


class CultureSection(BaseModel):
    paragraph: str
    key_claims: list[KeyClaim]
    stated: bool
    glassdoor_rating: str = ""
    glassdoor_positives: str = ""
    glassdoor_negatives: str = ""
    glassdoor_source: str = ""


class ProductSection(BaseModel):
    paragraph: str
    key_claims: list[KeyClaim]
    stated: bool
    github_signal: str = ""
    github_source: str = ""


class AIEvidence(BaseModel):
    signal: str
    type: str   # product_delivered | product_planned | hiring_signal | engineering_blog | marketing_copy
    source: str = ""
    date: str = ""


class AISignalOutput(BaseModel):
    classification: str  # ai_core | ai_enabled | non_ai | unknown
    evidence: list[AIEvidence]
    stated: bool


class CompanyBriefLLM(BaseModel):
    """What the LLM synthesis call produces."""
    confidence_overall: str          # high | medium | low
    history: BriefSection
    culture_and_people: CultureSection
    product_and_technology: ProductSection
    recent_developments: BriefSection
    current_stance: BriefSection
    ai_signal: AISignalOutput
    pm_talking_points: list[str]
    smart_questions: list[str]
    research_warnings: list[str]


# ---------------------------------------------------------------------------
# Full output models
# ---------------------------------------------------------------------------

class CompanySections(BaseModel):
    history: BriefSection
    culture_and_people: CultureSection
    product_and_technology: ProductSection
    recent_developments: BriefSection
    current_stance: BriefSection


class PlatformCoverage(BaseModel):
    website: bool = False
    linkedin: bool = False
    github: bool = False
    glassdoor: bool = False
    twitter: bool = False


class CompanyBrief(BaseModel):
    status: str = "ok"
    company_name: str
    research_date: str
    decision_level: str = ""
    confidence_overall: str
    cache_hit: bool = False
    sections: CompanySections
    ai_signal: AISignalOutput
    pm_talking_points: list[str]
    smart_questions: list[str]
    platform_coverage: PlatformCoverage
    research_warnings: list[str] = Field(default_factory=list)


class RejectedBrief(BaseModel):
    status: str = "rejected"
    reason: str
    message: str
    company_name: str = ""


CompanyBriefOutput = Union[CompanyBrief, RejectedBrief]


# ---------------------------------------------------------------------------
# Agent-level failure — raised to orchestrator
# ---------------------------------------------------------------------------

class CompanyResearcherError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """You are the Company Researcher for a PM job search crew.

Your job: synthesize web search results into a PM-interview-ready company brief.

CARDINAL SIN: stating, implying, or paraphrasing ANYTHING about this company
that is not directly supported by a retrieved search result provided to you.
- NEVER use your training data as a source for claims about this company
- NEVER guess or infer details not present in the search results
- If a section has no supporting results, set stated:false and paragraph:""
- One fabricated fact could send a candidate into an interview with wrong information

CARDINAL SIN EXAMPLES (all forbidden):
- "The company was founded in [year from your training data]"
- "They raised a Series X at [valuation from memory]"
- "Their CEO is [name from training data]"
- Anything you "know" about this company that isn't in the provided search results

RULES — every one is mandatory:
1. Every key_claims entry MUST have a source URL from the search results. No URL = no claim.
2. glassdoor_negatives: always surface negative themes when present. Hiding them fails the PM.
3. AI signals: label accurately. product_delivered = shipped product. marketing_copy = "we're excited about AI". Never conflate.
4. smart_questions: must be unanswerable from the company homepage. Generated from specific research signals only.
5. pm_talking_points: reference specific findings, not generic company facts.
6. research_warnings: fire for every gap (no GitHub, no Glassdoor, conflicting sources, etc.)
7. stated:false means no reliable source was found. Empty paragraph + empty key_claims.

CONFIDENCE:
- high: 3+ independent sources per major section
- medium: 1-2 sources per section  
- low: very thin coverage or company too small/new for meaningful research

AI SIGNAL CLASSIFICATION:
- ai_core: AI is the core product (not just a feature)
- ai_enabled: AI is a significant feature or roadmap direction
- non_ai: no AI signals found
- unknown: insufficient information

AI EVIDENCE TYPES (ordered by signal strength):
- product_delivered: actual shipped AI product with evidence
- product_planned: announced AI roadmap item with specifics
- hiring_signal: job postings requiring AI/ML skills
- engineering_blog: technical post about AI implementation  
- marketing_copy: vague "AI strategy" language, "we're excited about AI"

RESEARCH WARNINGS to use when appropriate:
company_not_found, minimal_web_presence, conflicting_sources,
sources_are_pr_only, no_ai_signals_detected, github_not_found,
glassdoor_not_found, linkedin_not_found, company_name_ambiguous, other
"""


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _get_serp_api_key() -> str:
    key = os.environ.get("SERPAPI_KEY") or os.environ.get("SERP_API_KEY")
    if not key:
        raise CompanyResearcherError(
            "missing_serpapi_key",
            "Set SERPAPI_KEY environment variable. Get a key at serpapi.com."
        )
    return key


def _search(query: str, num: int = SERPAPI_RESULTS_PER_QUERY) -> list[dict]:
    """Run a single SerpAPI query. Returns list of organic results."""
    try:
        params = {
            "api_key": _get_serp_api_key(),
            "q": query,
            "num": num,
            "hl": "en",
            "gl": "us",
        }
        resp = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # SerpAPI returns an error field on failure
        if "error" in data:
            raise CompanyResearcherError("serpapi_error", data["error"])

        return data.get("organic_results", [])

    except CompanyResearcherError:
        raise
    except requests.RequestException as e:
        raise CompanyResearcherError("search_network_error", str(e))


def _build_queries(company_name: str, product_name: str = "") -> list[str]:
    """Build 5-6 targeted search queries, one per platform."""
    c = company_name
    queries = [
        f'"{c}" company history founding story culture values mission',
        f'site:linkedin.com "{c}" product announcement leadership team 2025',
        f'site:github.com "{c}" AI machine learning repositories recent',
        f'site:glassdoor.com "{c}" culture salary reviews employees',
        f'"{c}" news announcement product launch 2025 '
        f'site:techcrunch.com OR site:venturebeat.com OR site:twitter.com OR site:x.com',
    ]
    if product_name:
        queries.append(f'"{c}" "{product_name}" features product')
    return queries


def _detect_platform_coverage(all_results: list[dict]) -> dict[str, bool]:
    """Determine which platforms returned usable results."""
    coverage = {
        "website": False,
        "linkedin": False,
        "github": False,
        "glassdoor": False,
        "twitter": False,
    }
    for r in all_results:
        link = (r.get("link") or "").lower()
        if "linkedin.com" in link:
            coverage["linkedin"] = True
        elif "github.com" in link:
            coverage["github"] = True
        elif "glassdoor.com" in link:
            coverage["glassdoor"] = True
        elif "twitter.com" in link or "x.com" in link:
            coverage["twitter"] = True
        else:
            coverage["website"] = True
    return coverage


def _format_results_for_llm(
    queries_and_results: list[tuple[str, list[dict]]]
) -> str:
    """Format all search results into a clean text block for the LLM."""
    lines = []
    for query, results in queries_and_results:
        lines.append(f"\n=== SEARCH QUERY: {query} ===")
        if not results:
            lines.append("(no results returned for this query)")
            continue
        for r in results:
            lines.append(f"TITLE: {r.get('title', '')}")
            lines.append(f"URL: {r.get('link', '')}")
            snippet = r.get("snippet") or r.get("rich_snippet", {})
            if isinstance(snippet, dict):
                snippet = str(snippet)
            lines.append(f"SNIPPET: {snippet}")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _normalize_company_name(company_name: str) -> str:
    """Normalize to safe filename."""
    return re.sub(r"[^\w]", "_", company_name.lower()).strip("_")


def _get_cache_path(company_name: str, cache_dir: str = CACHE_DIR) -> str:
    safe = _normalize_company_name(company_name)
    return os.path.join(cache_dir, f"{safe}.json")


def _read_cache(
    company_name: str,
    cache_dir: str = CACHE_DIR,
    ttl_days: int = CACHE_TTL_DAYS,
) -> CompanyBrief | None:
    """Return cached brief if fresh, else None."""
    path = _get_cache_path(company_name, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        research_date = datetime.fromisoformat(data.get("research_date", "2000-01-01"))
        if datetime.now() - research_date > timedelta(days=ttl_days):
            return None  # stale
        brief = CompanyBrief(**data)
        brief.cache_hit = True
        return brief
    except Exception:
        return None  # corrupt cache — re-research


def _write_cache(
    brief: CompanyBrief,
    company_name: str,
    cache_dir: str = CACHE_DIR,
) -> None:
    """Write brief to local JSON cache. Silently fails — never block on cache errors."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        path = _get_cache_path(company_name, cache_dir)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(brief.model_dump(), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[CompanyResearcher] Warning: cache write failed: {e}")



def _normalize_company_brief_llm(data: dict) -> dict:
    """Normalize LLM JSON to match CompanyBriefLLM schema exactly."""
    import re as _re2

    # --- Default empty section structures ---
    def _default_section():
        return {"paragraph": "", "key_claims": [], "stated": False}

    # --- Normalize standard sections ---
    for field in ["history", "recent_developments", "current_stance"]:
        if field not in data or not isinstance(data.get(field), dict):
            data[field] = _default_section()
        else:
            if "stated" not in data[field]:
                data[field]["stated"] = bool(data[field].get("paragraph"))

    if "culture_and_people" not in data or not isinstance(data.get("culture_and_people"), dict):
        data["culture_and_people"] = {**_default_section(),
            "glassdoor_rating": "", "glassdoor_positives": "",
            "glassdoor_negatives": "", "glassdoor_source": ""}
    else:
        c = data["culture_and_people"]
        if "stated" not in c:
            c["stated"] = bool(c.get("paragraph"))

    if "product_and_technology" not in data or not isinstance(data.get("product_and_technology"), dict):
        data["product_and_technology"] = {**_default_section(),
            "github_signal": "", "github_source": ""}
    else:
        p = data["product_and_technology"]
        if "stated" not in p:
            p["stated"] = bool(p.get("paragraph"))

    # --- Normalize key_claims in all sections ---
    for sec in ["history", "culture_and_people", "product_and_technology",
                "recent_developments", "current_stance"]:
        if sec in data and isinstance(data[sec], dict):
            claims = data[sec].get("key_claims", [])
            normalized = []
            for c in (claims if isinstance(claims, list) else []):
                if isinstance(c, str):
                    normalized.append({"claim": c, "source": "", "date": ""})
                elif isinstance(c, dict):
                    normalized.append({
                        "claim": c.get("claim", c.get("text", str(c))),
                        "source": c.get("source", c.get("url", "")),
                        "date": c.get("date", ""),
                    })
            data[sec]["key_claims"] = normalized

    # --- confidence_overall ---
    if "confidence_overall" not in data:
        data["confidence_overall"] = "medium"

    # --- ai_signal ---
    ai = data.get("ai_signal", {})
    if not isinstance(ai, dict):
        ai = {}
    if "stated" not in ai:
        ai["stated"] = bool(ai.get("evidence") or ai.get("classification", "") not in ("", "unknown"))
    if "classification" not in ai:
        ai["classification"] = "unknown"
    raw_ev = ai.get("evidence", []) if isinstance(ai.get("evidence"), list) else []
    norm_ev = []
    for ev in raw_ev:
        if isinstance(ev, str):
            norm_ev.append({"signal": ev, "type": "marketing_copy", "source": "", "date": ""})
        elif isinstance(ev, dict):
            norm_ev.append({
                "signal": ev.get("signal", ev.get("text", ev.get("detail", str(ev)))),
                "type": ev.get("type", "marketing_copy"),
                "source": ev.get("source", ev.get("url", "")),
                "date": ev.get("date", ""),
            })
    ai["evidence"] = norm_ev
    data["ai_signal"] = ai

    # --- pm_talking_points: extract strings from dicts ---
    pts = data.get("pm_talking_points", [])
    data["pm_talking_points"] = [
        pt if isinstance(pt, str)
        else next((pt[k] for k in ["point", "text", "talking_point", "value"] if k in pt and isinstance(pt[k], str)), str(pt))
        for pt in (pts if isinstance(pts, list) else [])
    ]

    # --- smart_questions: extract strings from dicts ---
    qs = data.get("smart_questions", [])
    data["smart_questions"] = [
        q if isinstance(q, str)
        else next((q[k] for k in ["question", "text", "value"] if k in q and isinstance(q[k], str)), str(q))
        for q in (qs if isinstance(qs, list) else [])
    ]

    # --- research_warnings: extract strings from dicts ---
    warns = data.get("research_warnings", [])
    data["research_warnings"] = [
        w if isinstance(w, str)
        else next((w[k] for k in ["detail", "message", "warning", "text"] if k in w and isinstance(w[k], str)), w.get("type", "warning"))
        for w in (warns if isinstance(warns, list) else [])
    ]

    return data


# ---------------------------------------------------------------------------
# LLM synthesis call
# ---------------------------------------------------------------------------

def _synthesize(
    company_name: str,
    queries_and_results: list[tuple[str, list[dict]]],
    decision_level: str,
    platform_coverage: dict[str, bool],
) -> CompanyBrief:
    """Synthesize search results into a structured CompanyBrief via one LLM call."""

    search_text = _format_results_for_llm(queries_and_results)
    today = datetime.now().strftime("%Y-%m-%d")

    user_content = (
        f"Company to research: {company_name}\n"
        f"Today's date: {today}\n"
        f"'Recent' means published after {(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')}.\n\n"
        f"Use ONLY the search results below. Do NOT use any training data knowledge about {company_name}.\n"
        f"Every claim needs a source URL from these results. stated:false if no URL found.\n\n"
        f"{search_text}"
    )

    import json as _json, re as _re
    _json_system = SYNTHESIS_SYSTEM_PROMPT + (
        "\n\nRespond with ONLY a valid JSON object. No preamble, no markdown fences."
    )
    resp = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_SYNTHESIS,
        system=_json_system,
        messages=[{"role": "user", "content": user_content}],
    )
    if resp.stop_reason == "max_tokens":
        raise CompanyResearcherError("max_tokens", "synthesis call truncated")
    _text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    _text = _re.sub(r"^```(?:json)?\s*", "", _text)
    _text = _re.sub(r"\s*```$", "", _text).strip()
    try:
        raw = _json.loads(_text)
        raw = _normalize_company_brief_llm(raw)
        llm = CompanyBriefLLM.model_validate(raw)
    except Exception as e:
        raise CompanyResearcherError("parse_error", f"JSON parse failed: {e}")

    # Add no_ai_signals_detected warning if applicable
    warnings = list(llm.research_warnings)
    if llm.ai_signal.classification == "unknown" and not llm.ai_signal.evidence:
        if "no_ai_signals_detected" not in warnings:
            warnings.append("no_ai_signals_detected")

    # Add platform-specific not_found warnings
    if not platform_coverage.get("github"):
        if "github_not_found" not in warnings:
            warnings.append("github_not_found")
    if not platform_coverage.get("glassdoor"):
        if "glassdoor_not_found" not in warnings:
            warnings.append("glassdoor_not_found")
    if not platform_coverage.get("linkedin"):
        if "linkedin_not_found" not in warnings:
            warnings.append("linkedin_not_found")

    return CompanyBrief(
        status="ok",
        company_name=company_name,
        research_date=today,
        decision_level=decision_level,
        confidence_overall=llm.confidence_overall,
        cache_hit=False,
        sections=CompanySections(
            history=llm.history,
            culture_and_people=llm.culture_and_people,
            product_and_technology=llm.product_and_technology,
            recent_developments=llm.recent_developments,
            current_stance=llm.current_stance,
        ),
        ai_signal=llm.ai_signal,
        pm_talking_points=llm.pm_talking_points,
        smart_questions=llm.smart_questions,
        platform_coverage=PlatformCoverage(**platform_coverage),
        research_warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def research_company(
    company_name: str = "",
    decision_level: str = "",
    product_name: str = "",
    job_record=None,
    cache_dir: str = CACHE_DIR,
    ttl_days: int = CACHE_TTL_DAYS,
) -> CompanyBriefOutput:
    """
    Research a company and produce a PM-interview-ready brief.

    Args:
        company_name:   Company name to research. Falls back to job_record.company_name.value.
        decision_level: "apply" | "conditional_apply" | "" (empty = no trigger check).
        product_name:   Optional product name — adds a 6th targeted search query.
        job_record:     Optional OkRecord from Job Analyst for context.
        cache_dir:      Directory for local JSON cache. Default: "company_cache".
        ttl_days:       Cache TTL in days. Default: 7.

    Returns:
        CompanyBrief on success, RejectedBrief on input rejection.
        Raises CompanyResearcherError on search/model failures (caught by orchestrator).
    """
    # --- step 1: input guards ---

    # Resolve company name from job_record if not provided directly
    if not company_name and job_record is not None:
        company_name = getattr(
            getattr(job_record, "company_name", None), "value", ""
        ) or ""

    if not company_name:
        return RejectedBrief(
            reason="missing_company_name",
            message="company_name is required. Pass it directly or via job_record.",
        )

    # Trigger gate: only run for Apply and Conditional Apply
    if decision_level and decision_level not in ("apply", "conditional_apply"):
        return RejectedBrief(
            reason="not_triggered",
            message=(
                f"Company Researcher only runs for Apply and Conditional Apply decisions. "
                f"Got: '{decision_level}'."
            ),
            company_name=company_name,
        )

    # --- step 2: check cache ---
    cached = _read_cache(company_name, cache_dir, ttl_days)
    if cached:
        print(f"[CompanyResearcher] Cache hit for '{company_name}' — skipping search.")
        return cached

    # --- step 3: run SerpAPI queries ---
    queries = _build_queries(company_name, product_name)
    queries_and_results: list[tuple[str, list[dict]]] = []
    all_results_flat: list[dict] = []
    any_results = False

    for query in queries:
        try:
            results = _search(query)
            queries_and_results.append((query, results))
            all_results_flat.extend(results)
            if results:
                any_results = True
        except CompanyResearcherError as e:
            # Log platform failure, continue with remaining queries
            print(f"[CompanyResearcher] Query failed: '{query}' — {e.kind}: {e.detail}")
            queries_and_results.append((query, []))

    # If ALL queries failed, reject
    if not any_results:
        return RejectedBrief(
            reason="search_api_error",
            message="All SerpAPI queries returned empty results. Check SERPAPI_KEY and network.",
            company_name=company_name,
        )

    # Detect platform coverage deterministically from returned URLs
    platform_coverage = _detect_platform_coverage(all_results_flat)

    # --- step 4: LLM synthesis ---
    brief = _synthesize(
        company_name=company_name,
        queries_and_results=queries_and_results,
        decision_level=decision_level,
        platform_coverage=platform_coverage,
    )

    # Add company_not_found warning if all sections are stated:false
    all_unstated = all([
        not brief.sections.history.stated,
        not brief.sections.culture_and_people.stated,
        not brief.sections.product_and_technology.stated,
        not brief.sections.recent_developments.stated,
        not brief.sections.current_stance.stated,
    ])
    if all_unstated and "company_not_found" not in brief.research_warnings:
        brief.research_warnings.append("company_not_found")
        brief.status = "partial"

    # Partial status when confidence is low
    if brief.confidence_overall == "low" and brief.status == "ok":
        brief.status = "partial"

    # --- step 5: write cache ---
    _write_cache(brief, company_name, cache_dir)

    return brief


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    company = sys.argv[1] if len(sys.argv) > 1 else "Babylist"
    print(f"[CompanyResearcher] Researching: {company}")
    print(f"[CompanyResearcher] Cache TTL: {CACHE_TTL_DAYS} days")
    print(f"[CompanyResearcher] Cache dir: {CACHE_DIR}/\n")

    result = research_company(
        company_name=company,
        decision_level="apply",
    )

    if isinstance(result, RejectedBrief):
        print(f"REJECTED: {result.reason} — {result.message}")
    else:
        print(f"Status: {result.status}")
        print(f"Cache hit: {result.cache_hit}")
        print(f"Confidence: {result.confidence_overall}")
        print(f"AI classification: {result.ai_signal.classification}")
        print(f"Platform coverage: {result.platform_coverage.model_dump()}")
        print(f"Warnings: {result.research_warnings}")
        print(f"\nHistory: {result.sections.history.paragraph[:200]}...")
        print(f"\nPM talking points:")
        for pt in result.pm_talking_points:
            print(f"  - {pt}")
        print(f"\nSmart questions:")
        for q in result.smart_questions:
            print(f"  - {q}")
