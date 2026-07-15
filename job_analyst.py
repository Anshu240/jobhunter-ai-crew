"""
Job Analyst — runnable agent (option b).

Pure-extraction agent for the JobHunter AI Crew: one raw JD in, one validated
structured record out. Built on the Anthropic SDK's NATIVE STRUCTURED OUTPUTS
(client.messages.parse + a Pydantic output_format), which constrains decoding so
the model's output is schema-valid by construction.

This file SUPERSEDES the option-(a) schema+prompt file. Keep ONE source of truth
— this one. (If you kept the other, reconcile to this.)

--------------------------------------------------------------------------------
SCHEMA REFINEMENT vs option (a) — and WHY:
Native structured outputs cap union-typed params (e.g. `int | null`) at 16,
because they are exponentially expensive to compile. The option-(a) schema used
Optional[...] (a union) on nearly every field and would have blown that budget.
Fix: string fields use a "" sentinel (required, non-null); only the numeric
min/max stay nullable. `stated` remains the single source of truth — a blank
value still means "JD was silent," because we read `stated`, never emptiness.
--------------------------------------------------------------------------------

DETERMINISTIC vs LLM boundary (spec section f):
  Code owns : empty/too_short rejection, job_id from URL, traceable,
              refusal / max_tokens handling (raised to the orchestrator).
  LLM owns  : only the JobParse payload, and structured outputs guarantees it
              is schema-valid.

Setup:  pip install "anthropic>=0.40" pydantic>=2
        export ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import re
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field
from anthropic import Anthropic


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"   # GA for structured outputs; cost/quality fit for parsing.
                              # "claude-opus-4-8" also works if you want max accuracy.
MAX_TOKENS = 2000
MIN_WORDS = 40                # too_short threshold — UNTUNED v1 guess (spec open item #2)

_client = Anthropic()         # reads ANTHROPIC_API_KEY from env


# ---------------------------------------------------------------------------
# Enums  (closed vocabularies)
# ---------------------------------------------------------------------------

class Status(str, Enum):
    ok = "ok"
    rejected = "rejected"


class RejectReason(str, Enum):
    empty_input = "empty_input"
    too_short = "too_short"


class SourceLang(str, Enum):
    en = "en"
    fr = "fr"


class ParseWarningCode(str, Enum):
    # --- posting-ambiguity tags (the MODEL emits these; earned from real postings) ---
    experience_implied_not_stated = "experience_implied_not_stated"   # seniority word, no years
    work_arrangement_missing = "work_arrangement_missing"             # no remote/hybrid/onsite
    location_missing = "location_missing"                             # no/imprecise location
    skills_in_prose = "skills_in_prose"                               # real skills, no header -> unclassified
    preferred_skills_not_enumerated = "preferred_skills_not_enumerated"  # "preferred quals" named, none listed
    salary_ambiguous = "salary_ambiguous"                             # spans levels / no currency / "competitive"
    # --- input-condition tag (CODE sets this, not the model; different class of flag) ---
    no_url_provided = "no_url_provided"                               # no job_url -> record not traceable
    # --- safety valve ---
    other = "other"                                                   # anything unlisted; detail goes in `note`


class Warning(BaseModel):
    code: ParseWarningCode
    note: str = ""   # required only for `other`; the plain-English detail. Empty for fixed tags.


class WorkArrangement(str, Enum):
    remote = "remote"
    hybrid = "hybrid"
    onsite = "onsite"
    unspecified = "unspecified"     # used when stated == false (avoids a union type)


class CompPeriod(str, Enum):
    year = "year"
    hour = "hour"
    unspecified = "unspecified"     # used when stated == false


# ---------------------------------------------------------------------------
# Field models — the `stated` pattern.
# stated is the ONLY source of truth for presence. "" / [] / "unspecified" are
# sentinels for "silent", never to be inferred-from. All fields REQUIRED so the
# emitted JSON schema marks them required (enforces "full schema every time").
# ---------------------------------------------------------------------------

class StringField(BaseModel):
    value: str = ""           # "" when stated == false
    stated: bool
    raw: str = ""             # "" when stated == false


class ListField(BaseModel):
    value: list[str] = Field(default_factory=list)
    stated: bool
    raw: str = ""


class WorkArrangementField(BaseModel):
    value: WorkArrangement = WorkArrangement.unspecified
    stated: bool
    raw: str = ""


class YearsExperience(BaseModel):
    # min/max stay nullable — "" makes no sense for a number. (2 union types.)
    min: Optional[int] = None
    max: Optional[int] = None
    stated: bool
    raw: str = ""


class Compensation(BaseModel):
    min: Optional[int] = None       # 2 union types
    max: Optional[int] = None
    currency: str = ""              # "" when not stated
    period: CompPeriod = CompPeriod.unspecified
    stated: bool
    raw: str = ""


# ---------------------------------------------------------------------------
# JobParse — EXACTLY what the LLM returns (the output_format schema).
# Total nullable/union params = 4 (years.min/max, comp.min/max) << 16 limit.
# ---------------------------------------------------------------------------

class JobParse(BaseModel):
    source_lang: SourceLang
    multiple_jds_detected: bool
    truncation_suspected: bool
    parse_warnings: list[Warning] = Field(default_factory=list)

    job_title: StringField
    company_name: StringField
    location: StringField
    work_arrangement: WorkArrangementField

    years_experience_required: YearsExperience
    required_skills: ListField
    nice_to_have_skills: ListField
    unclassified_skills: ListField
    certifications: ListField

    # AI signals — raw evidence for the Fit Scorer's AI-depth classification.
    # Pure extraction: any AI-related term literally present in the JD.
    # The Fit Scorer classifies these into AI-core / AI-enabled / Non-AI.
    # The Job Analyst NEVER classifies — it only surfaces the raw evidence.
    ai_signals: ListField

    compensation: Compensation


# ---------------------------------------------------------------------------
# Full output records (what downstream consumes). Branch on `status` first.
# ---------------------------------------------------------------------------

class OkRecord(JobParse):
    status: Status = Status.ok
    job_id: Optional[str] = None    # parsed from URL by code; None if no URL
    traceable: bool = False         # False => orphan record (no URL given)


class RejectedRecord(BaseModel):
    status: Status = Status.rejected
    reason: RejectReason
    job_id: Optional[str] = None


JobAnalystOutput = Union[OkRecord, RejectedRecord]


# ---------------------------------------------------------------------------
# Agent-level failure (NOT a normal record) — raised to the orchestrator.
# Covers refusal / truncated-by-max_tokens / empty model output. Spec section f:
# the agent cannot reliably self-report its own failure, so the orchestrator
# catches this.
# ---------------------------------------------------------------------------

class JobAnalystError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind          # "model_refusal" | "max_tokens" | "no_output"
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)


# ---------------------------------------------------------------------------
# System prompt — the behavior rules.
# Invoked only AFTER the length guard, so empty/too_short never reach the model.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Job Analyst, a PURE-EXTRACTION agent in the JobHunter AI Crew.
You receive the text of ONE job description and describe what it LITERALLY
STATES. You extract; you never judge.

OUT OF SCOPE — a downstream agent does these, not you:
- Do NOT score, rank, or assess fit.
- Do NOT classify how "AI" the role is.
- Do NOT flag concerns or red flags.
- Do NOT infer seniority, years, or skills that are not written in the text.

NEVER:
- NEVER fabricate a value. If the JD does not state it, set "stated": false and
  leave the value empty. Never infer, estimate, or fill from typical/market
  knowledge. (Example: no salary given -> compensation stays unstated. Do not
  guess a market rate.)
- NEVER guess which bucket a skill belongs to. Headerless or ambiguous skills go
  to "unclassified_skills".
- NEVER paraphrase, summarize, or editorialize when translating. Translation
  only.

THE "stated" FLAG is the source of truth for presence:
- "stated": true  -> the JD said it. Put the value in "value" and the exact
  original words in "raw".
- "stated": false -> the JD was silent. Leave "value" empty ("" or []), "raw" "",
  enums "unspecified", numbers null.

FIELD RULES:
- company_name: extract from ANYWHERE in the JD — company header, "About Us/Company"
  section, job URL, email domain, or any mention in the text. "Join Stripe..." -> "Stripe".
  Most JDs state the company name somewhere. Only set stated:false if it truly appears
  NOWHERE in the text.
- years_experience_required: "5+ years" -> min 5, max null. "3-5 years" -> min 3,
  max 5. Exact "5 years" -> min 5, max 5. Silent -> stated false, min/max null.
  Do NOT infer years from a title like "Senior".
- required_skills: under explicit "Requirements / Must-have / Qualifications".
- nice_to_have_skills: under explicit "Preferred / Nice-to-have / Bonus".
- unclassified_skills: skills with no clear header or ambiguous placement.
- Keep each skill separate. "Python/Java" -> two entries.
- work_arrangement.value is remote | hybrid | onsite (or unspecified). Nuance
  goes in raw, e.g. value "remote", raw "Remote (Ontario only)".
  Classification rules:
  * remote   = fully remote, no office requirement, work from anywhere
  * hybrid   = some office required ("50% in office", "2-3 days in office",
               "flexible", "may vary depending on role", "some in-person",
               "occasional travel", any % in office language)
  * onsite   = fully in office, no remote option stated at all
  * When in doubt between hybrid and onsite: choose hybrid.
  * "50% in office" is HYBRID, not onsite.
- compensation: "competitive salary" -> stated true, min/max null, raw
  "competitive salary". Total silence -> stated false.
- ai_signals: extract ANY AI-related term literally present in the JD — anywhere
  in the text (title, requirements, responsibilities, company description). This
  includes but is not limited to: LLM, RAG, GPT, Claude, Gemini, AI agents,
  agentic workflows, generative AI, prompt engineering, fine-tuning, embeddings,
  vector database, NLP, ML, machine learning, deep learning, AI observability,
  foundation models, computer vision, reinforcement learning, AI infrastructure,
  AI evaluation, and any other AI/ML term you recognize. Do NOT classify or judge
  the role's AI depth — just extract the terms that are literally there.
  stated false if no AI terms appear anywhere in the JD.

LANGUAGE:
- Input may be English or French. Set source_lang accordingly.
- If French, NORMALIZE all values to English (e.g. "gestion de projet" ->
  "project management"). Translation only — never add a skill not in the text.

MESSY INPUT:
- Not actually a job description: parse anyway; nearly everything comes back
  stated false. Do not refuse.
- Two postings concatenated: parse the FIRST, set multiple_jds_detected true.
- Contradictory content (e.g. "Entry level" title + "8+ years" body): capture
  BOTH, preserve each in raw, do NOT resolve.
- Cuts off mid-content: parse what is present, set truncation_suspected true.
- parse_warnings: flag anything hard to parse using ONLY these tags:
    * experience_implied_not_stated  -> a seniority word (e.g. "Senior") but no years given
    * work_arrangement_missing       -> no remote/hybrid/onsite stated
    * location_missing               -> no location, or only a country/region (too vague)
    * skills_in_prose                -> real skills appeared in narrative, not under a header
    * preferred_skills_not_enumerated-> posting mentions "preferred quals" but lists none
    * salary_ambiguous               -> pay spans levels, lacks a currency, or is "competitive"
    * other                          -> anything not covered above; put a short plain-English
                                        description in "note"
  Each warning is {"code": <tag>, "note": ""}. Use "note" ONLY for "other".
  Do NOT use "no_url_provided" — that tag is set automatically and is not your job.
  Empty list if nothing was ambiguous.

OUTPUT FORMAT — every field uses the "stated" wrapper. Examples:
  String field:  {"value": "Stripe", "stated": true, "raw": "Stripe"}
  String absent: {"value": "", "stated": false, "raw": ""}
  List field:    {"value": ["Python", "LLMs"], "stated": true, "raw": "Python, LLMs"}
  List absent:   {"value": [], "stated": false, "raw": ""}
  compensation.period must be exactly one of: "year", "hour", "unspecified"
    ("annual" and "yearly" -> "year"; "hourly" -> "hour"; unknown -> "unspecified")
  work_arrangement.value must be exactly one of: "remote", "hybrid", "onsite", "unspecified"

"""


# ---------------------------------------------------------------------------
# Deterministic helpers (no LLM)
# ---------------------------------------------------------------------------

_JOB_ID_PATTERNS = (
    re.compile(r"/jobs/view/(\d+)"),          # .../jobs/view/3987654321
    re.compile(r"[?&]currentJobId=(\d+)"),    # .../jobs/search/?currentJobId=3987654321
    re.compile(r"-(\d+)(?:/|\?|$)"),          # .../title-at-company-3987654321
)


def extract_job_id(job_url: Optional[str]) -> Optional[str]:
    if not job_url:
        return None
    for pat in _JOB_ID_PATTERNS:
        m = pat.search(job_url)
        if m:
            return m.group(1)
    return None


def _reject(reason: RejectReason, job_id: Optional[str]) -> RejectedRecord:
    return RejectedRecord(reason=reason, job_id=job_id)



def _normalize_llm_output(data: dict) -> dict:
    """
    Normalize LLM JSON output to match JobParse schema exactly.
    Handles cases where the LLM returns natural formats instead of
    the exact {value, stated, raw} wrapper structure.
    """
    # --- StringField normalization ---
    string_fields = ["job_title", "company_name", "location"]
    for field in string_fields:
        val = data.get(field)
        if val is None:
            data[field] = {"value": "", "stated": False, "raw": ""}
        elif isinstance(val, str):
            data[field] = {"value": val, "stated": bool(val), "raw": val}
        elif isinstance(val, dict) and "stated" not in val:
            v = val.get("value", "")
            data[field] = {"value": v, "stated": bool(v), "raw": val.get("raw", v)}

    # Ensure 'raw' and 'value' inside StringFields are always plain strings
    for field in string_fields:
        if field in data and isinstance(data[field], dict):
            raw = data[field].get("raw", "")
            if isinstance(raw, dict):
                data[field]["raw"] = str(raw.get("value", "") or raw.get("raw", "") or "")
            elif not isinstance(raw, str):
                data[field]["raw"] = str(raw)
            val = data[field].get("value", "")
            if isinstance(val, dict):
                data[field]["value"] = str(val.get("value", "") or val.get("name", "") or "")
                if not data[field]["value"]:
                    data[field]["stated"] = False
            elif not isinstance(val, str):
                data[field]["value"] = str(val)

    # --- ListField normalization ---
    list_fields = ["required_skills", "nice_to_have_skills",
                   "unclassified_skills", "certifications", "ai_signals"]
    for field in list_fields:
        val = data.get(field)
        if val is None:
            data[field] = {"value": [], "stated": False, "raw": ""}
        elif isinstance(val, list):
            # LLM returned a plain list — extract strings
            items = []
            for item in val:
                if isinstance(item, str):
                    items.append(item)
                elif isinstance(item, dict):
                    # Handle {"skill": "..."}, {"name": "..."}, {"value": "..."}
                    for key in ["skill", "name", "value", "text", "item"]:
                        if key in item and isinstance(item[key], str):
                            items.append(item[key])
                            break
            data[field] = {
                "value": items,
                "stated": bool(items),
                "raw": ", ".join(items),
            }
        elif isinstance(val, dict) and "stated" not in val:
            # Malformed dict — reset to empty
            data[field] = {"value": [], "stated": False, "raw": ""}

    # --- WorkArrangementField normalization ---
    arr = data.get("work_arrangement", {})
    if isinstance(arr, str):
        v = arr.lower()
        data["work_arrangement"] = {"value": v, "stated": True, "raw": arr}
    elif isinstance(arr, dict):
        v = arr.get("value", "unspecified")
        if isinstance(v, str):
            v_lower = v.lower()
            mapping = {
                "in-office": "onsite", "in office": "onsite",
                "on-site": "onsite", "on site": "onsite",
                "fully remote": "remote", "work from home": "remote",
                "wfh": "remote", "fully-remote": "remote",
                "partially remote": "hybrid", "flexible": "hybrid",
            }
            arr["value"] = mapping.get(v_lower, v_lower if v_lower in ("remote", "hybrid", "onsite") else "unspecified")
        data["work_arrangement"] = arr

    # --- Compensation period normalization ---
    comp = data.get("compensation", {})
    if isinstance(comp, dict):
        period = comp.get("period", "unspecified")
        if isinstance(period, str):
            period_map = {
                "annual": "year", "yearly": "year", "per year": "year",
                "per annum": "year", "annually": "year",
                "hourly": "hour", "per hour": "hour", "hr": "hour",
                "": "unspecified",
            }
            comp["period"] = period_map.get(period.lower(), period if period in ("year", "hour", "unspecified") else "unspecified")
        if "stated" not in comp:
            comp["stated"] = bool(comp.get("min") or comp.get("max") or comp.get("raw"))
        data["compensation"] = comp

    # --- YearsExperience normalization ---
    yrs = data.get("years_experience_required", {})
    if isinstance(yrs, dict) and "stated" not in yrs:
        yrs["stated"] = bool(yrs.get("min") or yrs.get("max"))
        data["years_experience_required"] = yrs

    return data


# ---------------------------------------------------------------------------
# The LLM call — structured output. Raises JobAnalystError on agent failure.
# ---------------------------------------------------------------------------

def _call_llm(jd_text: str) -> JobParse:
    """Call LLM with JSON prompt. Avoids messages.parse grammar timeout."""
    import json
    import re as _re

    json_system = (
        SYSTEM_PROMPT
        + "\n\nRespond with ONLY a valid JSON object. "
        + "No preamble, no markdown fences, no explanation. Just the JSON."
    )

    resp = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=json_system,
        messages=[{"role": "user", "content": jd_text}],
    )

    if resp.stop_reason == "max_tokens":
        raise JobAnalystError("max_tokens", f"hit {MAX_TOKENS}-token cap; output truncated")

    text = "".join(
        block.text for block in resp.content
        if hasattr(block, "text")
    ).strip()

    # Strip markdown fences if present
    text = _re.sub(r"^```(?:json)?\s*", "", text)
    text = _re.sub(r"\s*```$", "", text).strip()

    try:
        data = json.loads(text)
        data = _normalize_llm_output(data)
        return JobParse.model_validate(data)
    except Exception as e:
        raise JobAnalystError("parse_error", f"JSON parse/validation failed: {e}")


def parse_job(jd_text: str, job_url: Optional[str] = None) -> JobAnalystOutput:
    job_id = extract_job_id(job_url)

    # --- deterministic input guards (before any LLM cost) ---
    if jd_text is None or not jd_text.strip():
        return _reject(RejectReason.empty_input, job_id)
    if len(jd_text.split()) < MIN_WORDS:
        return _reject(RejectReason.too_short, job_id)

    # --- LLM parse (schema-guaranteed) ---
    parse = _call_llm(jd_text)

    # no_url_provided is an INPUT-condition flag, set by code (the model can't know
    # whether a URL was pasted). Different class from the posting-ambiguity tags.
    warnings = list(parse.parse_warnings)
    if job_id is None:
        warnings.append(Warning(code=ParseWarningCode.no_url_provided))

    # --- assemble full record with deterministic fields ---
    fields = parse.model_dump()
    fields["parse_warnings"] = [w.model_dump() for w in warnings]
    return OkRecord(
        **fields,
        job_id=job_id,
        traceable=job_id is not None,   # no URL -> orphan record, stated as fact
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = """About the role
    We are hiring a Senior AI Product Manager to own our LLM-powered features.
    Requirements: 5+ years of product management experience, strong SQL,
    experience with RAG systems and prompt design. Preferred: experience with
    Stripe, familiarity with Next.js. This is a remote role, Canada only.
    Compensation: competitive salary."""
    sample_url = "https://www.linkedin.com/jobs/view/3987654321/"

    result = parse_job(sample, sample_url)
    print(result.model_dump_json(indent=2))
