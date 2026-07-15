"""
Profile Builder — runnable agent.

Runs ONCE per resume version. Takes a resume PDF + hardcoded preferences config,
produces ONE structured candidate profile consumed by the Fit Scorer.

Pure extraction + direct pass-through:
  - Resume fields: extracted faithfully — never invented, never inferred beyond
    what's written.
  - Preference fields: read directly from preferences_config — no parsing,
    no interpretation, passed through exactly as given.

PDF handling: Anthropic native document API — PDF passed as base64 document
block directly to Claude. No extra PDF library required.

Setup:  pip install "anthropic>=0.40" pydantic>=2
        export ANTHROPIC_API_KEY=...

Usage:
    from profile_builder import build_profile, PreferencesConfig, TargetComp

    prefs = PreferencesConfig(
        work_arrangement_preference="remote",
        target_compensation=TargetComp(min=120000, max=145000, currency="CAD")
    )
    result = build_profile("path/to/resume.pdf", prefs)
    print(result.model_dump_json(indent=2))
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import base64
import os
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field
from anthropic import Anthropic


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"   # GA for structured outputs; cost/quality fit
MAX_TOKENS = 2000
EMPTY_PDF_BYTES = 200         # files smaller than this are treated as empty

_client = Anthropic()         # reads ANTHROPIC_API_KEY from env


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Status(str, Enum):
    ok = "ok"
    rejected = "rejected"


class RejectReason(str, Enum):
    unreadable_pdf = "unreadable_pdf"
    empty_pdf = "empty_pdf"
    encrypted_pdf = "encrypted_pdf"
    invalid_preferences_config = "invalid_preferences_config"


class CurrentLevel(str, Enum):
    junior_pm = "junior_pm"
    mid_pm = "mid_pm"
    senior_pm = "senior_pm"
    staff_pm = "staff_pm"
    principal_pm = "principal_pm"
    unclassified = "unclassified"   # no PM title found; level_evidence populated


class ProfileWarningCode(str, Enum):
    # --- extraction warnings (surfaced from resume) ---
    role_dates_missing = "role_dates_missing"
    skills_section_missing = "skills_section_missing"
    level_unclassified = "level_unclassified"
    years_calculation_incomplete = "years_calculation_incomplete"
    certifications_not_found = "certifications_not_found"
    # --- safety valve ---
    other = "other"


class ProfileWarning(BaseModel):
    code: ProfileWarningCode
    note: str = ""   # required only for `other`


# ---------------------------------------------------------------------------
# Preferences config (input — hardcoded by the user, passed to build_profile)
# ---------------------------------------------------------------------------

class TargetComp(BaseModel):
    min: int
    max: int
    currency: str


class PreferencesConfig(BaseModel):
    work_arrangement_preference: str   # "remote" | "hybrid" | "remote_or_hybrid"
    target_compensation: TargetComp


# ---------------------------------------------------------------------------
# ProfileLLMParse — EXACTLY what the LLM returns.
# Zero Optional/union types (union budget = 0):
#   - string fields use "" sentinel when not found
#   - numeric fields use 0 / 0.0 sentinel + warning when not found
#   - list fields use empty list
# `stated` pattern is replaced by parse_warnings — a warning fires whenever
# a field couldn't be extracted confidently.
# ---------------------------------------------------------------------------

class ProfileLLMParse(BaseModel):
    # Identity
    candidate_name: str = ""
    current_level: CurrentLevel
    level_evidence: list[str] = Field(default_factory=list)   # PM-signal keywords
    level_note: str = ""                                       # both populated only
                                                               # when unclassified
    location: str = ""

    # Experience
    years_total: int = 0      # 0 + years_calculation_incomplete warning if unknown
    years_as_pm: float = 0.0  # PM-titled roles ONLY; strict definition

    # Skills
    skills: list[str] = Field(default_factory=list)
    ai_skills: list[str] = Field(default_factory=list)           # subset of skills
    unclassified_skills: list[str] = Field(default_factory=list) # prose-only, no header

    # Background
    certifications: list[str] = Field(default_factory=list)
    industry_background: list[str] = Field(default_factory=list)

    # Warnings
    parse_warnings: list[ProfileWarning] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Full output records
# ---------------------------------------------------------------------------

class CandidateProfile(ProfileLLMParse):
    """Complete profile = LLM parse + preferences merged in by code."""
    status: Status = Status.ok
    work_arrangement_preference: str = ""
    target_compensation: Optional[TargetComp] = None


class RejectedProfile(BaseModel):
    status: Status = Status.rejected
    reason: RejectReason
    message: str


CandidateProfileOutput = Union[CandidateProfile, RejectedProfile]


# ---------------------------------------------------------------------------
# Agent-level failure — raised to the orchestrator
# ---------------------------------------------------------------------------

class ProfileBuilderError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)


# ---------------------------------------------------------------------------
# System prompt — extraction rules
# Invoked only AFTER input guards pass.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Profile Builder, a PURE-EXTRACTION agent in the JobHunter AI Crew.
You receive a resume PDF and extract the candidate's professional profile into
a structured JSON object. You extract faithfully — never invent, never infer
beyond what is written.

NEVER:
- NEVER invent a skill not present in the resume. If it is not written, it does
  not exist. One invented skill compromises every downstream scoring run.
- NEVER infer a certification. If not explicitly listed, certifications stays [].
- NEVER map a non-PM title to the PM level ladder. "Fellow", "Consultant",
  "Analyst", "Founder", "BA", "QA" are NOT PM titles. If the most recent title
  does not map cleanly to the ladder, set current_level to "unclassified" and
  populate level_evidence and level_note instead.

ALWAYS:
- Extract skills from BOTH the Skills section AND responsibilities prose.
  Skills in prose go to unclassified_skills if there is no Skills section header.
- Populate ai_skills as a strict SUBSET of skills. Every term in ai_skills must
  also appear in skills. AI skills include: LLM, RAG, GPT, Claude, Gemini, AI
  agents, agentic workflows, generative AI, prompt engineering, fine-tuning,
  embeddings, vector database, NLP, ML, machine learning, deep learning, AI
  observability, foundation models, and any other AI/ML term you recognize.
- Use PM-titled roles ONLY for years_as_pm. Roles with "Product Manager" in the
  title count. BA, QA, Analyst, Fellow, Consultant do NOT count.
- Fire a parse_warning for every field you cannot extract confidently.

FIELD RULES:

current_level: map the most recent PM title to this ladder:
  junior_pm (0-2 yrs) | mid_pm (2-5 yrs) | senior_pm (5-8 yrs) |
  staff_pm (8-12 yrs) | principal_pm (12+ yrs)
  If the title is NOT a PM title -> current_level = "unclassified".
  Do NOT use years to assign a level when the title is ambiguous — use
  "unclassified" and populate level_evidence + level_note.

level_evidence: ONLY when current_level is "unclassified". List PM-signal
  keywords found in responsibilities: roadmap planning, stakeholder management,
  product strategy, backlog management, user research, discovery, prioritization,
  go-to-market, product roadmap, OKRs, sprint planning, product vision, etc.
  Empty list when current_level is NOT unclassified.

level_note: ONLY when current_level is "unclassified". Short plain-English note:
  "No PM title found. Skills suggest PM background — level unclassified."
  Empty string when current_level is NOT unclassified.

years_total: total career length from earliest role start to most recent role
  end (or present). Count overlapping roles without deduplication (both count).
  If dates are missing or unclear, use your best estimate and fire
  years_calculation_incomplete warning.

years_as_pm: sum of durations of roles with "Product Manager" in the title only.
  Overlapping PM roles both count. If no PM roles found, return 0.0.

skills: extract from Skills section first, then responsibilities prose. Each
  skill is a short phrase. Keep separate — never merge "Python/Java" into one.
  If no Skills section exists, extract all from prose -> unclassified_skills
  (not skills), and fire skills_section_missing warning.

certifications: only from an explicit Certifications, Credentials, or Education
  section. Never inferred. [] if none found — fire certifications_not_found
  warning if no section exists.

industry_background: infer from company descriptions in the Experience section.
  Short phrases: "enterprise B2B SaaS", "fintech", "banking", etc.

parse_warnings: fire a warning for every extraction problem:
  * role_dates_missing     -> one or more roles had no dates
  * skills_section_missing -> no Skills header found; skills in unclassified_skills
  * level_unclassified     -> no PM title found; level_evidence populated
  * years_calculation_incomplete -> date gaps; years may be understated
  * certifications_not_found -> no certifications section found
  * other                  -> anything else; put detail in "note"
"""


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def _read_pdf_bytes(path: str) -> bytes:
    """Read PDF file bytes. Raises FileNotFoundError if missing."""
    with open(path, "rb") as f:
        return f.read()


def _check_pdf(path: str) -> Optional[RejectedProfile]:
    """
    Deterministic PDF guards — run before any LLM cost.
    Returns a RejectedProfile if the file should be rejected, None if OK.
    """
    if not os.path.exists(path):
        return RejectedProfile(
            reason=RejectReason.unreadable_pdf,
            message=f"File not found: {path}. Please check the path and try again."
        )

    try:
        pdf_bytes = _read_pdf_bytes(path)
    except (IOError, OSError) as e:
        return RejectedProfile(
            reason=RejectReason.unreadable_pdf,
            message=f"Could not read file: {e}. Please provide an accessible PDF."
        )

    if len(pdf_bytes) < EMPTY_PDF_BYTES:
        return RejectedProfile(
            reason=RejectReason.empty_pdf,
            message="PDF appears to be empty. Please provide a complete resume PDF."
        )

    # Light encryption detection — encrypted PDFs declare /Encrypt in header.
    # No extra library needed; just a bytes scan of the first 2KB.
    header_sample = pdf_bytes[:2048]
    if b"/Encrypt" in header_sample:
        return RejectedProfile(
            reason=RejectReason.encrypted_pdf,
            message="PDF appears to be password-protected. Please provide an unencrypted PDF."
        )

    return None   # PDF passed all guards


def _enforce_ai_skills_subset(parse: ProfileLLMParse) -> ProfileLLMParse:
    """
    Cardinal sin check: ai_skills must be a strict subset of skills.
    If any ai_skill is missing from skills, add it to skills and fire a warning.
    This is a defensive correction — the agent should never produce this state,
    but we catch it rather than silently passing bad data downstream.
    """
    skills_set = set(s.lower() for s in parse.skills)
    rogue_ai_skills = [s for s in parse.ai_skills if s.lower() not in skills_set]

    if rogue_ai_skills:
        # Add missing ai_skills back to the main skills list (recovery)
        parse.skills = list(parse.skills) + rogue_ai_skills
        parse.parse_warnings = list(parse.parse_warnings) + [
            ProfileWarning(
                code=ProfileWarningCode.other,
                note=f"ai_skills subset violation corrected — added to skills: {rogue_ai_skills}"
            )
        ]
    return parse


# ---------------------------------------------------------------------------
# LLM call — Anthropic native document API + structured outputs
# ---------------------------------------------------------------------------

def _call_llm(pdf_bytes: bytes) -> ProfileLLMParse:
    """
    Pass the PDF natively to Claude as a base64 document block.
    No external PDF library needed — Claude reads the document directly.
    Raises ProfileBuilderError on model-level failures.
    """
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    resp = _client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Please parse this resume according to your instructions."
                    }
                ],
            }
        ],
        output_format=ProfileLLMParse,
    )

    if resp.stop_reason == "refusal":
        raise ProfileBuilderError("model_refusal", "model declined; output not schema-valid")
    if resp.stop_reason == "max_tokens":
        raise ProfileBuilderError("max_tokens", f"hit {MAX_TOKENS}-token cap; output may be incomplete")

    parsed = resp.parsed_output
    if parsed is None:
        raise ProfileBuilderError("no_output", "no parsed_output returned")

    return parsed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_profile(
    resume_pdf_path: str,
    preferences: PreferencesConfig,
) -> CandidateProfileOutput:
    """
    One resume PDF + preferences config -> one structured CandidateProfile.

    Usage:
        prefs = PreferencesConfig(
            work_arrangement_preference="remote",
            target_compensation=TargetComp(min=120000, max=145000, currency="CAD")
        )
        result = build_profile("resume.pdf", prefs)
        if result.status == "ok":
            # use result as the candidate profile in the Fit Scorer
            ...
        else:
            print(result.message)  # rejection message
    """
    # --- step 1: validate preferences (fast, no LLM cost) ---
    # Pydantic validates on construction; if it raised, preferences is invalid.
    # We catch any edge case where it arrives malformed.
    if preferences is None:
        return RejectedProfile(
            reason=RejectReason.invalid_preferences_config,
            message="No preferences config provided. Please pass a PreferencesConfig object and re-run."
        )

    # --- step 2: deterministic PDF guards ---
    rejection = _check_pdf(resume_pdf_path)
    if rejection:
        return rejection

    pdf_bytes = _read_pdf_bytes(resume_pdf_path)

    # --- step 3: LLM parse (schema-guaranteed via structured outputs) ---
    parse = _call_llm(pdf_bytes)

    # --- step 4: post-parse integrity check (cardinal sin guard) ---
    parse = _enforce_ai_skills_subset(parse)

    # --- step 5: assemble full profile with preferences merged in ---
    return CandidateProfile(
        **parse.model_dump(),
        work_arrangement_preference=preferences.work_arrangement_preference,
        target_compensation=preferences.target_compensation,
    )


# ---------------------------------------------------------------------------
# Demo — update RESUME_PATH before running
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RESUME_PATH = "path/to/your_resume.pdf"   # <-- update this

    prefs = PreferencesConfig(
        work_arrangement_preference="remote",
        target_compensation=TargetComp(min=120000, max=145000, currency="CAD")
    )

    result = build_profile(RESUME_PATH, prefs)
    print(result.model_dump_json(indent=2))
