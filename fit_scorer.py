"""
Fit Scorer — runnable agent.

Takes one structured job record (from Job Analyst) and one structured candidate
profile (from Profile Builder) and produces a weighted fit score + decision.

ARCHITECTURE — two LLM calls, code owns all math:
  Call 1 (structured): LLM makes judgment calls only — AI classification,
    industry classification, seniority match, near-match skill detection.
    No numbers. No scores.
  Code: computes ALL numeric scores from the LLM's classifications.
    Zero arithmetic hallucination risk. Cardinal sin is impossible here.
  Call 2 (plain text): LLM writes 3-sentence reasoning AFTER the code has
    computed the real scores — so reasoning references actual numbers.

This file imports from job_analyst.py and profile_builder.py.
Keep all three files in the same directory.

Setup:  pip install "anthropic>=0.40" pydantic>=2
        export ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Local imports — job_analyst.py and profile_builder.py must be in same dir
# ---------------------------------------------------------------------------
try:
    from job_analyst import OkRecord, RejectedRecord, Status as JAStatus
    from profile_builder import CandidateProfile, TargetComp
except ImportError as e:
    raise ImportError(
        f"Could not import from job_analyst.py or profile_builder.py. "
        f"Make sure all three files are in the same directory. Error: {e}"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
MAX_TOKENS_JUDGMENTS = 1500
MAX_TOKENS_REASONING = 500

_client = Anthropic()

# Scoring thresholds
APPLY_THRESHOLD       = 75.0
CONDITIONAL_THRESHOLD = 55.0
MAYBE_THRESHOLD       = 45.0

# Boundary-rule tiebreakers (top two weighted dimensions)
REQUIRED_SKILLS_TIEBREAK = 15.0   # out of 25
AI_DEPTH_TIEBREAK         = 12.0  # out of 20


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FitStatus(str, Enum):
    ok = "ok"
    rejected = "rejected"


class FitRejectReason(str, Enum):
    upstream_rejection = "upstream_rejection"   # Job Analyst returned rejected record
    missing_profile = "missing_profile"         # no candidate profile passed


class AIClassification(str, Enum):
    ai_core    = "ai_core"      # 20 pts — product IS the AI
    ai_enabled = "ai_enabled"   # 14 pts — AI is a major feature
    non_ai     = "non_ai"       #  5 pts — AI incidental or absent


class IndustryClassification(str, Enum):
    ai_core    = "ai_core"      # 15 pts
    ai_enabled = "ai_enabled"   # 11 pts
    adjacent   = "adjacent"     #  7 pts — fintech, SaaS, enterprise tech
    unrelated  = "unrelated"    #  3 pts
    unknown    = "unknown"      #  5 pts — not determinable


class SeniorityMatch(str, Enum):
    exact      = "exact"       # 10 pts — Senior = Senior
    one_down   = "one_down"    #  8 pts — overqualified (Senior → Mid)
    one_up     = "one_up"      #  6 pts — stretch (Senior → Staff)
    two_down   = "two_down"    #  4 pts — two levels above (Senior → Junior)
    mismatch   = "mismatch"    #  0 pts — major gap (Director, VP, Associate)


class ScoringFlag(str, Enum):
    salary_not_stated             = "salary_not_stated"
    work_arrangement_not_stated   = "work_arrangement_not_stated"
    ai_signals_empty              = "ai_signals_empty"
    skills_list_empty             = "skills_list_empty"
    location_restriction_detected = "location_restriction_detected"
    multiple_jds_detected         = "multiple_jds_detected"
    insufficient_data             = "insufficient_data"   # 3+ flags → low confidence


# ---------------------------------------------------------------------------
# LLM Judgment schema (Call 1) — judgments ONLY, no numbers
# ---------------------------------------------------------------------------

class LLMJudgments(BaseModel):
    ai_classification: AIClassification
    industry_classification: IndustryClassification
    seniority_match: SeniorityMatch
    # Near-matches: skills the LLM identified as semantically equivalent
    # but not exact string matches. Format: "required ≈ candidate"
    near_match_skills: list[str] = Field(default_factory=list)
    # True when JD mentions geographic restriction (e.g. "US only", "US residents")
    location_restriction_detected: bool = False


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class DimensionScore(BaseModel):
    score: float
    max: float
    detail: str


class DimensionScores(BaseModel):
    required_skills: DimensionScore
    ai_genai_depth:  DimensionScore
    seniority_fit:   DimensionScore
    nice_to_have:    DimensionScore
    work_arrangement: DimensionScore
    industry_alignment: DimensionScore
    compensation:    DimensionScore


class FitScoreResult(BaseModel):
    status: FitStatus = FitStatus.ok
    job_id: Optional[str] = None
    candidate_name: str = ""
    total_score: float
    decision: str           # apply | conditional_apply | maybe | skip
    condition: str = ""     # populated only for conditional_apply
    reasoning: str = ""     # 3 sentences: strength / weakness / action

    ai_classification: str

    dimension_scores: DimensionScores

    matched_skills:       list[str] = Field(default_factory=list)
    missing_skills:       list[str] = Field(default_factory=list)
    near_match_skills:    list[str] = Field(default_factory=list)
    matched_nice_to_have: list[str] = Field(default_factory=list)
    missing_nice_to_have: list[str] = Field(default_factory=list)

    scoring_flags: list[str] = Field(default_factory=list)


class RejectedFitScore(BaseModel):
    status: FitStatus = FitStatus.rejected
    reason: FitRejectReason
    message: str
    job_id: Optional[str] = None


FitScoreOutput = Union[FitScoreResult, RejectedFitScore]


# ---------------------------------------------------------------------------
# Agent-level failure — raised to orchestrator
# ---------------------------------------------------------------------------

class FitScorerError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)


# ---------------------------------------------------------------------------
# Deterministic scoring functions — CODE owns all math, no LLM arithmetic
# ---------------------------------------------------------------------------

def _match_skills(
    required: list[str],
    candidate_skills: list[str],
) -> tuple[list[str], list[str]]:
    """
    Flexible matching — four layers in order:
    1. Exact case-insensitive match
    2. Required skill is a substring of any candidate skill
    3. Any candidate skill is a substring of the required skill
    4. Keyword overlap — meaningful words (length > 3) shared between
       required skill and any candidate skill
    Returns (matched, missing).
    """
    candidate_lower = [s.lower() for s in candidate_skills]
    candidate_set   = set(candidate_lower)

    matched, missing = [], []
    for skill in required:
        skill_lower = skill.lower()

        # Layer 1: exact match
        if skill_lower in candidate_set:
            matched.append(skill)
            continue

        # Layer 2: required skill contained in a candidate skill
        if any(skill_lower in c for c in candidate_lower):
            matched.append(skill)
            continue

        # Layer 3: candidate skill contained in required skill (min 4 chars)
        if any(c in skill_lower for c in candidate_lower if len(c) > 3):
            matched.append(skill)
            continue

        # Layer 4: keyword overlap — words longer than 3 chars
        req_words  = {w for w in skill_lower.split() if len(w) > 3}
        cand_words = set()
        for c in candidate_lower:
            cand_words.update(w for w in c.split() if len(w) > 3)

        if req_words & cand_words:
            matched.append(skill)
            continue

        missing.append(skill)

    return matched, missing


def _score_required_skills(matched: list[str], total: int) -> tuple[float, str]:
    if total == 0:
        return 0.0, "No required skills listed in this posting — scored 0"
    score = round((len(matched) / total) * 25, 2)
    return score, f"{len(matched)} of {total} required skills matched"


def _score_ai_depth(cls: AIClassification) -> tuple[float, str]:
    table = {
        AIClassification.ai_core:    (20.0, "AI-core — product IS the AI"),
        AIClassification.ai_enabled: (14.0, "AI-enabled — AI is a major feature"),
        AIClassification.non_ai:     ( 5.0, "Non-AI role"),
    }
    return table[cls]


def _score_seniority(match: SeniorityMatch) -> tuple[float, str]:
    table = {
        SeniorityMatch.exact:     (10.0, "Exact level match"),
        SeniorityMatch.one_down:  ( 8.0, "Overqualified by one level"),
        SeniorityMatch.one_up:    ( 6.0, "Stretch — one level above candidate"),
        SeniorityMatch.two_down:  ( 4.0, "Two levels above candidate"),
        SeniorityMatch.mismatch:  ( 0.0, "Major level mismatch"),
    }
    return table[match]


def _score_nice_to_have(matched: list[str], total: int) -> tuple[float, str]:
    if total == 0:
        return 5.0, "No nice-to-haves listed — full points by default"
    score = round((len(matched) / total) * 5, 2)
    return score, f"{len(matched)} of {total} nice-to-have skills matched"


def _score_work_arrangement(job_record: OkRecord) -> tuple[float, str]:
    arr = job_record.work_arrangement
    if not arr.stated or arr.value.value == "unspecified":
        return 7.0, "Work arrangement not stated — neutral score"
    val = arr.value.value.lower()
    if val == "remote":
        return 15.0, "Remote — matches preference"
    if val == "hybrid":
        return 11.0, "Hybrid (some office required) — workable but not ideal; confirm office expectations with recruiter"
    if val == "onsite":
        return 0.0, "Onsite only — does not match remote preference"
    return 7.0, "Work arrangement unclear — neutral score"


def _score_compensation(
    job_record: OkRecord,
    target: TargetComp,
) -> tuple[float, str]:
    comp = job_record.compensation
    if not comp.stated:
        return 5.0, "Salary not stated — neutral score"
    if comp.min is None and comp.max is None:
        return 5.0, f"Salary mentioned but not numeric ({comp.raw or 'competitive'}) — neutral"

    # USD-only role when candidate targets CAD
    if comp.currency and comp.currency.upper() == "USD" and target.currency.upper() == "CAD":
        return 7.0, f"USD ${comp.min:,}–${comp.max or comp.min:,} — strong signal, currency uncertainty"

    job_min = comp.min or 0
    job_max = comp.max or job_min

    if job_min >= target.min:
        return 10.0, f"{comp.currency} ${job_min:,}–${job_max:,} — within or above target range"
    if job_max >= target.min:
        return 7.0, f"{comp.currency} ${job_min:,}–${job_max:,} — overlaps target range"
    return 3.0, f"{comp.currency} ${job_min:,}–${job_max:,} — below target range"


def _score_industry(cls: IndustryClassification) -> tuple[float, str]:
    table = {
        IndustryClassification.ai_core:    (15.0, "AI-core company"),
        IndustryClassification.ai_enabled: (11.0, "AI-enabled company"),
        IndustryClassification.adjacent:   ( 7.0, "Adjacent industry — fintech / SaaS / enterprise tech"),
        IndustryClassification.unrelated:  ( 3.0, "Unrelated industry"),
        IndustryClassification.unknown:    ( 5.0, "Industry not determinable — neutral"),
    }
    return table[cls]


def _make_decision(
    total: float,
    required_score: float,
    ai_score: float,
) -> str:
    """
    Hard-threshold decision with boundary rule.
    At exactly 75.0 or 55.0 or 45.0, check top two dimensions.
    """
    def tiebreak_favors_higher() -> bool:
        return (
            required_score >= REQUIRED_SKILLS_TIEBREAK
            and ai_score >= AI_DEPTH_TIEBREAK
        )

    if total > APPLY_THRESHOLD:
        return "apply"
    if total == APPLY_THRESHOLD:
        return "apply" if tiebreak_favors_higher() else "conditional_apply"
    if total > CONDITIONAL_THRESHOLD:
        return "conditional_apply"
    if total == CONDITIONAL_THRESHOLD:
        return "conditional_apply" if tiebreak_favors_higher() else "maybe"
    if total > MAYBE_THRESHOLD:
        return "maybe"
    if total == MAYBE_THRESHOLD:
        return "maybe" if tiebreak_favors_higher() else "skip"
    return "skip"


def _find_condition(dim_scores: DimensionScores) -> str:
    """
    For conditional_apply: identify the dimension(s) that dragged score below 75%.
    Returns the worst-ratio dimension as the named condition.
    """
    dims = {
        "Required Skills":    (dim_scores.required_skills.score,    25.0, dim_scores.required_skills.detail),
        "AI/GenAI Depth":     (dim_scores.ai_genai_depth.score,     20.0, dim_scores.ai_genai_depth.detail),
        "Seniority Fit":      (dim_scores.seniority_fit.score,      10.0, dim_scores.seniority_fit.detail),
        "Work Arrangement":   (dim_scores.work_arrangement.score,   15.0, dim_scores.work_arrangement.detail),
        "Industry Alignment": (dim_scores.industry_alignment.score, 15.0, dim_scores.industry_alignment.detail),
        "Compensation":       (dim_scores.compensation.score,       10.0, dim_scores.compensation.detail),
    }
    worst_name = min(dims, key=lambda k: dims[k][0] / dims[k][1])
    score, max_score, detail = dims[worst_name]
    return f"{worst_name} — {detail} ({score:.0f}/{max_score:.0f})"


def _compute_flags(job_record: OkRecord, judgments: LLMJudgments) -> list[str]:
    flags = []
    if not job_record.compensation.stated:
        flags.append(ScoringFlag.salary_not_stated.value)
    if job_record.work_arrangement.value.value == "unspecified":
        flags.append(ScoringFlag.work_arrangement_not_stated.value)
    if not job_record.ai_signals.stated or not job_record.ai_signals.value:
        flags.append(ScoringFlag.ai_signals_empty.value)
    if not job_record.required_skills.stated or not job_record.required_skills.value:
        flags.append(ScoringFlag.skills_list_empty.value)
    if judgments.location_restriction_detected:
        flags.append(ScoringFlag.location_restriction_detected.value)
    if job_record.multiple_jds_detected:
        flags.append(ScoringFlag.multiple_jds_detected.value)
    # Insufficient data: 3+ degraded flags (excluding multiple_jds)
    degraded = [
        f for f in flags
        if f != ScoringFlag.multiple_jds_detected.value
    ]
    if len(degraded) >= 3:
        flags.append(ScoringFlag.insufficient_data.value)
    return flags


# ---------------------------------------------------------------------------
# System prompt — Call 1: judgment calls only
# ---------------------------------------------------------------------------

JUDGMENT_SYSTEM_PROMPT = """\
You are the Fit Scorer judgment module. You receive a structured job record and
a structured candidate profile (both as JSON). Your job is to make FOUR
judgment calls — nothing more. You do NOT compute scores. You do NOT produce
numbers. The code that calls you will compute all numbers from your judgments.

JUDGMENT 1 — ai_classification:
Classify the role's AI depth using the job's ai_signals field and company
context. Choose exactly one:
  ai_core    -> the product IS the AI (e.g. AI observability platform, LLM API,
                AI coding tool, AI agent framework). Long ai_signals list with
                core AI infrastructure terms.
  ai_enabled -> AI is a significant feature but not the whole product (e.g.
                conversational AI inside a telecom, AI support agent at a
                payments company, AI builder role at a consumer platform).
  non_ai     -> AI is incidental, buzzword-only, or absent.

JUDGMENT 2 — industry_classification:
Classify the company's industry. Choose exactly one:
  ai_core    -> company's primary product is AI technology
  ai_enabled -> technology company where AI is a major differentiator
  adjacent   -> fintech, enterprise SaaS, banking, consulting, B2B tech
  unrelated  -> retail, healthcare, media, education, government, other
  unknown    -> cannot determine from available information

JUDGMENT 3 — seniority_match:
Compare the job's required seniority (from title, years_experience_required,
and any explicit level mentions) against the candidate's current_level and
years_as_pm. Choose exactly one:
  exact     -> job targets the same level as the candidate
  one_down  -> job is one level below candidate (candidate overqualified)
  one_up    -> job is one level above candidate (stretch)
  two_down  -> job is two levels below candidate
  mismatch  -> major gap in either direction (Director/VP or Associate/entry)

Seniority ladder (low to high):
  junior_pm -> mid_pm -> senior_pm -> staff_pm -> principal_pm

If job lists two levels (e.g. "Senior or Staff"), use the lower one (benefit
of the doubt for the candidate).

JUDGMENT 4 — near_match_skills:
You will receive a list of MISSING required skills — skills not found by exact
string matching in the candidate's profile. Your job: identify which of these
missing skills are semantically equivalent to a skill the candidate DOES have.
Format each near-match as: "required_skill ≈ candidate_skill"
Example: "AI platforms ≈ Claude API, RAG pipelines"
Only flag genuine equivalents. If no near-matches exist, return [].

JUDGMENT 5 — location_restriction_detected:
Return true if the job explicitly restricts to candidates in a specific country
or region (e.g. "US only", "must be located in the US", "US residents only").
Return false if remote/hybrid with no geographic restriction, or if not stated.

Return ONLY the five judgments as a JSON object matching the schema.
Do NOT include scores, recommendations, or any other content.
"""


REASONING_SYSTEM_PROMPT = """\
You are the Fit Scorer reasoning module. You receive a summary of a job fit
assessment that has already been scored by code. Your job is to write the
"reasoning" field — exactly 3 sentences in plain English, readable by a
non-technical recruiter.

Sentence 1: What is STRONG about this fit (the highest-scoring dimensions).
Sentence 2: What is WEAK or limiting (the lowest-scoring dimension or the
  named condition).
Sentence 3: A specific, actionable recommendation — what to do next
  (e.g. "apply and lead with X", "address Y in cover letter", "skip unless Z").

Reference the actual scores by name and number. Be direct. No jargon.
No markdown. Plain paragraph, 3 sentences.
"""


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _call_judgments(
    job_record: OkRecord,
    profile: CandidateProfile,
    missing_skills: list[str],
) -> LLMJudgments:
    """
    Call 1: structured judgment. No scores, no math.
    Passes both records as JSON + the missing skills list for near-match detection.
    """
    user_content = json.dumps({
        "job_record": job_record.model_dump(),
        "candidate_profile": profile.model_dump(),
        "missing_required_skills": missing_skills,
    }, default=str)

    import json as _json, re as _re
    _json_system = JUDGMENT_SYSTEM_PROMPT + (
        "\n\nRespond with ONLY a valid JSON object. No preamble, no markdown fences."
    )
    resp = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_JUDGMENTS,
        temperature=0,
        system=_json_system,
        messages=[{"role": "user", "content": user_content}],
    )
    if resp.stop_reason == "max_tokens":
        raise FitScorerError("max_tokens", "judgment call truncated")
    _text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    _text = _re.sub(r"^```(?:json)?\s*", "", _text)
    _text = _re.sub(r"\s*```$", "", _text).strip()
    try:
        return LLMJudgments.model_validate(_json.loads(_text))
    except Exception as e:
        raise FitScorerError("parse_error", f"JSON parse failed: {e}")


def _call_reasoning(
    job_title: str,
    company: str,
    total_score: float,
    decision: str,
    condition: str,
    dim_scores: DimensionScores,
    scoring_flags: list[str],
) -> str:
    """
    Call 2: plain-text reasoning. Sees the actual computed scores.
    Returns a 3-sentence string. Uses messages.create (not .parse) — no schema.
    """
    dim_summary = "\n".join([
        f"  Required Skills:    {dim_scores.required_skills.score:.1f}/25  — {dim_scores.required_skills.detail}",
        f"  AI/GenAI Depth:     {dim_scores.ai_genai_depth.score:.1f}/20  — {dim_scores.ai_genai_depth.detail}",
        f"  Seniority Fit:      {dim_scores.seniority_fit.score:.1f}/10  — {dim_scores.seniority_fit.detail}",
        f"  Nice-to-Haves:      {dim_scores.nice_to_have.score:.1f}/5   — {dim_scores.nice_to_have.detail}",
        f"  Work Arrangement:   {dim_scores.work_arrangement.score:.1f}/15  — {dim_scores.work_arrangement.detail}",
        f"  Industry Alignment: {dim_scores.industry_alignment.score:.1f}/15  — {dim_scores.industry_alignment.detail}",
        f"  Compensation:       {dim_scores.compensation.score:.1f}/10  — {dim_scores.compensation.detail}",
    ])
    condition_line = f"Condition (why not Apply): {condition}" if condition else ""
    flags_line = f"Scoring flags: {', '.join(scoring_flags)}" if scoring_flags else ""

    user_content = f"""Job: {job_title} at {company}
Total score: {total_score:.1f}/100
Decision: {decision.upper()}
{condition_line}

Dimension breakdown:
{dim_summary}

{flags_line}

Write the 3-sentence reasoning now."""

    resp = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_REASONING,
        system=REASONING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    if resp.stop_reason == "max_tokens":
        raise FitScorerError("max_tokens", "reasoning call truncated")

    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_job(
    job_record: OkRecord,
    profile: CandidateProfile,
) -> FitScoreOutput:
    """
    One OkRecord + one CandidateProfile -> one FitScoreResult.

    The orchestrator runs this in a loop over many job records.
    The candidate profile is constant across all runs.

    Raises FitScorerError on model-level failures (orchestrator catches).
    """
    # --- step 1: input guards (before any LLM cost) ---
    if hasattr(job_record, "status") and str(job_record.status) in ("rejected", "Status.rejected"):
        return RejectedFitScore(
            reason=FitRejectReason.upstream_rejection,
            message="Job Analyst returned a rejected record — nothing to score.",
            job_id=getattr(job_record, "job_id", None),
        )

    if profile is None:
        return RejectedFitScore(
            reason=FitRejectReason.missing_profile,
            message="No candidate profile provided. Please run the Profile Builder first.",
        )

    # --- step 2: deterministic exact skill matching (code, no LLM) ---
    required   = job_record.required_skills.value or []
    nths       = job_record.nice_to_have_skills.value or []
    candidate_skills = list(profile.skills or []) + list(profile.ai_skills or [])

    matched,  missing  = _match_skills(required, candidate_skills)
    matched_nth, missing_nth = _match_skills(nths, candidate_skills)

    # --- step 3: LLM call 1 — judgments only ---
    judgments = _call_judgments(job_record, profile, missing)

    # --- step 4: code computes ALL scores (no arithmetic hallucination possible) ---
    req_score, req_detail = _score_required_skills(matched, len(required))
    ai_score,  ai_detail  = _score_ai_depth(judgments.ai_classification)
    sen_score, sen_detail = _score_seniority(judgments.seniority_match)
    nth_score, nth_detail = _score_nice_to_have(matched_nth, len(nths))
    arr_score, arr_detail = _score_work_arrangement(job_record)
    ind_score, ind_detail = _score_industry(judgments.industry_classification)
    com_score, com_detail = _score_compensation(job_record, profile.target_compensation)

    dim_scores = DimensionScores(
        required_skills    = DimensionScore(score=req_score, max=25.0, detail=req_detail),
        ai_genai_depth     = DimensionScore(score=ai_score,  max=20.0, detail=ai_detail),
        seniority_fit      = DimensionScore(score=sen_score, max=10.0, detail=sen_detail),
        nice_to_have       = DimensionScore(score=nth_score, max=5.0,  detail=nth_detail),
        work_arrangement   = DimensionScore(score=arr_score, max=15.0, detail=arr_detail),
        industry_alignment = DimensionScore(score=ind_score, max=15.0, detail=ind_detail),
        compensation       = DimensionScore(score=com_score, max=10.0, detail=com_detail),
    )

    # --- step 5: total + arithmetic self-verification ---
    raw_total = (
        req_score + ai_score + sen_score + nth_score +
        arr_score + ind_score + com_score
    )
    total_score = round(raw_total, 1)
    # Verify sum (cardinal sin check — should never fail since code computed it)
    recomputed = round(sum([
        dim_scores.required_skills.score,
        dim_scores.ai_genai_depth.score,
        dim_scores.seniority_fit.score,
        dim_scores.nice_to_have.score,
        dim_scores.work_arrangement.score,
        dim_scores.industry_alignment.score,
        dim_scores.compensation.score,
    ]), 1)
    if abs(total_score - recomputed) > 0.01:
        raise FitScorerError(
            "arithmetic_inconsistency",
            f"total_score {total_score} != sum of dimensions {recomputed}"
        )

    # --- step 6: decision + condition + flags ---
    decision  = _make_decision(total_score, req_score, ai_score)
    condition = _find_condition(dim_scores) if decision == "conditional_apply" else ""
    flags     = _compute_flags(job_record, judgments)

    # --- step 7: LLM call 2 — reasoning (sees actual scores) ---
    reasoning = _call_reasoning(
        job_title    = job_record.job_title.value or "Unknown Role",
        company      = job_record.company_name.value or "Unknown Company",
        total_score  = total_score,
        decision     = decision,
        condition    = condition,
        dim_scores   = dim_scores,
        scoring_flags= flags,
    )

    # --- step 8: assemble final result ---
    return FitScoreResult(
        job_id           = job_record.job_id,
        candidate_name   = profile.candidate_name,
        total_score      = total_score,
        decision         = decision,
        condition        = condition,
        reasoning        = reasoning,
        ai_classification= judgments.ai_classification.value,
        dimension_scores = dim_scores,
        matched_skills       = matched,
        missing_skills       = missing,
        near_match_skills    = judgments.near_match_skills,
        matched_nice_to_have = matched_nth,
        missing_nice_to_have = missing_nth,
        scoring_flags        = flags,
    )


# ---------------------------------------------------------------------------
# Demo — requires job_analyst.py and profile_builder.py in same directory
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # To run: build real objects from the other agents first, then call:
    #
    #   from job_analyst import parse_job
    #   from profile_builder import build_profile, PreferencesConfig, TargetComp
    #
    #   job = parse_job(jd_text, job_url)
    #   profile = build_profile("resume.pdf", PreferencesConfig(
    #       work_arrangement_preference="remote",
    #       target_compensation=TargetComp(min=120000, max=145000, currency="CAD")
    #   ))
    #
    #   if job.status == "ok" and profile.status == "ok":
    #       result = score_job(job, profile)
    #       print(result.model_dump_json(indent=2))
    #   else:
    #       print("Job or profile rejected:", job.status, profile.status)

    print("Fit Scorer loaded. Import score_job and call with an OkRecord + CandidateProfile.")
    print("See the comment block above for a full usage example.")
