"""
Resume Strategist — JobHunter AI Crew.

Translator, not inventor. Takes profile.yaml as grounded source of truth
and Fit Scorer output as targeting signal. Produces:
  - 3-5 targeted bullet rewrites (old → new → rationale → source)
  - A skills section modifier (add / remove / reorder / rename)

Every claim in the output is sourced to a real entry in profile.yaml.
Post-generation validator checks every number against the cited source —
the cardinal sin (invented metric) is architecturally caught, not hoped-away.

ARCHITECTURE — two LLM calls, code owns validation and section selection:
  Deterministic : load profile.yaml, select sections based on ai_classification,
                  build context, validate numbers post-generation.
  Call 1 (structured): LLM produces bullet rewrites sourced to profile entries.
  Call 2 (structured): LLM produces skills modifier operations.

Keep in the same directory as fit_scorer.py and job_analyst.py.

Setup:  pip install "anthropic>=0.40" pydantic>=2 pyyaml
        export ANTHROPIC_API_KEY=...

Usage:
    from resume_strategist import strategize
    from fit_scorer import score_job
    from job_analyst import parse_job

    job    = parse_job(jd_text, job_url)
    score  = score_job(job, profile)
    result = strategize(score, job, profile_path="profile.yaml")
    print(result.model_dump_json(indent=2))
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import re
from enum import Enum
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, ConfigDict
from anthropic import Anthropic

try:
    from fit_scorer import FitScoreResult
    from job_analyst import OkRecord
except ImportError as e:
    raise ImportError(
        f"Could not import from fit_scorer.py or job_analyst.py. "
        f"Keep all files in the same directory. Error: {e}"
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
MAX_TOKENS_REWRITES = 7000
MAX_TOKENS_SKILLS   = 4000

_client = Anthropic()

# Which profile.yaml section keys map to which AI classification
_AI_CORE_SECTIONS    = ["mytravelwallet", "mamamealbuddy_travelbuzz"]
_NON_AI_SECTIONS     = ["blueprint_pm"]
_ALL_BULLET_SECTIONS = ["mytravelwallet", "blueprint_pm", "mamamealbuddy_travelbuzz"]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RewriteWarningCode(str, Enum):
    no_source_found      = "no_source_found"
    gap_conflict         = "gap_conflict"
    soft_gap_caution     = "soft_gap_caution"
    vocabulary_divergence= "vocabulary_divergence"
    bullet_already_optimal = "bullet_already_optimal"
    hallucination_detected = "hallucination_detected"
    other                = "other"


# ---------------------------------------------------------------------------
# LLM response schemas (structured output contracts)
# Zero Optional/union types — all fields required with sentinels
# ---------------------------------------------------------------------------

class BulletRewrite(BaseModel):
    model_config = ConfigDict(extra="ignore")
    section: str = ""                   # e.g. "blueprint_pm"
    original: str                       # exact original bullet text
    rewritten: str                      # "" if deduplicated or optimal
    rationale: str
    source: list[str] = Field(default_factory=list)   # bullet-level paths
    jd_signals_addressed: list[str] = Field(default_factory=list)
    confidence: str = ""                # typed reason: "high — direct vocabulary match"
    hero_story_sourced: bool = False
    deduplicated: bool = False


class SectionNote(BaseModel):
    model_config = ConfigDict(extra="ignore")
    section: str
    reason: str


class BulletRewriteResponse(BaseModel):
    """What LLM Call 1 returns."""
    model_config = ConfigDict(extra="ignore")
    strategy_note: str = ""
    sections_touched: list[SectionNote] = Field(default_factory=list)
    sections_skipped: list[SectionNote] = Field(default_factory=list)
    bullet_rewrites: list[BulletRewrite]


class AddSkillOp(BaseModel):
    model_config = ConfigDict(extra="ignore")
    skill: str
    reason: str
    source: str = ""


class RemoveSkillOp(BaseModel):
    model_config = ConfigDict(extra="ignore")
    skill: str
    reason: str


class ReorderSkillOp(BaseModel):
    model_config = ConfigDict(extra="ignore")
    skill: str
    move_to: str = ""
    reason: str


class RenameSkillOp(BaseModel):
    model_config = ConfigDict(extra="ignore")
    current: str
    suggested: str
    reason: str


class SkillsModifierResponse(BaseModel):
    """What LLM Call 2 returns."""
    model_config = ConfigDict(extra="ignore")
    add: list[AddSkillOp] = Field(default_factory=list)
    remove: list[RemoveSkillOp] = Field(default_factory=list)
    reorder: list[ReorderSkillOp] = Field(default_factory=list)
    rename: list[RenameSkillOp] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Full output models
# ---------------------------------------------------------------------------

class RewriteWarning(BaseModel):
    code: RewriteWarningCode
    note: str = ""


class ResumeStrategyResult(BaseModel):
    status: str = "ok"
    job_id: Optional[str] = None
    decision_level: str
    condition_addressed: str = ""
    strategy_note: str
    sections_touched: list[SectionNote]
    sections_skipped: list[SectionNote]
    bullet_rewrites: list[BulletRewrite]
    skills_modifier: SkillsModifierResponse
    rewrite_warnings: list[RewriteWarning] = Field(default_factory=list)


class RejectedStrategy(BaseModel):
    status: str = "rejected"
    reason: str
    message: str
    job_id: Optional[str] = None


ResumeStrategyOutput = Union[ResumeStrategyResult, RejectedStrategy]


# ---------------------------------------------------------------------------
# Agent-level failure — raised to orchestrator
# ---------------------------------------------------------------------------

class ResumeStrategistError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def _load_profile(path: str) -> dict:
    """Load and parse profile.yaml. Raises ResumeStrategistError if missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ResumeStrategistError("missing_profile", f"profile.yaml not found at: {path}")
    except yaml.YAMLError as e:
        raise ResumeStrategistError("invalid_profile_yaml", str(e))


def _select_sections(
    ai_classification: str,
    decision: str,
    condition: str,
) -> tuple[list[str], list[str]]:
    """
    Deterministic section selection based on ai_classification and condition.
    Returns (primary_sections, secondary_sections).
    Primary sections get rewritten first; secondary are de-emphasized or skipped.
    """
    cond = (condition or "").lower()

    if ai_classification == "ai_core":
        primary   = ["mytravelwallet"]
        secondary = ["blueprint_pm", "mamamealbuddy_travelbuzz"]

    elif ai_classification == "non_ai":
        primary   = ["blueprint_pm"]
        secondary = ["mytravelwallet", "mamamealbuddy_travelbuzz"]

    else:  # ai_enabled — check condition to decide lead
        if any(x in cond for x in ["seniority", "level", "experience", "years"]):
            # Seniority stretch — lead with more tenured Blueprint
            primary   = ["blueprint_pm"]
            secondary = ["mytravelwallet"]
        elif any(x in cond for x in ["ai", "genai", "depth"]):
            # AI depth gap — lead with AI-heavy MTW
            primary   = ["mytravelwallet"]
            secondary = ["blueprint_pm"]
        else:
            primary   = ["mytravelwallet"]
            secondary = ["blueprint_pm"]

    return primary, secondary


def _extract_bullets_with_sources(profile: dict, sections: list[str]) -> list[dict]:
    """
    Extract bullets from the given sections with their source paths labeled.
    Returns a list of {section, company, title, type, text, source_path}.
    """
    results = []
    bullets_map = profile.get("bullets", {})

    for section_key in sections:
        section = bullets_map.get(section_key)
        if not section:
            continue
        company = section.get("company", "")
        title   = section.get("title", "")
        for i, b in enumerate(section.get("achievements", [])):
            results.append({
                "section": section_key,
                "company": company,
                "title": title,
                "type": "achievement",
                "text": b,
                "source_path": f"bullets.{section_key}.achievements[{i}]",
            })
        for i, b in enumerate(section.get("responsibilities", [])):
            results.append({
                "section": section_key,
                "company": company,
                "title": title,
                "type": "responsibility",
                "text": b,
                "source_path": f"bullets.{section_key}.responsibilities[{i}]",
            })
    return results


def _extract_hero_stories(profile: dict) -> list[dict]:
    """Extract hero stories (action + result + skills_demonstrated) with source paths."""
    results = []
    for i, story in enumerate(profile.get("hero_stories", [])):
        results.append({
            "title": story.get("title", ""),
            "action": story.get("action", ""),
            "result": story.get("result", ""),
            "skills_demonstrated": story.get("skills_demonstrated", []),
            "source_prefix": f"hero_stories[{i}]",
        })
    return results


def _extract_numbers(text: str) -> set[str]:
    """
    Extract all numeric values from text, normalized (commas removed).
    Handles: integers, decimals, percentages, ratios, currency amounts.
    e.g. "8/10", "400+", "85%", "$3,600", "3", "1.5M"
    """
    # Match digit sequences with optional separators and suffixes
    pattern = r'\d[\d,\.]*(?:[/\+](?:\d[\d,\.]*)?|[%KMBk])?'
    matches = re.findall(pattern, text)
    # Normalize: remove commas
    return {m.replace(",", "") for m in matches if m and re.search(r'\d', m)}


def _validate_rewrites(
    response: BulletRewriteResponse,
    profile: dict,
) -> tuple[BulletRewriteResponse, list[RewriteWarning]]:
    """
    Post-generation validator. Checks every number in every rewrite against
    its cited source in profile.yaml. Hallucinated numbers → rewrite rejected,
    hallucination_detected warning fired.

    Returns (validated_response, warnings).
    """
    warnings: list[RewriteWarning] = []
    clean_rewrites: list[BulletRewrite] = []

    # Build a flat lookup of all text in profile for source-path lookups
    source_text_lookup = _build_source_lookup(profile)

    for rewrite in response.bullet_rewrites:
        if not rewrite.rewritten or rewrite.deduplicated:
            clean_rewrites.append(rewrite)
            continue

        # Gather all source text for this rewrite
        all_source_text = ""
        for src_path in rewrite.source:
            all_source_text += " " + source_text_lookup.get(src_path, "")

        # If source paths didn't resolve, also search the raw original
        if not all_source_text.strip():
            all_source_text = rewrite.original

        # Extract numbers from rewrite and check against source
        rewrite_nums = _extract_numbers(rewrite.rewritten)
        source_nums  = _extract_numbers(all_source_text)

        # A rewrite number is "found" if it appears in any source number string
        hallucinated = []
        for num in rewrite_nums:
            found = any(
                num in s_num or s_num in num
                for s_num in source_nums
            )
            if not found:
                hallucinated.append(num)

        if hallucinated:
            # Reject this rewrite, fire hallucination_detected warning
            warnings.append(RewriteWarning(
                code=RewriteWarningCode.hallucination_detected,
                note=f"Section '{rewrite.section}': numbers {hallucinated} not found "
                     f"in source '{rewrite.source}'. Rewrite rejected.",
            ))
            # Keep the rewrite record but null out the rewritten text
            rewrite.rewritten = ""
            rewrite.rationale = (
                f"[REJECTED — hallucination_detected: {hallucinated} not in source] "
                + rewrite.rationale
            )
        clean_rewrites.append(rewrite)

    response.bullet_rewrites = clean_rewrites
    return response, warnings


def _build_source_lookup(profile: dict) -> dict[str, str]:
    """
    Build a flat {source_path: text} lookup for all bullets and hero stories.
    Used by the post-generation validator.
    """
    lookup: dict[str, str] = {}
    bullets_map = profile.get("bullets", {})

    for section_key, section in bullets_map.items():
        if not isinstance(section, dict):
            continue
        for i, b in enumerate(section.get("achievements", [])):
            lookup[f"bullets.{section_key}.achievements[{i}]"] = b
        for i, b in enumerate(section.get("responsibilities", [])):
            lookup[f"bullets.{section_key}.responsibilities[{i}]"] = b

    for i, story in enumerate(profile.get("hero_stories", [])):
        prefix = f"hero_stories[{i}]"
        lookup[f"{prefix}.action"]  = story.get("action", "")
        lookup[f"{prefix}.result"]  = story.get("result", "")

    return lookup


def _compute_warnings(
    response: BulletRewriteResponse,
    fit_score: FitScoreResult,
    profile: dict,
) -> list[RewriteWarning]:
    """
    Generate rewrite_warnings based on missing skills, gaps, and optimal bullets.
    Runs before the LLM call to surface known issues upfront.
    """
    warnings: list[RewriteWarning] = []
    hard_gaps = profile.get("gaps", {}).get("hard", [])
    soft_gaps = profile.get("gaps", {}).get("soft", [])

    # Flag missing skills that hit hard gaps
    for skill in (fit_score.missing_skills or []):
        for gap in hard_gaps:
            if skill.lower() in gap.lower() or gap.lower() in skill.lower():
                warnings.append(RewriteWarning(
                    code=RewriteWarningCode.gap_conflict,
                    note=f"Required skill '{skill}' conflicts with hard gap: '{gap}'"
                ))
                break
        else:
            for gap in soft_gaps:
                if skill.lower() in gap.lower() or gap.lower() in skill.lower():
                    warnings.append(RewriteWarning(
                        code=RewriteWarningCode.soft_gap_caution,
                        note=f"Required skill '{skill}' touches soft gap: '{gap}'. Review rewrite carefully."
                    ))
                    break

    return warnings


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

REWRITE_SYSTEM_PROMPT = """\
You are the Resume Strategist, a TRANSLATOR not an inventor.

Your job: take the candidate's existing bullets (with source paths) and
rewrite the most impactful ones to mirror the JD's language — while staying
100% grounded in what the candidate actually wrote.

CARDINAL SIN: inventing a metric, number, or achievement not in the provided
source material. ONE invented claim makes the entire output untrustworthy.
A candidate will use these rewrites in a real interview. If you invent
"increased revenue by 40%" and they get asked about it, their career is harmed.

ALWAYS:
- Produce EXACTLY 5 bullet_rewrites — no more, no fewer. Choose the 5 highest-impact
  bullets across all provided sections. If fewer than 5 honest rewrites exist, include
  the remainder with rewritten:"" and deduplicated:true or explain in rationale.
- Every rewrite must cite its source_path from the provided bullets/hero stories.
- Mirror the JD's exact vocabulary (from raw JD text) — not your paraphrase of it.
- Flag hero_story_sourced:true when any source is from hero_stories[n].
- Behave differently per decision level:
    apply            → polish strengths, minor vocabulary alignment
    conditional_apply → address the named condition directly
    maybe            → maximum honest case, mine every transferable signal

NEVER:
- Invent any number, metric, percentage, or dollar amount not in the source.
- Rewrite an achievement bullet as a responsibility or vice versa.
- Produce two rewrites for the same role that convey the same underlying achievement.
  If you would, mark the weaker one deduplicated:true and leave rewritten:"".
- Add skills or claims that conflict with the listed hard gaps.
- Exceed 5 bullet_rewrites under any circumstances.

CONFIDENCE format (typed reason, not bare label):
  "high — direct vocabulary match"
  "medium — near-match, not exact"
  "low — inferring from adjacent experience"
  "low — vocabulary divergence, review before using"

SOURCE paths format:
  "bullets.{section_key}.achievements[{index}]"
  "bullets.{section_key}.responsibilities[{index}]"
  "hero_stories[{index}].action"
  "hero_stories[{index}].result"

Multiple sources → source is a list.

strategy_note: write a plain-English paragraph explaining which sections you
touched, which you skipped, and what Fit Scorer signal drove the decisions.
Also note if profile.yaml has a reconciliation warning.
"""

SKILLS_MODIFIER_SYSTEM_PROMPT = """\
You are the Resume Strategist skills section optimizer.

Given the candidate's current skills sections, the JD's required/nice-to-have
skills, and the gap rules — produce four types of operations:
  add     → skills to add (must be evidenced in profile; check source)
  remove  → skills to de-emphasize or remove (not relevant to this JD)
  reorder → skills to move (e.g. move to top because JD mentions it 5 times)
  rename  → vocabulary translation (e.g. "Agentic AI workflows" → "Agentic systems")

NEVER add a skill that conflicts with the hard gaps listed.
NEVER invent a skill not evidenced in the profile skills sections.
Use the JD's raw vocabulary for rename suggestions — mirror exact phrasing.
Every operation must have a clear reason grounded in the JD signals or profile.

Return ONLY a JSON object matching this exact schema:
{
  "add":     [{"skill": str, "category": str, "reason": str}],
  "remove":  [{"skill": str, "reason": str}],
  "reorder": [{"skill": str, "move_to": str, "reason": str}],
  "rename":  [{"current": str, "suggested": str, "reason": str}]
}
The "rename" objects MUST include both "current" (existing skill label) and
"suggested" (replacement label using JD vocabulary). Omitting "suggested" is invalid.
"""


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _normalize_rewrite_response(data: dict) -> dict:
    """Coerce common LLM field-name variations into the BulletRewriteResponse schema."""
    data.setdefault("strategy_note", "")
    data.setdefault("sections_touched", [])
    data.setdefault("sections_skipped", [])
    for bullet in data.get("bullet_rewrites", []):
        # source_path (string) → source (list)
        if "source" not in bullet and "source_path" in bullet:
            sp = bullet.pop("source_path")
            bullet["source"] = [sp] if isinstance(sp, str) else list(sp)
        elif isinstance(bullet.get("source"), str):
            bullet["source"] = [bullet["source"]]
        bullet.setdefault("source", [])
        bullet.setdefault("jd_signals_addressed", [])
        bullet.setdefault("confidence", "")
        # infer section from source path when absent
        if not bullet.get("section") and bullet["source"]:
            parts = bullet["source"][0].split(".")
            if len(parts) >= 2 and parts[0] == "bullets":
                bullet["section"] = parts[1]
    return data


def _call_rewrites(
    context: dict,
    fit_score: FitScoreResult,
) -> BulletRewriteResponse:
    """Call 1: structured bullet rewrites."""
    user_content = json.dumps(context, ensure_ascii=False, default=str)

    import json as _json, re as _re
    _sys = REWRITE_SYSTEM_PROMPT + "\n\nRespond with ONLY valid JSON. No preamble, no markdown fences."
    resp = _client.messages.create(model=MODEL, max_tokens=MAX_TOKENS_REWRITES, system=_sys, messages=[{"role": "user", "content": user_content}])
    if resp.stop_reason == "max_tokens":
        raise ResumeStrategistError("max_tokens", "rewrites call truncated")
    _t = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    _t = _re.sub(r"^```(?:json)?\s*", "", _t); _t = _re.sub(r"\s*```$", "", _t).strip()
    try:
        return BulletRewriteResponse.model_validate(_normalize_rewrite_response(_json.loads(_t)))
    except Exception as e:
        raise ResumeStrategistError("parse_error", f"JSON parse failed: {e}")


def _normalize_skills_response(data: dict) -> dict:
    """Coerce common LLM field-name variations into the SkillsModifierResponse schema."""
    for op in data.get("reorder", []):
        # model uses: category / section / to / target instead of move_to
        if "move_to" not in op:
            for alias in ("category", "section", "to", "target", "destination"):
                if alias in op:
                    op["move_to"] = op.pop(alias)
                    break
        op.setdefault("move_to", "")
    valid_renames = []
    for op in data.get("rename", []):
        # model uses: old/from/original → current; new/to/replacement → suggested
        if "current" not in op:
            for alias in ("old", "from", "original", "existing"):
                if alias in op:
                    op["current"] = op.pop(alias)
                    break
        if "suggested" not in op:
            for alias in ("new", "to", "replacement", "updated", "new_name", "rename_to", "target", "label"):
                if alias in op:
                    op["suggested"] = op.pop(alias)
                    break
        if "suggested" in op:
            valid_renames.append(op)
    data["rename"] = valid_renames
    return data


def _call_skills_modifier(
    context: dict,
) -> SkillsModifierResponse:
    """Call 2: structured skills modifier."""
    user_content = json.dumps(context, ensure_ascii=False, default=str)

    import json as _json, re as _re
    _sys = SKILLS_MODIFIER_SYSTEM_PROMPT + "\n\nRespond with ONLY valid JSON. No preamble, no markdown fences."
    resp = _client.messages.create(model=MODEL, max_tokens=MAX_TOKENS_SKILLS, system=_sys, messages=[{"role": "user", "content": user_content}])
    if resp.stop_reason == "max_tokens":
        raise ResumeStrategistError("max_tokens", "skills call truncated")
    _t = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    _t = _re.sub(r"^```(?:json)?\s*", "", _t); _t = _re.sub(r"\s*```$", "", _t).strip()
    try:
        return SkillsModifierResponse.model_validate(_normalize_skills_response(_json.loads(_t)))
    except Exception as e:
        raise ResumeStrategistError("parse_error", f"JSON parse failed: {e}")


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _build_rewrite_context(
    fit_score: FitScoreResult,
    job_record: OkRecord,
    profile: dict,
    primary_sections: list[str],
    secondary_sections: list[str],
) -> dict:
    """Build the structured context JSON for the rewrite LLM call."""

    # Bullets with source paths (primary first, then secondary)
    primary_bullets   = _extract_bullets_with_sources(profile, primary_sections)
    secondary_bullets = _extract_bullets_with_sources(profile, secondary_sections)
    hero_stories      = _extract_hero_stories(profile)

    # JD signals (value for matching, raw for vocabulary)
    req_skills_value = job_record.required_skills.value or []
    req_skills_raw   = job_record.required_skills.raw or ""
    nth_skills_value = job_record.nice_to_have_skills.value or []
    nth_skills_raw   = job_record.nice_to_have_skills.raw or ""
    ai_signals       = job_record.ai_signals.value or []

    # Gaps
    hard_gaps = profile.get("gaps", {}).get("hard", [])
    soft_gaps = profile.get("gaps", {}).get("soft", [])

    # Reconciliation warning
    reconciliation_note = ""
    if "RECONCILIATION REQUIRED" in str(profile.get("_meta", "")):
        reconciliation_note = (
            "profile.yaml may differ from master resume — "
            "verify rewrites against master resume before using."
        )

    return {
        "task": {
            "decision_level": fit_score.decision,
            "condition_to_address": fit_score.condition or "",
            "ai_classification": fit_score.ai_classification,
            "primary_sections": primary_sections,
            "secondary_sections": secondary_sections,
        },
        "fit_signals": {
            "matched_skills":     fit_score.matched_skills or [],
            "missing_skills":     fit_score.missing_skills or [],
            "near_match_skills":  fit_score.near_match_skills or [],
        },
        "jd_signals": {
            "job_title":          job_record.job_title.value or "",
            "company":            job_record.company_name.value or "",
            "required_skills_normalized": req_skills_value,
            "required_skills_raw_jd_text": req_skills_raw,
            "nice_to_have_skills_normalized": nth_skills_value,
            "nice_to_have_skills_raw_jd_text": nth_skills_raw,
            "ai_signals_in_jd":   ai_signals,
        },
        "primary_bullets":   primary_bullets,
        "secondary_bullets": secondary_bullets,
        "hero_stories":      hero_stories,
        "gaps": {
            "hard": hard_gaps,
            "soft": soft_gaps,
        },
        "reconciliation_note": reconciliation_note,
    }


def _build_skills_context(
    fit_score: FitScoreResult,
    job_record: OkRecord,
    profile: dict,
) -> dict:
    """Build the structured context JSON for the skills modifier LLM call."""
    return {
        "current_skills": profile.get("skills", {}),
        "jd_required_skills_normalized": job_record.required_skills.value or [],
        "jd_required_skills_raw":        job_record.required_skills.raw or "",
        "jd_nice_to_have_normalized":    job_record.nice_to_have_skills.value or [],
        "jd_nice_to_have_raw":           job_record.nice_to_have_skills.raw or "",
        "ai_signals_in_jd":              job_record.ai_signals.value or [],
        "ai_classification":             fit_score.ai_classification,
        "decision_level":                fit_score.decision,
        "matched_skills":                fit_score.matched_skills or [],
        "missing_skills":                fit_score.missing_skills or [],
        "hard_gaps":                     profile.get("gaps", {}).get("hard", []),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def strategize(
    fit_score: FitScoreResult,
    job_record: OkRecord,
    profile_path: str = "profile.yaml",
) -> ResumeStrategyOutput:
    """
    One FitScoreResult + one OkRecord + profile.yaml -> one ResumeStrategyResult.

    Raises ResumeStrategistError on model-level failures (orchestrator catches).

    Args:
        fit_score:    Output from score_job(). Must not be status:rejected.
        job_record:   Output from parse_job(). Must be an OkRecord.
        profile_path: Path to profile.yaml. Default: "profile.yaml".

    Returns:
        ResumeStrategyResult on success, RejectedStrategy on input rejection.
    """
    # --- step 1: input guards ---
    if str(getattr(fit_score, "status", "ok")) in ("rejected", "Status.rejected"):
        return RejectedStrategy(
            reason="upstream_rejection",
            message="Fit Scorer returned a rejected record — nothing to strategize.",
            job_id=getattr(fit_score, "job_id", None),
        )

    if fit_score.decision == "skip":
        return RejectedStrategy(
            reason="skip_filtered",
            message="Skip decisions do not receive resume strategy. Focus energy on Apply and Conditional Apply roles.",
            job_id=fit_score.job_id,
        )

    # --- step 2: load profile (deterministic) ---
    profile = _load_profile(profile_path)

    # --- step 3: select sections (deterministic) ---
    primary_sections, secondary_sections = _select_sections(
        ai_classification=fit_score.ai_classification,
        decision=fit_score.decision,
        condition=fit_score.condition or "",
    )

    # --- step 4: pre-call warnings (gap conflicts, known issues) ---
    pre_warnings = _compute_warnings(
        response=None,  # type: ignore
        fit_score=fit_score,
        profile=profile,
    )

    # --- step 5: LLM call 1 — bullet rewrites ---
    rewrite_context = _build_rewrite_context(
        fit_score, job_record, profile, primary_sections, secondary_sections
    )
    rewrite_response = _call_rewrites(rewrite_context, fit_score)

    # --- step 6: post-generation validation (cardinal sin check) ---
    rewrite_response, validation_warnings = _validate_rewrites(rewrite_response, profile)

    # --- step 7: LLM call 2 — skills modifier ---
    skills_context  = _build_skills_context(fit_score, job_record, profile)
    skills_response = _call_skills_modifier(skills_context)

    # --- step 8: assemble final result ---
    all_warnings = pre_warnings + validation_warnings

    return ResumeStrategyResult(
        job_id            = fit_score.job_id,
        decision_level    = fit_score.decision,
        condition_addressed = fit_score.condition or "",
        strategy_note     = rewrite_response.strategy_note,
        sections_touched  = rewrite_response.sections_touched,
        sections_skipped  = rewrite_response.sections_skipped,
        bullet_rewrites   = rewrite_response.bullet_rewrites,
        skills_modifier   = skills_response,
        rewrite_warnings  = all_warnings,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Full usage example:
    #
    # from job_analyst import parse_job
    # from fit_scorer import score_job
    # from presenter import attach_display_meta
    #
    # jd_text = "...paste JD here..."
    # job_url = "https://www.linkedin.com/jobs/view/4409361708/"
    #
    # job     = parse_job(jd_text, job_url)
    # profile_input = build_profile("resume.pdf", prefs)   # from profile_builder.py
    # score   = score_job(job, profile_input)
    # result  = strategize(score, job, profile_path="profile.yaml")
    #
    # print(result.model_dump_json(indent=2))
    #
    # The output JSON gives you:
    #   - strategy_note       (why these sections, what the signal was)
    #   - bullet_rewrites     (old → new → rationale → source → confidence)
    #   - skills_modifier     (add / remove / reorder / rename with reasons)
    #   - rewrite_warnings    (gaps, hallucinations caught, optimal bullets)
    #
    # Every rewrite has a source path you can trace back to profile.yaml.
    # Every number has been validated against that source.

    # Quick profile load test (no API call needed)
    import os
    profile_file = "profile.yaml"
    if os.path.exists(profile_file):
        profile = _load_profile(profile_file)
        print(f"profile.yaml loaded. Sections found: {list(profile.get('bullets', {}).keys())}")
        print(f"Hard gaps: {len(profile.get('gaps', {}).get('hard', []))}")
        print(f"Hero stories: {len(profile.get('hero_stories', []))}")
        print("Resume Strategist ready. Call strategize(score, job, profile_path) to run.")
    else:
        print("profile.yaml not found in current directory.")
        print("Place profile.yaml alongside this file and run again.")
