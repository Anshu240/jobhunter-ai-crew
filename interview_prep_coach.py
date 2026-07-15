"""
Interview Prep Coach — JobHunter AI Crew.

Takes a structured JD, company brief, and the candidate's real hero stories,
maps each predicted question to the specific story that answers it best, and
produces a ready-to-use prep pack covering behavioral, product sense, metrics,
AI-specific, technical, and executive questions.

CARDINAL SIN: a model answer containing any detail not in profile.yaml
or knowledge_base.yaml. One invented claim at a CEO/CTO round ends the
interview. This agent never fabricates experience.

ARCHITECTURE:
  - Load profile.yaml and knowledge_base.yaml (human-maintained)
  - Infer interviewer levels from JD or accept human override
  - Build structured context for one LLM call
  - One LLM call → InterviewPrepPack (structured output)
  - Post-generation validation: story source paths exist, no orphaned answers
  - Return result

Setup:  pip install "anthropic>=0.40" pydantic>=2 pyyaml
        export ANTHROPIC_API_KEY=...

Usage:
    from interview_prep_coach import prep_interview
    from job_analyst import parse_job
    from fit_scorer import score_job
    from company_researcher import research_company

    job   = parse_job(jd_text, job_url)
    score = score_job(job, candidate_profile)
    brief = research_company(company_name=job.company_name.value)

    pack = prep_interview(
        job_record=job,
        fit_score=score,
        company_brief=brief,
        interviewer_levels=["pm", "director"],
        profile_path="profile.yaml",
        knowledge_base_path="knowledge_base.yaml",
    )
    print(pack.model_dump_json(indent=2))
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field
from anthropic import Anthropic

try:
    from job_analyst import OkRecord
except ImportError:
    OkRecord = None  # type: ignore

try:
    from fit_scorer import FitScoreResult
except ImportError:
    FitScoreResult = None  # type: ignore

try:
    from company_researcher import CompanyBrief
except ImportError:
    CompanyBrief = None  # type: ignore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
MAX_TOKENS_SELECTOR  = 1500   # Call 1: selection only
MAX_TOKENS_GENERATOR = 16000  # Call 2: writing answers for selected questions only

_client = Anthropic()

# Interviewer level keywords to detect from JD text
_DIRECTOR_SIGNALS = ["director", "vp of product", "head of product", "principal pm"]
_CPO_SIGNALS      = ["chief product officer", "cpo", "vp product"]
_CEO_SIGNALS      = ["chief executive", "ceo", "founder", "co-founder"]
_CTO_SIGNALS      = ["chief technology", "cto", "chief architect"]


# ---------------------------------------------------------------------------
# Output schema — zero Optional / union types throughout
# All optional fields default to "" or []
# ---------------------------------------------------------------------------

class ModelAnswer(BaseModel):
    situation:    str = ""
    action:       str = ""
    result:       str = ""
    demonstrates: str = ""


class PrepQuestion(BaseModel):
    id:                     str
    question:               str
    category:               str       # behavioral, behavioral_ai, product_sense, metrics, executive
    likelihood:             str       # high, likely, possible
    interviewer_level:      str = "pm" # pm, director, vp, ceo, all
    jd_signal:              str = ""
    framework:              str = ""
    recommended_story:      str = ""  # source path e.g. "hero_stories[0]"
    alternative_story:      str = ""  # second story source path or ""
    story_missing:          bool = False
    answer_type:            str       # story_grounded, framework_applied, concept_explained, bridge_answer
    model_answer:           ModelAnswer
    framework_hint:         str = ""
    coaching_note:          str       # mandatory — never blank
    time_budget_minutes:    str = ""  # e.g. "2-3 min", "15-25 min"
    anticipated_followups:  list[str] = Field(default_factory=list)


class TechnicalQuestion(BaseModel):
    id:                    str
    question:              str
    likelihood:            str       # high, likely, possible
    gap_type:              str       # none, soft_gap, hard_gap
    answer_type:           str       # story_grounded, concept_explained, bridge_answer
    your_experience:       str = ""  # source path or ""
    concept_framework:     str = ""
    bridge_answer:         str = ""
    coaching_note:         str
    time_budget_minutes:   str = ""
    anticipated_followups: list[str] = Field(default_factory=list)


class ExecutiveQuestion(BaseModel):
    id:               str
    question:         str
    answer_type:      str
    coaching_note:    str
    answer_structure: str = ""


class ExecutivePrep(BaseModel):
    director: list[ExecutiveQuestion] = Field(default_factory=list)
    vp_cpo:   list[ExecutiveQuestion] = Field(default_factory=list)
    ceo:      list[ExecutiveQuestion] = Field(default_factory=list)


class InterviewPrepPack(BaseModel):
    """Full output of the Interview Prep Coach."""
    status:                    str = "ok"
    job_title:                 str
    company:                   str
    decision_level:            str = ""
    interviewer_levels_detected: list[str]
    prep_warnings:             list[str] = Field(default_factory=list)
    estimated_prep_time_hours: int = 3
    must_prepare_first:        list[str] = Field(default_factory=list)
    question_bank:             list[PrepQuestion]
    technical_questions:       list[TechnicalQuestion] = Field(default_factory=list)
    executive_prep:            ExecutivePrep = Field(default_factory=ExecutivePrep)
    company_specific_questions: list[PrepQuestion] = Field(default_factory=list)
    smart_questions_to_ask:    list[str] = Field(default_factory=list)


class RejectedPrepPack(BaseModel):
    status:  str = "rejected"
    reason:  str
    message: str


PrepPackOutput = Union[InterviewPrepPack, RejectedPrepPack]
class SelectorOutput(BaseModel):
    """Output of Call 1 — selector picks what to include, generator writes it."""
    selected_question_ids: list[str] = Field(default_factory=list)
    selected_tech_concepts: list[str] = Field(default_factory=list)
    selected_exec_levels:   list[str] = Field(default_factory=list)
    must_prepare_first:     list[str] = Field(default_factory=list)
    prep_warnings:          list[str] = Field(default_factory=list)
    estimated_prep_time_hours: int = 3
    company_specific_angle: str = ""





# ---------------------------------------------------------------------------
# Agent-level failure
# ---------------------------------------------------------------------------

class InterviewPrepCoachError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        self.kind   = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

PREP_SYSTEM_PROMPT = """You are the Interview Prep Coach for a PM job search crew.

Your job: produce a specific, grounded, ready-to-use interview prep pack.
Every answer must trace to real candidate experience in the provided profile.

CARDINAL SIN: any model_answer containing a detail, metric, or achievement
not explicitly present in the candidate's profile.yaml or knowledge_base.yaml.
One fabricated claim at a CEO/CTO round can end the interview permanently.

STORY SELECTION PRIORITY (in strict order):
1. Explicit mappings in knowledge_base story_mappings → always first
2. Strength-based: STRONG > IN-FLIGHT > PROCESS-DEPTH
3. When two STRONG stories both fit → set BOTH recommended_story AND alternative_story.
   Explain the tradeoff in coaching_note. NEVER silently pick one.

FRAMEWORK SELECTION:
- AI PM questions → Dr. Nancy Li frameworks (primary)
- General PM questions → Lewis Lin DIGS / McDowell SPAR (primary)
- Always name the framework in the framework field (e.g. "MYCSPCHD", "behavioral_spar")

ANSWER TYPES:
- story_grounded:   answer uses a real hero story. Set recommended_story to path.
- framework_applied: case/product-sense question. Apply the named framework.
- concept_explained: technical concept question. Explain the concept.
- bridge_answer:    gap question. Honest bridge from adjacent experience.

EXECUTIVE ROUNDS (director / vp_cpo / ceo):
- Crisp strategic paragraphs ONLY — no long SPAR structures
- Answers must reference company brief signals if available
- CEO/CTO: conviction-led. One or two specific points, not a survey.
- coaching_note for executive questions must be concise and action-oriented

STORY FLAGS:
- IN-FLIGHT stories: coaching_note MUST include "process-strong, outcome-light — practice more"
- PROCESS-DEPTH stories: coaching_note MUST include "lead with decision quality, not outcome"

TECHNICAL ROUTING:
- gap_type none + experience exists → story_grounded (use your_experience source path)
- gap_type soft_gap → concept_explained + bridge_answer (show adjacent experience)
- gap_type hard_gap → bridge_answer only (be honest, never fabricate, flag clearly)

PREP WARNINGS — fire these when applicable:
jd_sparse, no_hero_stories, stories_not_mapped, missing_company_brief_for_senior_round,
hard_gap_detected, story_missing, other

MANDATORY FIELDS:
- coaching_note: never blank on any question. Always actionable.
- answer_type: always set.
- time_budget_minutes: set from question_category_time_budgets in knowledge base. Format: "X-Y min".
- anticipated_followups: 2-3 follow-up questions the interviewer is most likely to ask after this answer. These are the "second wave" questions. For behavioral questions, always include "How would you measure if that decision was right?" For product sense, include scope/tradeoff follow-ups. For AI questions, include reliability/ethics follow-ups.
- must_prepare_first: at least 3 question IDs. Rank by: high likelihood → gap questions → IN-FLIGHT stories.
- estimated_prep_time_hours: calculate from question count + complexity.

smart_questions_to_ask: generate 3-5 questions the candidate should ask the interviewer.
Must be specific to this company/role from the company brief. Not generic.
If no company brief, mark prep_warnings with missing_company_brief_for_senior_round if senior level.

QUESTION FRAMING — MANDATORY:
Every question in question_bank MUST be framed using the specific language from the JD VERBATIM section.
Do NOT write "Tell me about a time you built an AI product" — that is generic.
DO write "Stripe's JD says the AI specialist must know where to draw the line and when to hand off — walk me through how you designed that boundary in a product you shipped."
Use the company name, the role's specific responsibilities, and the JD's exact phrases.

MODEL ANSWERS — MANDATORY GROUNDING:
Every model_answer MUST reference a specific named hero story from the HERO STORIES section.
Use the story's KEY NUMBERS exactly as written — do not round or paraphrase them.
If a number appears in key_numbers, it MUST appear in the model answer.
If no story fits, set story_missing: true and write a framework-only answer — do not invent experience.

STRICT OUTPUT BUDGET:
question_bank: 6 max. technical_questions: exactly 3. executive_prep.director: exactly 2. vp_cpo: 1 or []. ceo: [].

TECHNICAL — MANDATORY: Populate technical_questions with exactly 3 questions. AI roles always have technical questions.
EXECUTIVE — MANDATORY: Populate executive_prep.director with exactly 2 strategic questions using company brief.
"""



SELECTOR_SYSTEM_PROMPT = """\
You are a precise interview prep selector for a job candidate.

Given a JD, fit score, and list of available question IDs, your job is ONLY to:
1. Pick the 6 most relevant question IDs from the available list
2. Pick 3 technical concept keys to cover
3. Identify which executive levels to prep for
4. Flag key prep warnings

RULES:
- selected_question_ids: exactly 6 IDs from the AVAILABLE QUESTION IDs list
- selected_tech_concepts: exactly 3 keys from AVAILABLE TECH CONCEPTS list
- selected_exec_levels: ["director"] for most roles, add "vp_cpo" if senior
- must_prepare_first: top 3 question IDs the candidate must nail
- prep_warnings: honest gaps (revenue ownership, seniority stretch, etc.)
- company_specific_angle: one sentence on what this company specifically cares about

Respond with ONLY a valid JSON object. No preamble, no markdown fences.
"""

GENERATOR_SYSTEM_PROMPT = """\
You are an expert interview prep coach writing personalized answers for a job candidate.

You have been given EXACTLY the questions to answer — do not add or remove questions.
Write full, specific, story-grounded answers for each selected question.

RULES:
- Every model_answer MUST reference a specific named hero story
- Use the story's KEY NUMBERS exactly as written — never round or paraphrase
- Frame every question using the JD KEY LANGUAGE and COMPANY ANGLE provided
- story_missing: true only if no story fits — never invent experience
- coaching_note: one specific, actionable note per question
- anticipated_followups: 2-3 follow-up questions the interviewer will likely ask

TECHNICAL QUESTIONS:
Write exactly 3 technical questions using the TECH CONCEPTS provided.
Route by gap_type: none = story_grounded, soft_gap = bridge, hard_gap = honest bridge.

EXECUTIVE PREP:
Write 2 director questions and 1 vp_cpo question (conviction-led, strategic).
Use the COMPANY ANGLE — these questions should be specific to this company.

OUTPUT BUDGET:
question_bank: exactly 6 questions (the ones in SELECTED QUESTIONS)
technical_questions: exactly 3
executive_prep.director: exactly 2
executive_prep.vp_cpo: exactly 1
executive_prep.ceo: []
smart_questions_to_ask: 4-5 questions to ask the interviewer

Respond with ONLY a valid JSON object. No preamble, no markdown fences.
"""

# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------

def _load_yaml(path: str, label: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise InterviewPrepCoachError(f"missing_{label}", f"{label} not found at: {path}")
    except yaml.YAMLError as e:
        raise InterviewPrepCoachError(f"invalid_{label}_yaml", str(e))


# ---------------------------------------------------------------------------
# Interviewer level inference
# ---------------------------------------------------------------------------

def _infer_interviewer_levels(
    job_record,
    override: list[str],
) -> list[str]:
    """Infer interviewer levels from JD text. Human override takes precedence."""
    if override:
        return [l.lower().strip() for l in override]

    levels = {"pm"}  # always include base PM level

    # Try to extract JD text from job_record fields
    jd_text = ""
    if job_record is not None:
        for field in ["job_title", "company_name", "required_skills", "nice_to_have_skills"]:
            attr = getattr(job_record, field, None)
            if attr:
                val = getattr(attr, "raw", "") or getattr(attr, "value", "") or ""
                if isinstance(val, list):
                    val = " ".join(str(v) for v in val)
                jd_text += " " + str(val)
        jd_text = jd_text.lower()

    for signal in _DIRECTOR_SIGNALS:
        if signal in jd_text:
            levels.add("director")
    for signal in _CPO_SIGNALS:
        if signal in jd_text:
            levels.add("vp_cpo")
    for signal in _CEO_SIGNALS:
        if signal in jd_text:
            levels.add("ceo")
    for signal in _CTO_SIGNALS:
        if signal in jd_text:
            levels.add("ceo")  # CTO prep maps to ceo section

    return sorted(levels)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_story_lookup(profile: dict, kb: dict) -> dict[str, int]:
    """
    Build a {story_id: profile_index} lookup using title matching.
    Fails loudly if a story_registry entry can't be matched — catches index drift early.
    """
    stories = profile.get("hero_stories", [])
    registry = kb.get("story_registry", {})
    lookup: dict[str, int] = {}
    unmatched: list[str] = []

    for story_id, meta in registry.items():
        title_match = meta.get("title_match", "")
        found = False
        for i, story in enumerate(stories):
            if title_match.lower() in story.get("title", "").lower() or \
               story.get("title", "").lower() in title_match.lower():
                lookup[story_id] = i
                found = True
                break
        if not found:
            unmatched.append(f"{story_id} → '{title_match}'")

    if unmatched:
        raise InterviewPrepCoachError(
            "story_registry_mismatch",
            f"These story IDs in knowledge_base.yaml don't match any hero story title in profile.yaml: "
            f"{unmatched}. Update story_registry title_match values or check profile.yaml story titles."
        )

    return lookup


def _resolve_story_ref(story_ref: str, lookup: dict[str, int]) -> str:
    """Convert a named story ID to a hero_stories[N] path for the output schema."""
    if story_ref.startswith("story:"):
        idx = lookup.get(story_ref)
        if idx is not None:
            return f"hero_stories[{idx}]"
        return ""  # unresolvable — validator will catch it
    return story_ref  # already a raw path — pass through



# ---------------------------------------------------------------------------
# Two-call context builders
# ---------------------------------------------------------------------------

def _build_selector_context(job_record, fit_score, profile: dict, kb: dict) -> str:
    """Lightweight context for Call 1 — IDs and summaries only, no full text."""
    lines = []

    # JD summary
    jt       = getattr(getattr(job_record, "job_title",    None), "value", "") if job_record else ""
    cn       = getattr(getattr(job_record, "company_name", None), "value", "") if job_record else ""
    req      = getattr(getattr(job_record, "required_skills", None), "value", []) or []
    arr      = getattr(getattr(job_record, "work_arrangement", None), "value", None)
    arr_val  = getattr(arr, "value", str(arr)) if arr else ""
    req_raw  = getattr(getattr(job_record, "required_skills", None), "raw", "") or ""

    decision  = getattr(fit_score, "decision",  "") if fit_score else ""
    condition = getattr(fit_score, "condition", "") if fit_score else ""
    missing   = getattr(fit_score, "missing_skills", []) or []
    ai_class  = getattr(fit_score, "ai_classification", "") if fit_score else ""

    lines.append(f"JOB: {jt} @ {cn}")
    lines.append(f"AI CLASSIFICATION: {ai_class}")
    lines.append(f"DECISION: {decision.upper()} | CONDITION: {condition}")
    lines.append(f"REQUIRED: {', '.join(str(s) for s in req[:10])}")
    lines.append(f"MISSING: {', '.join(str(s) for s in missing[:5])}")
    lines.append(f"WORK: {arr_val}")
    if req_raw:
        lines.append(f"JD KEY LANGUAGE: {req_raw[:200]}")

    # Available question IDs (text truncated to help selection)
    qbank = kb.get("question_bank", {})
    lines.append("\nAVAILABLE QUESTION IDs — pick 6 most relevant:")
    for q in qbank.get("ai_specific", []):
        lines.append(f"  {q['id']}: {q.get('question','')[:90]}")
    for q in qbank.get("general_pm", []):
        lines.append(f"  {q['id']}: {q.get('question','')[:90]}")
    crew = kb.get("crewai_portfolio_questions", {})
    for q in crew.get("architecture_questions", [])[:3]:
        qid = q.get("id", f"CREWAI-{q.get('question','')[:10]}")
        lines.append(f"  {qid}: {q.get('question','')[:90]}")

    # Story IDs
    stories = profile.get("hero_stories", [])
    lines.append("\nAVAILABLE STORIES:")
    for i, s in enumerate(stories):
        lines.append(f"  hero_stories[{i}]: {s.get('title','')[:70]}")

    # Tech concept keys
    tech = kb.get("technical_concepts", {})
    lines.append("\nAVAILABLE TECH CONCEPTS — pick 3:")
    for key, c in tech.items():
        lines.append(f"  {key}: gap={c.get('gap_type','none')}")

    return "\n".join(lines)


def _build_generator_context(
    selection: "SelectorOutput",
    profile: dict,
    kb: dict,
    job_record,
    company_brief,
    interviewer_levels: list[str],
) -> str:
    """Full-detail context for Call 2 — only selected items, complete text."""
    lines = []

    # Job / company signals
    jt      = getattr(getattr(job_record, "job_title",    None), "value", "") if job_record else ""
    cn      = getattr(getattr(job_record, "company_name", None), "value", "") if job_record else ""
    req_raw = getattr(getattr(job_record, "required_skills", None), "raw", "") or ""

    lines.append(f"JOB: {jt} @ {cn}")
    if req_raw:
        lines.append(f"JD KEY LANGUAGE: {req_raw[:250]}")
    if selection.company_specific_angle:
        lines.append(f"COMPANY ANGLE: {selection.company_specific_angle}")

    # Company brief signals
    if company_brief:
        ai_class = getattr(getattr(company_brief, "ai_signal", None), "classification", "")
        pts      = getattr(company_brief, "pm_talking_points", []) or []
        lines.append(f"AI SIGNAL: {ai_class}")
        if pts:
            lines.append(f"PM TALKING POINTS: {'; '.join(str(p) for p in pts[:2])}")

    # Selected questions with full text
    qbank    = kb.get("question_bank", {})
    crew     = kb.get("crewai_portfolio_questions", {})
    all_qs   = (qbank.get("ai_specific", []) + qbank.get("general_pm", []) +
                crew.get("architecture_questions", []) + crew.get("pm_judgment_questions", []))
    q_by_id  = {}
    for q in all_qs:
        qid = q.get("id") or q.get("question_id")
        if qid:
            q_by_id[qid] = q

    lines.append("\n=== SELECTED QUESTIONS — write answers for ALL of these ===")
    for qid in selection.selected_question_ids:
        q = q_by_id.get(qid)
        if q:
            lines.append(f"\n[{qid}] {q.get('question','')}")
            if q.get("framework"):
                lines.append(f"  Framework: {q['framework']}")
            if q.get("maps_to_profile"):
                lines.append(f"  Maps to: {q['maps_to_profile']}")
            if q.get("coaching_note"):
                lines.append(f"  Coaching hint: {q['coaching_note'][:120]}")

    # Hero stories — full detail
    stories = profile.get("hero_stories", [])
    lines.append("\n=== HERO STORIES — ground model answers in these ===")
    for i, s in enumerate(stories[:4]):
        title    = s.get("title", f"Story {i}")
        lines.append(f"\n[hero_stories[{i}]] — {title}")
        lines.append(f"  ACTION: {s.get('action','')[:350]}")
        lines.append(f"  RESULT: {s.get('result','')[:250]}")
        if s.get("key_numbers"):
            lines.append(f"  KEY NUMBERS (use these exactly): {', '.join(s['key_numbers'][:5])}")
        if s.get("honest_framing"):
            lines.append(f"  HONEST FRAMING: {s['honest_framing']}")

    # Selected tech concepts — full detail
    tech = kb.get("technical_concepts", {})
    lines.append("\n=== SELECTED TECH CONCEPTS — write technical_questions for these ===")
    for key in selection.selected_tech_concepts:
        c = tech.get(key, {})
        if c:
            lines.append(f"\n[{key}] gap={c.get('gap_type','none')}")
            lines.append(f"  What: {c.get('what_it_is','')[:150]}")
            if c.get("your_experience"):
                lines.append(f"  Your experience: {c['your_experience'][:150]}")
            if c.get("bridge_answer"):
                lines.append(f"  Bridge: {c['bridge_answer'][:150]}")

    # Prep warnings to include
    if selection.prep_warnings:
        lines.append(f"\nPREP WARNINGS (include in prep_warnings field): {'; '.join(selection.prep_warnings)}")

    # Must prepare
    if selection.must_prepare_first:
        lines.append(f"MUST PREPARE FIRST: {', '.join(selection.must_prepare_first)}")

    lines.append(f"ESTIMATED PREP HOURS: {selection.estimated_prep_time_hours}")

    return "\n".join(lines)


def _extract_hero_stories(profile: dict) -> str:
    """Format hero stories for LLM context — action + result + skills only."""
    stories = profile.get("hero_stories", [])
    lines = ["HERO STORIES (use ONLY these for behavioral model answers):"]
    for i, s in enumerate(stories[:3]):
        title    = s.get("title", f"Story {i}")
        notes    = s.get("notes", [])
        strength = "STRONG"
        if isinstance(notes, list):
            for n in notes:
                if "IN-FLIGHT" in str(n) or "in-flight" in str(n).lower():
                    strength = "IN-FLIGHT"
                    break
                if "PROCESS-DEPTH" in str(n) or "process-depth" in str(n).lower():
                    strength = "PROCESS-DEPTH"
                    break
        # Infer strength from title markers too
        if "IN-FLIGHT" in title.upper() or "(Decision in Flight)" in title:
            strength = "IN-FLIGHT"
        if "PROCESS-DEPTH" in title.upper():
            strength = "PROCESS-DEPTH"

        lines.append(f"\n[hero_stories[{i}]] ({strength}) — {title}")
        lines.append(f"  ACTION: {s.get('action', '')[:400]}")
        lines.append(f"  RESULT: {s.get('result', '')[:300]}")
        lines.append(f"  SKILLS: {', '.join(s.get('skills_demonstrated', []))}")
        if s.get("key_numbers"):
            lines.append(f"  KEY NUMBERS (use these exactly): {', '.join(s['key_numbers'][:5])}")
        if s.get("honest_framing"):
            lines.append(f"  HONEST FRAMING: {s['honest_framing']}")
    return "\n".join(lines)


def _extract_bullets_summary(profile: dict) -> str:
    """Extract key bullet achievements for technical question grounding."""
    bullets = profile.get("bullets", {})
    lines = ["KEY BULLETS (for technical answer grounding):"]
    for section_key, section in bullets.items():
        if not isinstance(section, dict):
            continue
        company = section.get("company", section_key)
        lines.append(f"\n[bullets.{section_key}] — {company}")
        for i, a in enumerate(section.get("achievements", [])[:3]):
            lines.append(f"  achievements[{i}]: {a[:200]}")
        for i, r in enumerate(section.get("responsibilities", [])[:4]):
            lines.append(f"  responsibilities[{i}]: {r[:200]}")
    return "\n".join(lines)


def _extract_products(profile: dict) -> str:
    """Extract products shipped for technical answers."""
    products = profile.get("products_shipped", [])
    lines = ["PRODUCTS SHIPPED:"]
    for p in products:
        lines.append(f"\n[{p.get('name', '')}] ({p.get('status', '')})")
        lines.append(f"  Stack: {', '.join(p.get('stack', []))}")
        for kd in p.get("key_decisions", [])[:3]:
            lines.append(f"  Decision: {kd}")
        for m in p.get("metrics", [])[:3]:
            lines.append(f"  Metric: {m}")
    return "\n".join(lines)


def _extract_gaps(profile: dict) -> str:
    """Format gaps as explicit rules."""
    gaps = profile.get("gaps", {})
    lines = ["GAPS — STRICT RULES:"]
    lines.append("HARD GAPS (NEVER suggest the candidate has experience in these):")
    for g in gaps.get("hard", []):
        lines.append(f"  - {g}")
    lines.append("SOFT GAPS (be careful, use adjacent experience + honest bridge):")
    for g in gaps.get("soft", []):
        lines.append(f"  - {g}")
    return "\n".join(lines)


def _extract_knowledge_base_context(
    kb: dict,
    ai_classification: str,
    required_skills: list[str],
    interviewer_levels: list[str],
) -> str:
    """Extract trimmed knowledge base context — essentials only to stay within token limits."""
    lines = []

    # Frameworks — names and use_when only (no full structures)
    lines.append("=== FRAMEWORKS ===")
    frameworks = kb.get("frameworks", {})
    framework_keys = ["behavioral_spar", "digs_method"]
    if ai_classification in ("ai_core", "ai_enabled"):
        framework_keys += ["ai_product_lifecycle", "ai_product_design", "metrics_mycspchd"]
    else:
        framework_keys += ["product_sense", "metrics_mycspchd"]
    for key in framework_keys:
        if key in frameworks:
            f = frameworks[key]
            lines.append(f"[{key}] {f.get('name', key)} — use when: {f.get('use_when', '')[:120]}")

    # Question bank — IDs and questions only, no coaching notes
    lines.append("\n=== QUESTION BANK (pick most relevant for this JD) ===")
    qbank = kb.get("question_bank", {})
    if ai_classification in ("ai_core", "ai_enabled"):
        lines.append("AI-SPECIFIC:")
        for q in qbank.get("ai_specific", [])[:10]:
            lines.append(f"  [{q['id']}] {q['question'][:120]} | framework:{q.get('framework','')}")
    lines.append("GENERAL PM:")
    for q in qbank.get("general_pm", [])[:8]:
        lines.append(f"  [{q['id']}] {q['question'][:120]} | framework:{q.get('framework','')}")

    # Story registry — IDs, strength, best_for only
    lines.append("\n=== STORY REGISTRY (check FIRST before selecting any story) ===")
    for story_id, meta in kb.get("story_registry", {}).items():
        lines.append(f"[{story_id}] {meta.get('title_match','')[:60]} | best_for: {', '.join(meta.get('best_for',[]))[:120]}")

    # Technical concepts — 6 most relevant, brief format
    lines.append("\n=== TECHNICAL CONCEPTS ===")
    tech = kb.get("technical_concepts", {})
    required_lower = [s.lower() for s in required_skills]
    included = 0
    for concept_key, concept in tech.items():
        if included >= 6:
            break
        is_relevant = any(
            concept_key.lower() in skill or skill in concept_key.lower()
            for skill in required_lower
        ) or concept_key in ("rag_pipeline", "hallucination_guardrails", "agentic_ai")
        if is_relevant or ai_classification in ("ai_core", "ai_enabled"):
            lines.append(f"[{concept_key}] gap={concept.get('gap_type','none')} — {concept.get('what_it_is','')[:100]}")
            if concept.get("bridge_answer"):
                lines.append(f"  Bridge: {concept['bridge_answer'][:100]}")
            included += 1

    # CrewAI questions — questions only, no answers (saves tokens)
    if ai_classification in ("ai_core", "ai_enabled", "unknown"):
        crew_qs = kb.get("crewai_portfolio_questions", {})
        if crew_qs:
            lines.append("\n=== CREWAI PORTFOLIO QUESTIONS (include in technical_questions) ===")
            for q in crew_qs.get("architecture_questions", [])[:4]:
                lines.append(f"  - {q['question'][:100]}")
            for q in crew_qs.get("pm_judgment_questions", [])[:2]:
                lines.append(f"  - {q['question'][:100]}")

    # Company calibration
    lines.append("\n=== COMPANY CALIBRATION ===")
    for key, cal in kb.get("company_calibration", {}).items():
        lines.append(f"[{key}]: {', '.join(cal.get('rules', [])[:2])}")

    return "\n".join(lines)


def _build_context(
    job_record,
    fit_score,
    company_brief,
    profile: dict,
    kb: dict,
    interviewer_levels: list[str],
) -> str:
    """Build the full LLM context from all inputs."""

    lines = []

    # JD signals
    lines.append("=== JD SIGNALS ===")
    if job_record is not None:
        jt = getattr(getattr(job_record, "job_title", None), "value", "Unknown")
        cn = getattr(getattr(job_record, "company_name", None), "value", "Unknown")
        lines.append(f"Job Title: {jt}")
        lines.append(f"Company: {cn}")

        req = getattr(getattr(job_record, "required_skills", None), "value", []) or []
        nth = getattr(getattr(job_record, "nice_to_have_skills", None), "value", []) or []
        ai_sig = getattr(getattr(job_record, "ai_signals", None), "value", []) or []
        lines.append(f"Required Skills: {', '.join(str(s) for s in req[:15])}")
        lines.append(f"Nice to Have: {', '.join(str(s) for s in nth[:10])}")
        lines.append(f"AI Signals in JD: {', '.join(str(s) for s in ai_sig[:8])}")

        # Raw JD language — the actual phrases the interviewer cares about
        # Use raw fields to capture verbatim JD language for question framing
        req_raw = getattr(getattr(job_record, "required_skills", None), "raw", "") or ""
        nth_raw = getattr(getattr(job_record, "nice_to_have_skills", None), "raw", "") or ""
        loc_raw = getattr(getattr(job_record, "location", None), "raw", "") or ""
        if req_raw:
            lines.append(f"JD KEY LANGUAGE (use verbatim in questions): {req_raw[:250]}")
    else:
        lines.append("job_record: not provided")

    # Fit score signals
    ai_classification = "unknown"
    if fit_score is not None:
        ai_classification = getattr(fit_score, "ai_classification", "unknown") or "unknown"
        lines.append(f"AI Classification: {ai_classification}")
        lines.append(f"Decision Level: {getattr(fit_score, 'decision', '')}")
        lines.append(f"Condition: {getattr(fit_score, 'condition', '') or 'none'}")
        lines.append(f"Matched Skills: {', '.join(getattr(fit_score, 'matched_skills', []) or [])}")
        lines.append(f"Missing Skills: {', '.join(getattr(fit_score, 'missing_skills', []) or [])}")
        lines.append(f"Near Match: {', '.join(getattr(fit_score, 'near_match_skills', []) or [])}")
    else:
        lines.append("fit_score: not provided — using profile gaps for technical routing")

    # Candidate profile
    lines.append("\n=== CANDIDATE PROFILE ===")
    identity = profile.get("identity", {})
    lines.append(f"Name: {identity.get('name', 'Candidate')}")
    lines.append(f"Role: {identity.get('current_role', '')}")
    lines.append(f"Years PM: {identity.get('years_pm', '')}")
    lines.append(f"Location: {identity.get('location', '')}")

    lines.append("\n" + _extract_products(profile))
    lines.append("\n" + _extract_hero_stories(profile))
    lines.append("\n" + _extract_bullets_summary(profile))
    lines.append("\n" + _extract_gaps(profile))

    # Knowledge base
    req_skills_list = []
    if job_record is not None:
        req_skills_raw = getattr(getattr(job_record, "required_skills", None), "value", []) or []
        req_skills_list = [str(s) for s in req_skills_raw]

    lines.append("\n=== KNOWLEDGE BASE ===")
    lines.append(_extract_knowledge_base_context(kb, ai_classification, req_skills_list, interviewer_levels))

    # Company brief
    if company_brief is not None and hasattr(company_brief, "sections"):
        lines.append("\n=== COMPANY BRIEF ===")
        cn = getattr(company_brief, "company_name", "")
        lines.append(f"Company: {cn}")
        ai_sig = getattr(company_brief, "ai_signal", None)
        if ai_sig:
            lines.append(f"AI Classification: {getattr(ai_sig, 'classification', '')}")
            for ev in getattr(ai_sig, "evidence", [])[:3]:
                lines.append(f"  AI Evidence: {getattr(ev, 'signal', '')} [{getattr(ev, 'type', '')}]")
        sections = getattr(company_brief, "sections", None)
        if sections:
            hist = getattr(sections, "history", None)
            if hist and getattr(hist, "stated", False):
                lines.append(f"History: {getattr(hist, 'paragraph', '')[:200]}")
            culture = getattr(sections, "culture_and_people", None)
            if culture and getattr(culture, "stated", False):
                lines.append(f"Culture: {getattr(culture, 'paragraph', '')[:150]}")
            current = getattr(sections, "current_stance", None)
            if current and getattr(current, "stated", False):
                lines.append(f"Current Stance: {getattr(current, 'paragraph', '')[:200]}")
        pts = getattr(company_brief, "pm_talking_points", []) or []
        if pts:
            lines.append(f"PM Talking Points: {'; '.join(pts[:3])}")
        sq = getattr(company_brief, "smart_questions", []) or []
        if sq:
            lines.append(f"Smart Questions from Research: {'; '.join(sq[:3])}")
    else:
        lines.append("\n=== COMPANY BRIEF: not provided ===")

    # Interviewer levels
    lines.append(f"\n=== INTERVIEWER LEVELS: {', '.join(interviewer_levels)} ===")
    has_senior = any(l in interviewer_levels for l in ["director", "vp_cpo", "ceo"])
    if has_senior:
        lines.append("Executive prep section REQUIRED. Crisp strategic answers only — NO SPAR.")
    if has_senior and company_brief is None:
        lines.append("WARNING: Senior round but no company_brief. Fire missing_company_brief_for_senior_round.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _normalize_prep_pack(data: dict, job_title: str = "", company: str = "") -> dict:
    """
    Normalize LLM JSON output to match InterviewPrepPack schema exactly.
    Handles natural JSON formats the LLM returns without constrained decoding.
    """

    # --- Top-level required fields ---
    if "job_title" not in data or not data.get("job_title"):
        data["job_title"] = job_title or "Unknown"
    if "company" not in data or not data.get("company"):
        data["company"] = company or "Unknown"
    if "interviewer_levels_detected" not in data:
        data["interviewer_levels_detected"] = ["pm", "director"]

    # --- prep_warnings: list of strings ---
    warns = data.get("prep_warnings", [])
    if isinstance(warns, list):
        normalized = []
        for w in warns:
            if isinstance(w, str):
                normalized.append(w)
            elif isinstance(w, dict):
                for key in ["detail", "message", "warning", "text", "description"]:
                    if key in w and isinstance(w[key], str):
                        normalized.append(w[key])
                        break
                else:
                    normalized.append(w.get("type", "warning"))
        data["prep_warnings"] = normalized

    # --- smart_questions_to_ask: list of strings ---
    qs = data.get("smart_questions_to_ask", [])
    if isinstance(qs, list):
        normalized = []
        for q in qs:
            if isinstance(q, str):
                normalized.append(q)
            elif isinstance(q, dict):
                for key in ["question", "text", "value", "content"]:
                    if key in q and isinstance(q[key], str):
                        normalized.append(q[key])
                        break
        data["smart_questions_to_ask"] = normalized

    # --- must_prepare_first: list of strings ---
    mpf = data.get("must_prepare_first", [])
    if isinstance(mpf, list):
        data["must_prepare_first"] = [
            x if isinstance(x, str) else str(x.get("id", x.get("question_id", str(x))))
            for x in mpf if x
        ]

    # --- question_bank: find under any reasonable key name ---
    qb_keys = ["question_bank", "questions", "behavioral_questions",
                "prep_questions", "interview_questions", "bank"]
    if "question_bank" not in data or not isinstance(data.get("question_bank"), list):
        for key in qb_keys:
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                data["question_bank"] = data[key]
                break
        else:
            data["question_bank"] = []

    # --- technical_questions: find under any reasonable key name ---
    tq_keys = ["technical_questions", "technical", "tech_questions",
               "technical_prep", "coding_questions"]
    if "technical_questions" not in data or not isinstance(data.get("technical_questions"), list):
        for key in tq_keys:
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                data["technical_questions"] = data[key]
                break
        else:
            # Fallback: extract technical-category questions from question_bank
            qb = data.get("question_bank", [])
            tech_from_qb = [q for q in qb if isinstance(q, dict) and
                           q.get("category", "").lower() in ("technical", "tech")]
            data["technical_questions"] = tech_from_qb if tech_from_qb else []

    # --- executive_prep: find under any reasonable key name ---
    ep_keys = ["executive_prep", "executive", "exec_prep", "leadership_prep",
               "executive_questions", "exec_questions"]
    if "executive_prep" not in data or not isinstance(data.get("executive_prep"), dict):
        for key in ep_keys:
            if key in data and isinstance(data[key], dict):
                data["executive_prep"] = data[key]
                break
        # Handle case where LLM returns list of exec questions
        if "executive_prep" not in data or not isinstance(data.get("executive_prep"), dict):
            for key in ep_keys:
                if key in data and isinstance(data[key], list) and data[key]:
                    data["executive_prep"] = {"director": data[key], "vp_cpo": [], "ceo": []}
                    break

    # --- Normalize PrepQuestion objects ---
    def _norm_prep_question(q: dict) -> dict:
        if not isinstance(q, dict):
            return {"question": str(q), "id": "q0", "category": "behavioral",
                    "likelihood": "possible", "time_budget_minutes": "5-7 min",
                    "anticipated_followups": [], "story_missing": True}
        # Map question_id -> id (LLM often uses question_id instead of id)
        if "id" not in q and "question_id" in q:
            q["id"] = q["question_id"]
        # Map question_text / text / prompt -> question
        if "question" not in q:
            for key in ["question_text", "text", "prompt", "content", "q"]:
                if key in q and isinstance(q[key], str):
                    q["question"] = q[key]
                    break
            else:
                # Last resort: use the question_id value as display text
                q["question"] = q.get("id", q.get("question_id", "Unknown question"))
        # Ensure required fields
        if "id" not in q:
            q["id"] = f"q{abs(hash(q.get('question','')))%1000}"
        if "category" not in q:
            q["category"] = "behavioral"
        if "likelihood" not in q:
            q["likelihood"] = "possible"
        if "time_budget_minutes" not in q:
            q["time_budget_minutes"] = "5-7 min"
        if "anticipated_followups" not in q:
            q["anticipated_followups"] = []
        elif isinstance(q["anticipated_followups"], list):
            q["anticipated_followups"] = [
                f if isinstance(f, str)
                else f.get("question", f.get("text", str(f)))
                for f in q["anticipated_followups"]
            ]
        if not q.get("answer_type"):
            # Infer from story_missing or default to story_grounded
            q["answer_type"] = "story_grounded" if not q.get("story_missing") else "framework_applied"
        if "story_missing" not in q:
            q["story_missing"] = not bool(q.get("recommended_story"))
        # Convert None string fields to empty string
        for str_field in ["alternative_story", "recommended_story", "framework",
                          "framework_hint", "coaching_note"]:
            if str_field in q and q[str_field] is None:
                q[str_field] = ""
        # Normalize model_answer
        ma = q.get("model_answer")
        if isinstance(ma, str):
            q["model_answer"] = {"situation": ma, "action": "", "result": "", "demonstrates": ""}
        elif isinstance(ma, dict):
            for field in ["situation", "action", "result", "demonstrates"]:
                if field not in ma:
                    ma[field] = ""
        return q

    data["question_bank"] = [_norm_prep_question(q) for q in data["question_bank"]]

    # --- Normalize company_specific_questions ---
    cqs = data.get("company_specific_questions", [])
    if isinstance(cqs, list):
        data["company_specific_questions"] = [_norm_prep_question(q) for q in cqs]

    # --- Normalize technical_questions ---
    tech_qs = data.get("technical_questions", [])
    if isinstance(tech_qs, list):
        norm_tech = []
        for q in tech_qs:
            if not isinstance(q, dict):
                q = {"question": str(q)}
            # Map question_id -> id
            if "id" not in q and "question_id" in q:
                q["id"] = q["question_id"]
            if "id" not in q:
                q["id"] = f"tech{abs(hash(q.get('question', '')))%1000}"
            # Map question_text/text -> question
            if "question" not in q:
                for key in ["question_text", "text", "prompt", "content"]:
                    if key in q and isinstance(q[key], str):
                        q["question"] = q[key]
                        break
                else:
                    q["question"] = q.get("id", "Unknown technical question")
            if "likelihood" not in q:
                q["likelihood"] = "possible"
            if "gap_type" not in q:
                q["gap_type"] = "none"
            if not q.get("answer_type"):
                gap = q.get("gap_type", "none")
                q["answer_type"] = "story_grounded" if gap == "none" else ("bridge_answer" if gap == "hard_gap" else "concept_explained")
            if "anticipated_followups" not in q:
                q["anticipated_followups"] = []
            elif isinstance(q["anticipated_followups"], list):
                q["anticipated_followups"] = [
                    f if isinstance(f, str)
                    else f.get("question", f.get("text", str(f)))
                    for f in q["anticipated_followups"]
                ]
            norm_tech.append(q)
        data["technical_questions"] = norm_tech

    # --- Normalize executive_prep ---
    ep = data.get("executive_prep")
    if isinstance(ep, dict):
        for level in ["director", "vp_cpo", "ceo"]:
            qs = ep.get(level, [])
            if isinstance(qs, list):
                norm_qs = []
                for q in qs:
                    if isinstance(q, str):
                        norm_qs.append({"id": f"{level}_q", "question": q, "answer_structure": "", "coaching_note": ""})
                    elif isinstance(q, dict):
                        # Map question_id -> id
                        if "id" not in q or not q.get("id"):
                            q["id"] = q.get("question_id") or f"{level}_q{abs(hash(str(q)))%100}"
                        # Map question_text -> question
                        if "question" not in q or not q.get("question"):
                            for key in ["question_text", "text", "prompt", "content"]:
                                if key in q and isinstance(q[key], str) and q[key]:
                                    q["question"] = q[key]
                                    break
                            else:
                                q["question"] = str(q.get("id", "Unknown question"))
                        # Required fields with defaults
                        if not q.get("answer_type"):
                            q["answer_type"] = "framework_applied"
                        if not q.get("answer_structure"):
                            q["answer_structure"] = ""
                        if not q.get("coaching_note"):
                            q["coaching_note"] = ""
                        norm_qs.append(q)
                ep[level] = norm_qs
        data["executive_prep"] = ep

    # --- estimated_prep_time_hours: ensure number ---
    ept = data.get("estimated_prep_time_hours", 3)
    if isinstance(ept, str):
        import re as _re
        nums = _re.findall(r"\d+", ept)
        data["estimated_prep_time_hours"] = int(nums[0]) if nums else 3
    elif not isinstance(ept, (int, float)):
        data["estimated_prep_time_hours"] = 3

    return data


def _call_selector(context: str) -> "SelectorOutput":
    """Call 1 — pick which questions, stories, tech concepts to include."""
    import json, re as _re

    resp = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_SELECTOR,
        system=SELECTOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )

    if resp.stop_reason == "max_tokens":
        raise InterviewPrepCoachError("selector_truncated", "selector call truncated")

    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    text = _re.sub(r"^```(?:json)?\s*", "", text)
    text = _re.sub(r"\s*```$", "", text).strip()

    try:
        data = json.loads(text)
        # Normalize selector output
        if "selected_question_ids" not in data:
            data["selected_question_ids"] = []
        if "selected_tech_concepts" not in data:
            data["selected_tech_concepts"] = []
        if "selected_exec_levels" not in data:
            data["selected_exec_levels"] = ["director"]
        if "prep_warnings" not in data:
            # Check for various key names
            for key in ["warnings", "prep_warning", "flags"]:
                if key in data and isinstance(data[key], list):
                    data["prep_warnings"] = data[key]
                    break
            else:
                data["prep_warnings"] = []
        # Ensure prep_warnings are strings
        data["prep_warnings"] = [
            w if isinstance(w, str)
            else w.get("detail", w.get("message", str(w)))
            for w in data["prep_warnings"]
            if w
        ]
        return SelectorOutput.model_validate(data)
    except Exception as e:
        raise InterviewPrepCoachError("selector_parse_error", f"Selector JSON failed: {e}")


def _call_generator(context: str, job_title: str, company: str) -> InterviewPrepPack:
    """Call 2 — write full answers for selected questions only."""
    import json, re as _re

    resp = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_GENERATOR,
        system=GENERATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )

    if resp.stop_reason == "max_tokens":
        raise InterviewPrepCoachError("max_tokens", "generator call truncated — increase MAX_TOKENS_GENERATOR")

    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    text = _re.sub(r"^```(?:json)?\s*", "", text)
    text = _re.sub(r"\s*```$", "", text).strip()

    try:
        data = json.loads(text)
        data = _normalize_prep_pack(data, job_title=job_title, company=company)
        return InterviewPrepPack.model_validate(data)
    except Exception as e:
        raise InterviewPrepCoachError("parse_error", f"JSON parse/validation failed: {e}")


def _validate_pack(
    pack: InterviewPrepPack,
    profile: dict,
    kb: dict,
    levels: list[str],
    story_lookup: dict[str, int],
) -> InterviewPrepPack:
    """Resolve story: references and flag missing stories across all question fields."""
    for q in pack.question_bank:
        q.recommended_story = _resolve_story_ref(q.recommended_story, story_lookup)
        q.alternative_story  = _resolve_story_ref(q.alternative_story,  story_lookup)
        if q.answer_type == "story_grounded" and not q.recommended_story:
            q.story_missing = True

    for q in pack.technical_questions:
        q.your_experience = _resolve_story_ref(q.your_experience, story_lookup)

    for q in pack.company_specific_questions:
        q.recommended_story = _resolve_story_ref(q.recommended_story, story_lookup)
        q.alternative_story  = _resolve_story_ref(q.alternative_story,  story_lookup)
        if q.answer_type == "story_grounded" and not q.recommended_story:
            q.story_missing = True

    return pack


def _call_prep_coach(
    job_record,
    fit_score,
    company_brief,
    profile: dict,
    kb: dict,
    interviewer_levels: list[str],
    job_title: str = "",
    company: str = "",
) -> InterviewPrepPack:
    """
    Two-call architecture — permanently solves the max_tokens problem.

    Call 1 (Selector, ~1500 tokens): Picks which 6 questions, 3 tech concepts,
    and exec levels to include. Lightweight — just IDs and summaries.

    Call 2 (Generator, ~6000 tokens): Writes full model answers for ONLY the
    selected questions. Focused — no selection overhead, just writing.
    """
    # ── Call 1: Select ────────────────────────────────────────────────────────
    selector_context = _build_selector_context(job_record, fit_score, profile, kb)
    selection = _call_selector(selector_context)

    # ── Call 2: Generate ──────────────────────────────────────────────────────
    generator_context = _build_generator_context(
        selection=selection,
        profile=profile,
        kb=kb,
        job_record=job_record,
        company_brief=company_brief,
        interviewer_levels=interviewer_levels,
    )
    return _call_generator(generator_context, job_title=job_title, company=company)

def prep_interview(
    job_record=None,
    fit_score=None,
    company_brief=None,
    interviewer_levels: list[str] = None,
    profile_path: str = "profile.yaml",
    knowledge_base_path: str = "knowledge_base.yaml",
) -> PrepPackOutput:
    """
    Generate a complete interview prep pack for one job application.

    Args:
        job_record:           OkRecord from Job Analyst. Required.
        fit_score:            FitScoreResult from Fit Scorer. Optional.
        company_brief:        CompanyBrief from Company Researcher. Optional.
        interviewer_levels:   Override list e.g. ["pm", "director", "ceo"].
                              If None, inferred from job_record.
        profile_path:         Path to profile.yaml.
        knowledge_base_path:  Path to knowledge_base.yaml.

    Returns:
        InterviewPrepPack on success, RejectedPrepPack on input rejection.
        Raises InterviewPrepCoachError on model failures (caught by orchestrator).
    """
    # --- step 1: input guards ---
    if job_record is None:
        return RejectedPrepPack(
            reason="missing_job_record",
            message="job_record is required. Run Job Analyst first.",
        )

    # --- step 2: load YAML files ---
    profile = _load_yaml(profile_path, "profile")
    kb      = _load_yaml(knowledge_base_path, "knowledge_base")

    # --- step 3: determine interviewer levels ---
    levels = _infer_interviewer_levels(job_record, interviewer_levels or [])

    # --- step 3b: build story lookup (title-based — index-drift proof) ---
    # Fails loudly if story_registry in knowledge_base.yaml doesn't match profile.yaml titles
    story_lookup = _build_story_lookup(profile, kb)

    # --- step 4: build LLM context ---
    context = _build_context(
        job_record=job_record,
        fit_score=fit_score,
        company_brief=company_brief,
        profile=profile,
        kb=kb,
        interviewer_levels=levels,
    )

    # --- step 5: LLM call ---
    jt = getattr(getattr(job_record, "job_title", None), "value", "") if job_record else ""
    cn = getattr(getattr(job_record, "company_name", None), "value", "") if job_record else ""
    pack = _call_prep_coach(
        job_record=job_record,
        fit_score=fit_score,
        company_brief=company_brief,
        profile=profile,
        kb=kb,
        interviewer_levels=levels,
        job_title=jt,
        company=cn,
    )

    # --- step 6: post-generation validation (resolves named IDs, validates paths) ---
    pack = _validate_pack(pack, profile, kb, levels, story_lookup)

    # Stamp interviewer levels detected
    pack.interviewer_levels_detected = levels

    return pack


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os

    profile_path = "profile.yaml"
    kb_path      = "knowledge_base.yaml"

    if not os.path.exists(profile_path):
        print(f"profile.yaml not found. Place it in the current directory.")
        sys.exit(1)
    if not os.path.exists(kb_path):
        print(f"knowledge_base.yaml not found. Place it in the current directory.")
        sys.exit(1)

    print("[InterviewPrepCoach] Running in demo mode (no job_record — using profile only)")
    print("[InterviewPrepCoach] Pass a real OkRecord via prep_interview() in production\n")

    # Demo: load profile only, no JD
    profile = _load_yaml(profile_path, "profile")
    kb      = _load_yaml(kb_path, "knowledge_base")

    print(f"profile.yaml loaded: {len(profile.get('hero_stories', []))} hero stories")
    print(f"knowledge_base.yaml loaded:")
    print(f"  Frameworks: {list(kb.get('frameworks', {}).keys())}")
    print(f"  AI questions: {len(kb.get('question_bank', {}).get('ai_specific', []))}")
    print(f"  General PM questions: {len(kb.get('question_bank', {}).get('general_pm', []))}")
    print(f"  Story mappings: {len(kb.get('story_mappings', {}))}")
    print(f"  Technical concepts: {len(kb.get('technical_concepts', {}))}")
    print("\n[InterviewPrepCoach] Ready. Call prep_interview(job_record, ...) to generate a prep pack.")
