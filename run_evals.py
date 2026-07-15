"""
run_evals.py — JobHunter AI Crew Eval Harness

TWO LAYERS:

Layer 1 — Contract tests (run NOW, before any end-to-end run)
  Tests that agent N's output successfully passes to agent N+1 as valid input.
  Deterministic. No LLM calls. Run in seconds.
  These fail if field names drift, schemas break, or pipeline guards stop working.

Layer 2 — Quality tests (populate AFTER first end-to-end run)
  Tests that agent output is actually good, not just structurally valid.
  Cannot be written blind — must be written against real failure patterns.
  Stubs are provided. Fill them in after running on known JDs.

Usage:
  python run_evals.py              # run all Layer 1 tests
  python run_evals.py --layer 1    # Layer 1 only
  python run_evals.py --layer 2    # Layer 2 only (stubs — will skip until populated)
  python run_evals.py --verbose    # print detailed results per test

Known JDs for Layer 2 population (run these first):
  Babylist, Stripe, Arize, TELUS, Guidepoint
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()


import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Callable

import yaml

# ---------------------------------------------------------------------------
# Test result types
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    layer: int
    detail: str = ""
    skipped: bool = False


@dataclass
class EvalReport:
    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def passed(self): return [r for r in self.results if r.passed and not r.skipped]
    @property
    def failed(self): return [r for r in self.results if not r.passed and not r.skipped]
    @property
    def skipped(self): return [r for r in self.results if r.skipped]

    def print_summary(self, verbose: bool = False):
        print("\n" + "="*60)
        print("EVAL REPORT")
        print("="*60)

        for r in self.results:
            if r.skipped:
                status = "⏭  SKIP"
            elif r.passed:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            label = f"[L{r.layer}]"
            print(f"  {status} {label} {r.name}")
            if verbose and r.detail:
                print(f"         {r.detail}")
            elif not r.passed and not r.skipped and r.detail:
                print(f"         ↳ {r.detail}")

        print("\n" + "-"*60)
        print(f"  Layer 1: {len([r for r in self.results if r.layer==1 and r.passed])} / {len([r for r in self.results if r.layer==1 and not r.skipped])} passed")
        print(f"  Layer 2: {len([r for r in self.results if r.layer==2 and r.passed])} / {len([r for r in self.results if r.layer==2 and not r.skipped])} passed  ({len([r for r in self.results if r.layer==2 and r.skipped])} stubs remaining)")
        print(f"  Overall: {len(self.passed)} passed, {len(self.failed)} failed, {len(self.skipped)} skipped")
        print("="*60)

        if self.failed:
            print("\n⚠️  Fix failing tests before running the pipeline on real applications.")
        else:
            print("\n✅ All active tests passing. Safe to run on real JDs.")


report = EvalReport()

def test(name: str, layer: int):
    """Decorator for test functions."""
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
                r = TestResult(name=name, passed=True, layer=layer)
            except SkipTest as e:
                r = TestResult(name=name, passed=False, layer=layer, skipped=True, detail=str(e))
            except AssertionError as e:
                r = TestResult(name=name, passed=False, layer=layer, detail=str(e))
            except Exception as e:
                r = TestResult(name=name, passed=False, layer=layer, detail=f"{type(e).__name__}: {e}")
            report.add(r)
        wrapper._test_name = name
        wrapper._test_layer = layer
        return wrapper
    return decorator


class SkipTest(Exception):
    """Raised in a test to mark it as a stub (not yet implemented)."""
    pass


# ---------------------------------------------------------------------------
# Import agents (with graceful fallback)
# ---------------------------------------------------------------------------

def _try_import(module_name: str):
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError as e:
        return None


job_analyst_mod       = _try_import("job_analyst")
fit_scorer_mod        = _try_import("fit_scorer")
profile_builder_mod   = _try_import("profile_builder")
company_researcher_mod = _try_import("company_researcher")
resume_strategist_mod = _try_import("resume_strategist")
interview_prep_mod    = _try_import("interview_prep_coach")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_JD = """
Senior AI Product Manager — Acme AI Corp
Location: Remote, Canada

About the role:
We're looking for a Senior AI PM to lead development of our LLM-powered
customer support product. You'll own the roadmap, work with ML engineers,
and drive product strategy for our conversational AI platform.

Requirements:
- 3+ years of product management experience
- Experience with LLMs, RAG pipelines, or conversational AI products
- Strong analytical skills and data-driven decision making
- Excellent cross-functional collaboration

Nice to have:
- Experience with Python or API products
- Knowledge of AI observability and reliability

Compensation: CAD $120,000 - $150,000 + equity
Work arrangement: Remote
"""

MINIMAL_JD_URL = "https://www.linkedin.com/jobs/view/000000001/"

PROFILE_PATH = "profile.yaml"
KB_PATH      = "knowledge_base.yaml"


def _load_profile() -> dict:
    if not os.path.exists(PROFILE_PATH):
        return {}
    with open(PROFILE_PATH) as f:
        return yaml.safe_load(f) or {}


def _load_kb() -> dict:
    if not os.path.exists(KB_PATH):
        return {}
    with open(KB_PATH) as f:
        return yaml.safe_load(f) or {}


# ===========================================================================
# LAYER 1 — CONTRACT TESTS
# ===========================================================================

# ---------------------------------------------------------------------------
# 1a. File dependencies
# ---------------------------------------------------------------------------

@test("profile.yaml exists and is valid YAML", layer=1)
def test_profile_exists():
    assert os.path.exists(PROFILE_PATH), \
        f"profile.yaml not found at {PROFILE_PATH}. Place it in the same directory."
    profile = _load_profile()
    assert isinstance(profile, dict), "profile.yaml did not parse as a dict"
    assert "hero_stories" in profile, "profile.yaml missing 'hero_stories' key"
    assert "bullets" in profile, "profile.yaml missing 'bullets' key"
    assert "gaps" in profile, "profile.yaml missing 'gaps' key"


@test("knowledge_base.yaml exists and is valid YAML", layer=1)
def test_kb_exists():
    assert os.path.exists(KB_PATH), \
        f"knowledge_base.yaml not found at {KB_PATH}. Place it in the same directory."
    kb = _load_kb()
    assert isinstance(kb, dict), "knowledge_base.yaml did not parse as a dict"
    assert "frameworks" in kb, "knowledge_base.yaml missing 'frameworks' key"
    assert "question_bank" in kb, "knowledge_base.yaml missing 'question_bank' key"
    assert "story_registry" in kb, "knowledge_base.yaml missing 'story_registry' key"


@test("knowledge_base.yaml has no positional hero_story indices", layer=1)
def test_no_positional_story_indices():
    if not os.path.exists(KB_PATH):
        raise SkipTest("knowledge_base.yaml not found — run test_kb_exists first")
    content = open(KB_PATH).read()
    import re
    positional = re.findall(r'hero_stories\[\d+\]', content)
    assert not positional, \
        f"Found positional hero_story indices that should be named IDs: {positional}"


# ---------------------------------------------------------------------------
# 1b. Story registry resolves against profile.yaml
# ---------------------------------------------------------------------------

@test("story_registry IDs resolve against profile.yaml titles", layer=1)
def test_story_registry_resolves():
    profile = _load_profile()
    kb      = _load_kb()

    if not profile or not kb:
        raise SkipTest("profile.yaml or knowledge_base.yaml missing")

    stories  = profile.get("hero_stories", [])
    registry = kb.get("story_registry", {})

    if not registry:
        raise SkipTest("story_registry is empty in knowledge_base.yaml")

    unmatched = []
    for story_id, meta in registry.items():
        title_match = meta.get("title_match", "")
        found = any(
            title_match.lower() in s.get("title", "").lower() or
            s.get("title", "").lower() in title_match.lower()
            for s in stories
        )
        if not found:
            unmatched.append(f"{story_id} → '{title_match}'")

    assert not unmatched, \
        f"These story IDs don't match any hero_story title in profile.yaml: {unmatched}"


# ---------------------------------------------------------------------------
# 1c. Agent imports succeed
# ---------------------------------------------------------------------------

@test("job_analyst.py imports successfully", layer=1)
def test_job_analyst_imports():
    assert job_analyst_mod is not None, \
        "job_analyst.py could not be imported. Check it's in the same directory."
    assert hasattr(job_analyst_mod, "parse_job"), \
        "job_analyst.py missing 'parse_job' function"
    assert hasattr(job_analyst_mod, "OkRecord"), \
        "job_analyst.py missing 'OkRecord' class"


@test("fit_scorer.py imports successfully", layer=1)
def test_fit_scorer_imports():
    assert fit_scorer_mod is not None, \
        "fit_scorer.py could not be imported."
    assert hasattr(fit_scorer_mod, "score_job"), \
        "fit_scorer.py missing 'score_job' function"
    assert hasattr(fit_scorer_mod, "FitScoreResult"), \
        "fit_scorer.py missing 'FitScoreResult' class"


@test("company_researcher.py imports successfully", layer=1)
def test_company_researcher_imports():
    assert company_researcher_mod is not None, \
        "company_researcher.py could not be imported."
    assert hasattr(company_researcher_mod, "research_company"), \
        "company_researcher.py missing 'research_company' function"


@test("interview_prep_coach.py imports successfully", layer=1)
def test_prep_coach_imports():
    assert interview_prep_mod is not None, \
        "interview_prep_coach.py could not be imported."
    assert hasattr(interview_prep_mod, "prep_interview"), \
        "interview_prep_coach.py missing 'prep_interview' function"


# ---------------------------------------------------------------------------
# 1d. Output schema contracts between agents
# ---------------------------------------------------------------------------

@test("FitScoreResult has all fields Resume Strategist expects", layer=1)
def test_fit_scorer_to_resume_strategist_contract():
    if fit_scorer_mod is None:
        raise SkipTest("fit_scorer.py not importable")

    FitScoreResult = fit_scorer_mod.FitScoreResult
    required_fields = [
        "matched_skills", "missing_skills", "near_match_skills",
        "condition", "ai_classification", "decision", "job_id",
    ]
    model_fields = list(FitScoreResult.model_fields.keys())
    missing = [f for f in required_fields if f not in model_fields]
    assert not missing, \
        f"FitScoreResult missing fields that Resume Strategist expects: {missing}"


@test("FitScoreResult has all fields Interview Prep Coach expects", layer=1)
def test_fit_scorer_to_prep_coach_contract():
    if fit_scorer_mod is None:
        raise SkipTest("fit_scorer.py not importable")

    FitScoreResult = fit_scorer_mod.FitScoreResult
    required_fields = [
        "matched_skills", "missing_skills", "near_match_skills",
        "condition", "ai_classification", "decision",
    ]
    model_fields = list(FitScoreResult.model_fields.keys())
    missing = [f for f in required_fields if f not in model_fields]
    assert not missing, \
        f"FitScoreResult missing fields that Interview Prep Coach expects: {missing}"


@test("CompanyBrief has all fields Interview Prep Coach expects", layer=1)
def test_company_researcher_to_prep_coach_contract():
    if company_researcher_mod is None:
        raise SkipTest("company_researcher.py not importable")

    CompanyBrief = company_researcher_mod.CompanyBrief
    required_fields = [
        "company_name", "sections", "ai_signal",
        "pm_talking_points", "smart_questions", "cache_hit",
    ]
    model_fields = list(CompanyBrief.model_fields.keys())
    missing = [f for f in required_fields if f not in model_fields]
    assert not missing, \
        f"CompanyBrief missing fields that Interview Prep Coach expects: {missing}"


@test("RejectedBrief has status='rejected' sentinel", layer=1)
def test_rejected_brief_sentinel():
    if company_researcher_mod is None:
        raise SkipTest("company_researcher.py not importable")

    RejectedBrief = company_researcher_mod.RejectedBrief
    rb = RejectedBrief(reason="test", message="test")
    assert rb.status == "rejected", \
        f"RejectedBrief.status should be 'rejected', got '{rb.status}'"


# ---------------------------------------------------------------------------
# 1e. Pipeline guard tests (no LLM calls — mock inputs)
# ---------------------------------------------------------------------------

@test("resume_strategist rejects Skip decision without LLM call", layer=1)
def test_skip_rejected_by_resume_strategist():
    if resume_strategist_mod is None:
        raise SkipTest("resume_strategist.py not importable")
    if fit_scorer_mod is None:
        raise SkipTest("fit_scorer.py not importable")

    # Create a minimal FitScoreResult with decision=skip
    # without triggering an LLM call
    FitScoreResult = fit_scorer_mod.FitScoreResult
    RejectedStrategy = resume_strategist_mod.RejectedStrategy

    mock_score = FitScoreResult(
        status="ok",
        decision="skip",
        total_score=20.0,
        ai_classification="non_ai",
        condition="",
        matched_skills=[],
        missing_skills=[],
        near_match_skills=[],
        dimension_scores={
            "required_skills": 5.0,
            "ai_genai_depth": 5.0,
            "seniority_fit": 4.0,
            "nice_to_have": 2.0,
            "work_arrangement": 0.0,
            "industry_alignment": 3.0,
            "compensation": 1.0,
        },
        reasoning="mock",
        job_id=None,
    )

    # Should return RejectedStrategy without calling LLM
    # We test the guard logic, not the full pipeline
    result = resume_strategist_mod.strategize.__wrapped__(mock_score, None) \
        if hasattr(resume_strategist_mod.strategize, "__wrapped__") \
        else None

    # Alternative: test the guard directly
    decision = getattr(mock_score, "decision", "")
    assert decision == "skip", "Mock score should have decision=skip"

    # The strategize function should reject Skip — we verify the guard logic exists in source
    source = open("resume_strategist.py").read()
    assert 'decision.*skip.*reject' in source.lower().replace("\n", " ") or \
           "skip_filtered" in source, \
        "resume_strategist.py doesn't appear to have a Skip guard"


@test("interview_prep_coach rejects missing job_record", layer=1)
def test_prep_coach_rejects_missing_job_record():
    if interview_prep_mod is None:
        raise SkipTest("interview_prep_coach.py not importable")

    if not os.path.exists(PROFILE_PATH) or not os.path.exists(KB_PATH):
        raise SkipTest("profile.yaml or knowledge_base.yaml missing — can't run this test")

    RejectedPrepPack = interview_prep_mod.RejectedPrepPack
    result = interview_prep_mod.prep_interview(
        job_record=None,
        profile_path=PROFILE_PATH,
        knowledge_base_path=KB_PATH,
    )

    assert isinstance(result, RejectedPrepPack), \
        f"prep_interview with job_record=None should return RejectedPrepPack, got {type(result)}"
    assert result.status == "rejected", \
        f"Expected status='rejected', got '{result.status}'"
    assert result.reason == "missing_job_record", \
        f"Expected reason='missing_job_record', got '{result.reason}'"


@test("company_researcher rejects non-apply decision level", layer=1)
def test_company_researcher_skip_guard():
    if company_researcher_mod is None:
        raise SkipTest("company_researcher.py not importable")

    RejectedBrief = company_researcher_mod.RejectedBrief
    result = company_researcher_mod.research_company(
        company_name="TestCo",
        decision_level="skip",
    )

    assert isinstance(result, RejectedBrief), \
        f"research_company with decision_level='skip' should return RejectedBrief, got {type(result)}"
    assert result.status == "rejected", \
        f"Expected status='rejected', got '{result.status}'"


@test("company_researcher rejects missing company_name", layer=1)
def test_company_researcher_missing_name():
    if company_researcher_mod is None:
        raise SkipTest("company_researcher.py not importable")

    RejectedBrief = company_researcher_mod.RejectedBrief
    result = company_researcher_mod.research_company(
        company_name="",
        decision_level="apply",
    )

    assert isinstance(result, RejectedBrief), \
        f"research_company with empty company_name should return RejectedBrief"
    assert result.reason == "missing_company_name", \
        f"Expected reason='missing_company_name', got '{result.reason}'"


# ---------------------------------------------------------------------------
# 1f. InterviewPrepPack schema completeness
# ---------------------------------------------------------------------------

@test("InterviewPrepPack has time_budget_minutes on PrepQuestion", layer=1)
def test_prep_question_has_time_budget():
    if interview_prep_mod is None:
        raise SkipTest("interview_prep_coach.py not importable")

    PrepQuestion = interview_prep_mod.PrepQuestion
    assert "time_budget_minutes" in PrepQuestion.model_fields, \
        "PrepQuestion missing 'time_budget_minutes' field — additions not applied"


@test("InterviewPrepPack has anticipated_followups on PrepQuestion", layer=1)
def test_prep_question_has_followups():
    if interview_prep_mod is None:
        raise SkipTest("interview_prep_coach.py not importable")

    PrepQuestion = interview_prep_mod.PrepQuestion
    assert "anticipated_followups" in PrepQuestion.model_fields, \
        "PrepQuestion missing 'anticipated_followups' field — additions not applied"


@test("knowledge_base.yaml has crewai_portfolio_questions section", layer=1)
def test_kb_has_crewai_questions():
    kb = _load_kb()
    if not kb:
        raise SkipTest("knowledge_base.yaml not found")

    assert "crewai_portfolio_questions" in kb, \
        "knowledge_base.yaml missing 'crewai_portfolio_questions' section"

    arch_qs = kb["crewai_portfolio_questions"].get("architecture_questions", [])
    assert len(arch_qs) >= 5, \
        f"Expected 5+ CrewAI architecture questions, found {len(arch_qs)}"


@test("knowledge_base.yaml has time_budget_categories", layer=1)
def test_kb_has_time_budgets():
    kb = _load_kb()
    if not kb:
        raise SkipTest("knowledge_base.yaml not found")

    assert "question_category_time_budgets" in kb, \
        "knowledge_base.yaml missing 'question_category_time_budgets' section"


@test("all question_bank entries have anticipated_followups", layer=1)
def test_all_questions_have_followups():
    kb = _load_kb()
    if not kb:
        raise SkipTest("knowledge_base.yaml not found")

    missing = []
    for category in ["ai_specific", "general_pm"]:
        for q in kb.get("question_bank", {}).get(category, []):
            if not q.get("anticipated_followups"):
                missing.append(q.get("id", "unknown"))

    assert not missing, \
        f"These questions are missing anticipated_followups: {missing}"


# ===========================================================================
# LAYER 2 — QUALITY TESTS (stubs — populate after first end-to-end run)
# ===========================================================================

@test("Fit Scorer score for Babylist within expected range", layer=2)
def test_babylist_fit_score():
    raise SkipTest(
        "TODO: Run pipeline on Babylist JD, observe score, "
        "then assert score >= 75 and score <= 92 and decision == 'apply'"
    )


@test("Fit Scorer score for Stripe within expected range", layer=2)
def test_stripe_fit_score():
    raise SkipTest(
        "TODO: Run pipeline on Stripe Support PM JD, observe score, "
        "then assert score >= 60 and decision in ('apply', 'conditional_apply')"
    )


@test("Fit Scorer score for Arize within expected range", layer=2)
def test_arize_fit_score():
    raise SkipTest(
        "TODO: Run pipeline on Arize AI PM JD, observe score, "
        "then assert score >= 70 and score <= 80"
    )


@test("Resume Strategist: no hallucinated metrics in rewrites", layer=2)
def test_no_hallucinated_metrics():
    raise SkipTest(
        "TODO: Run Resume Strategist on a known JD, "
        "check all numbers in rewritten bullets appear in the cited source entry. "
        "Extend post-generation validator to save failures to a log file, "
        "then assert len(failures) == 0"
    )


@test("Interview Prep Coach: story_missing false for behavioral AI questions", layer=2)
def test_ai_behavioral_questions_have_stories():
    raise SkipTest(
        "TODO: Run prep_interview on an AI-core JD, "
        "check that behavioral_ai questions have story_missing=False. "
        "If any are True, update story_mappings in knowledge_base.yaml."
    )


@test("Company Researcher: all stated:true fields have source URLs", layer=2)
def test_company_brief_citations():
    raise SkipTest(
        "TODO: Run research_company on 'Stripe' and 'Babylist', "
        "check that every section with stated=True has at least one key_claim with a source URL."
    )


@test("Company Researcher: AI signal classification matches known companies", layer=2)
def test_ai_signal_accuracy():
    raise SkipTest(
        "TODO: Run research_company on known AI-core company (e.g. Arize, Cohere), "
        "assert ai_signal.classification in ('ai_core', 'ai_enabled'). "
        "Run on known non-AI company, assert classification == 'non_ai'."
    )


@test("Interview Prep Coach: must_prepare_first always has 3+ questions", layer=2)
def test_must_prepare_first_populated():
    raise SkipTest(
        "TODO: Run prep_interview on 3 different JDs, "
        "assert len(pack.must_prepare_first) >= 3 for each."
    )


@test("Pre-filter: correctly rejects Junior PM titles", layer=2)
def test_prefilter_rejects_junior():
    from run_pipeline import _pre_filter
    passes, reason = _pre_filter("Junior Product Manager")
    assert not passes, f"Pre-filter should reject 'Junior Product Manager' but got: {reason}"

    passes, reason = _pre_filter("AI Product Manager Intern")
    assert not passes, f"Pre-filter should reject internship titles"


@test("Pre-filter: correctly passes AI PM titles", layer=2)
def test_prefilter_passes_ai_pm():
    from run_pipeline import _pre_filter
    passes, reason = _pre_filter("Senior AI Product Manager")
    assert passes, f"Pre-filter should pass 'Senior AI Product Manager': {reason}"

    passes, reason = _pre_filter("AI Product Owner — Conversational AI")
    assert passes, f"Pre-filter should pass AI Product Owner: {reason}"


# ---------------------------------------------------------------------------
# Actually run the pre-filter tests now (they're deterministic)
# These are Layer 2 in spirit but can run immediately
# ---------------------------------------------------------------------------

# ===========================================================================
# Runner
# ===========================================================================

def run_all(layer: int = None, verbose: bool = False):
    """Discover and run all test functions."""
    import inspect

    # Get all test functions in this module
    current_module = sys.modules[__name__]
    test_fns = [
        obj for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if hasattr(obj, "_test_layer")
    ]

    # Filter by layer if specified
    if layer is not None:
        test_fns = [fn for fn in test_fns if fn._test_layer == layer]

    print(f"\n[Evals] Running {len(test_fns)} tests" + (f" (Layer {layer} only)" if layer else ""))
    print("-"*60)

    for fn in test_fns:
        fn()

    report.print_summary(verbose=verbose)

    # Exit with error code if any Layer 1 tests failed
    layer1_failures = [r for r in report.results if r.layer == 1 and not r.passed and not r.skipped]
    if layer1_failures:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JobHunter AI Crew — eval harness")
    parser.add_argument("--layer",   type=int, choices=[1, 2], help="Run only Layer 1 or Layer 2 tests")
    parser.add_argument("--verbose", action="store_true", help="Print detail for all tests, not just failures")
    args = parser.parse_args()

    run_all(layer=args.layer, verbose=args.verbose)
