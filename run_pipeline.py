"""
run_pipeline.py — JobHunter AI Crew Orchestrator

Two entry points. Same pipeline underneath.

ENTRY POINT 1 — Manual:
    python run_pipeline.py --manual --jd "paste JD text here" --url "https://linkedin.com/jobs/..."

ENTRY POINT 2 — Automated (reads Gmail alerts + Indeed RSS):
    python run_pipeline.py --auto

PIPELINE FLOW:
    Job Analyst → Profile Builder (once) → Fit Scorer → Presenter (batch)
    For Apply / Conditional Apply:
        → Company Researcher → Interview Prep Coach → Interview Kit

INTAKE FLOW (automated only):
    Gmail alerts (LinkedIn + Indeed) ─┐
    Indeed RSS feeds                  ├─→ Pre-filter → Dedup → Queue → Pipeline
    (SerpAPI Google Jobs — optional)  ┘

SETUP:
    pip install anthropic pyyaml requests feedparser
    export ANTHROPIC_API_KEY=...
    export SERPAPI_KEY=...       # for Company Researcher
    
    Gmail intake requires ANTHROPIC_API_KEY only — uses Claude's Gmail MCP
    via the Anthropic API (no separate Google Cloud OAuth needed).

FILES CREATED BY THIS SCRIPT:
    seen_jobs.json         — deduplication store
    jobs_queue.json        — jobs pending pipeline processing
    output/                — per-job outputs (kit HTML files, resume diffs)
    batch_report.html      — Presenter batch dashboard (all scored jobs)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.request import urlopen
from urllib.parse import urlencode, quote_plus
import xml.etree.ElementTree as ET

import yaml
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Optional imports — graceful fallbacks
# ---------------------------------------------------------------------------
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

# Try importing crew agents
try:
    from job_analyst import parse_job, OkRecord
    from profile_builder import build_profile, PreferencesConfig, TargetComp
    from fit_scorer import score_job, FitScoreResult
    from presenter import generate_report, attach_display_meta
    from company_researcher import research_company, CompanyBrief
    from resume_strategist import strategize
    from interview_prep_coach import prep_interview
    from interview_kit import generate_interview_kit
except ImportError as e:
    print(f"[Orchestrator] Warning: could not import agent: {e}")
    print("[Orchestrator] Ensure all agent files are in the same directory.")

_client = Anthropic()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEN_JOBS_FILE   = "seen_jobs.json"
QUEUE_FILE       = "jobs_queue.json"
OUTPUT_DIR       = "output"
PROFILE_PATH     = "profile.yaml"
RESUME_PDF_PATH  = "resume.pdf"        # PDF resume for Profile Builder
KB_PATH          = "knowledge_base.yaml"
BATCH_REPORT     = "batch_report.html"

# Job titles to search (from Anshu's profile — override in CLI)
DEFAULT_SEARCH_TERMS = [
    "AI Product Manager",
    "AI Product Owner",
    "Product Manager AI",
    "AI PM",
]

DEFAULT_LOCATION = "Canada"

# Indeed RSS — one URL per search term
INDEED_RSS_TEMPLATE = "https://www.indeed.com/rss?q={query}&l={location}&sort=date&fromage=3"

# Gmail MCP search query for job alert emails
GMAIL_SEARCH_QUERY = (
    "from:(jobalert@linkedin.com OR jobalert@indeed.com OR jobs-noreply@linkedin.com) "
    "subject:(job alert OR new jobs) "
    "newer_than:2d"
)

# Pre-filter: role keywords that must appear in job title
ROLE_KEYWORDS = [
    "ai product manager", "ai product owner", "product manager", "ai pm",
    "genai pm", "product manager ai", "conversational ai", "llm product",
    "machine learning product", "ai platform pm",
]

# Pre-filter: title keywords that cause immediate rejection
NEGATIVE_KEYWORDS = [
    "junior", "intern", "internship", "co-op", "coop", "entry level",
    "entry-level", "graduate", "student", "data scientist", "data engineer",
    "software engineer", "ml engineer", "qa ", "quality assurance",
]

# Minimum Fit Scorer decision to process downstream agents
# "maybe" = process Apply, Conditional Apply, Maybe
# "conditional_apply" = only Apply and Conditional Apply
MIN_DECISION_FOR_DOWNSTREAM = "conditional_apply"

DECISION_RANK = {"apply": 4, "conditional_apply": 3, "maybe": 2, "skip": 1}


# ---------------------------------------------------------------------------
# Seen jobs / queue helpers
# ---------------------------------------------------------------------------

def _load_seen_jobs() -> dict:
    """Load seen_jobs.json. Returns {url: {title, company, date_seen, status}}."""
    if not os.path.exists(SEEN_JOBS_FILE):
        return {}
    try:
        with open(SEEN_JOBS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _mark_seen(url: str, title: str = "", company: str = "", status: str = "queued"):
    """Add a job URL to the seen jobs store."""
    seen = _load_seen_jobs()
    seen[url] = {
        "title": title,
        "company": company,
        "date_seen": datetime.now().isoformat(),
        "status": status,
    }
    with open(SEEN_JOBS_FILE, "w") as f:
        json.dump(seen, f, indent=2)


def _load_queue() -> list[dict]:
    """Load pending jobs queue."""
    if not os.path.exists(QUEUE_FILE):
        return []
    try:
        with open(QUEUE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_queue(queue: list[dict]):
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def _add_to_queue(job: dict):
    queue = _load_queue()
    queue.append(job)
    _save_queue(queue)


# ---------------------------------------------------------------------------
# Pre-filter (deterministic — no LLM, no API cost)
# ---------------------------------------------------------------------------

def _pre_filter(title: str, location: str = "") -> tuple[bool, str]:
    """
    Quick keyword filter before spending any API calls.
    Returns (passes, reason).
    """
    t = title.lower()

    # Reject negative keywords
    for neg in NEGATIVE_KEYWORDS:
        if neg in t:
            return False, f"rejected: negative keyword '{neg}'"

    # Must match at least one role keyword
    if not any(kw in t for kw in ROLE_KEYWORDS):
        return False, f"rejected: no role keyword match in '{title}'"

    # Location check (lenient — blank location passes)
    if location:
        loc = location.lower()
        loc_ok = any(kw in loc for kw in [
            "canada", "ontario", "toronto", "vancouver", "montreal",
            "remote", "anywhere", "hybrid", "work from home",
        ])
        if not loc_ok:
            return False, f"rejected: location '{location}'"

    return True, "pass"


# ---------------------------------------------------------------------------
# Indeed RSS intake
# ---------------------------------------------------------------------------

def _fetch_indeed_rss(
    search_terms: list[str] = None,
    location: str = DEFAULT_LOCATION,
) -> list[dict]:
    """
    Fetch job listings from Indeed RSS feeds.
    Returns list of {title, url, company, location, description, source}.
    """
    search_terms = search_terms or DEFAULT_SEARCH_TERMS
    jobs = []

    for term in search_terms:
        url = INDEED_RSS_TEMPLATE.format(
            query=quote_plus(term),
            location=quote_plus(location),
        )
        try:
            if HAS_FEEDPARSER:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    jobs.append({
                        "title":       entry.get("title", ""),
                        "url":         entry.get("link", ""),
                        "company":     entry.get("author", ""),
                        "location":    location,
                        "description": entry.get("summary", "")[:500],
                        "source":      "indeed_rss",
                        "search_term": term,
                    })
            else:
                # Fallback: raw XML parsing
                response = urlopen(url, timeout=10)
                tree = ET.parse(response)
                root = tree.getroot()
                for item in root.findall(".//item"):
                    title = item.findtext("title") or ""
                    link  = item.findtext("link") or ""
                    desc  = item.findtext("description") or ""
                    jobs.append({
                        "title":       title,
                        "url":         link,
                        "company":     "",
                        "location":    location,
                        "description": desc[:500],
                        "source":      "indeed_rss",
                        "search_term": term,
                    })
        except Exception as e:
            print(f"[Intake] Indeed RSS failed for '{term}': {e}")
            continue

    print(f"[Intake] Indeed RSS: found {len(jobs)} listings across {len(search_terms)} searches")
    return jobs


# ---------------------------------------------------------------------------
# Gmail intake (via Anthropic API + Gmail MCP)
# ---------------------------------------------------------------------------

GMAIL_MCP_URL = "https://gmailmcp.googleapis.com/mcp/v1"

def _fetch_gmail_alerts() -> list[dict]:
    """
    Read LinkedIn and Indeed job alert emails via Gmail MCP.
    Uses Anthropic API with Gmail MCP server — no separate Google OAuth needed.
    Returns list of {title, url, company, location, source}.
    """
    print("[Intake] Reading Gmail job alerts via Claude + Gmail MCP...")

    system_prompt = """You are a job alert email parser.
Search the user's Gmail for recent LinkedIn and Indeed job alert emails.
For each job listing found, extract: title, url, company, location.
Return ONLY a JSON array of objects with keys: title, url, company, location, source.
source should be "linkedin_email" or "indeed_email".
If no emails found, return an empty array [].
Return ONLY the JSON array, no other text."""

    user_message = (
        f"Search Gmail for job alert emails using this query: {GMAIL_SEARCH_QUERY}\n"
        f"Extract all individual job listings from those emails.\n"
        f"Return a JSON array of job objects."
    )

    try:
        response = _client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            mcp_servers=[{
                "type": "url",
                "url": GMAIL_MCP_URL,
                "name": "gmail-mcp",
            }],
            tools=[{
                "type": "mcp_toolset",
                "mcp_server_name": "gmail-mcp",
            }],
            betas=["mcp-client-2025-11-20"],
        )

        # Extract text from response
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        # Parse JSON array from response
        # Strip any markdown code fences
        text = re.sub(r"```(?:json)?", "", text).strip()

        jobs_raw = json.loads(text)
        jobs = []
        for j in jobs_raw:
            jobs.append({
                "title":    j.get("title", ""),
                "url":      j.get("url", ""),
                "company":  j.get("company", ""),
                "location": j.get("location", ""),
                "description": "",
                "source":   j.get("source", "gmail"),
            })

        print(f"[Intake] Gmail: found {len(jobs)} job listings from alert emails")
        return jobs

    except json.JSONDecodeError as e:
        print(f"[Intake] Gmail: JSON parse failed — {e}")
        return []
    except Exception as e:
        print(f"[Intake] Gmail: failed — {e}")
        print("[Intake] Gmail intake requires Gmail MCP to be connected.")
        return []


# ---------------------------------------------------------------------------
# Job intake — pre-filter + dedup + queue
# ---------------------------------------------------------------------------

def run_intake(
    sources: list[str] = None,
    search_terms: list[str] = None,
    location: str = DEFAULT_LOCATION,
    dry_run: bool = False,
) -> list[dict]:
    """
    Run job intake from all specified sources.
    Pre-filters against role/location keywords.
    Deduplicates against seen_jobs.json.
    Queues new matching jobs for the pipeline.

    Args:
        sources:      ["gmail", "indeed_rss"]. Default: both.
        search_terms: Job title search terms for RSS. Default: DEFAULT_SEARCH_TERMS.
        location:     Location filter for RSS. Default: "Canada".
        dry_run:      If True, show what would be queued without actually queuing.

    Returns:
        List of newly queued job dicts.
    """
    sources      = sources or ["gmail", "indeed_rss"]
    search_terms = search_terms or DEFAULT_SEARCH_TERMS

    seen = _load_seen_jobs()
    all_raw: list[dict] = []

    # Fetch from each source
    if "indeed_rss" in sources:
        all_raw.extend(_fetch_indeed_rss(search_terms, location))

    if "gmail" in sources:
        all_raw.extend(_fetch_gmail_alerts())

    print(f"\n[Intake] Total raw listings: {len(all_raw)}")

    # Pre-filter and deduplicate
    queued = []
    filtered_count = 0
    dedup_count = 0

    for job in all_raw:
        url   = job.get("url", "").strip()
        title = job.get("title", "").strip()

        if not url or not title:
            filtered_count += 1
            continue

        # Deduplication
        if url in seen:
            dedup_count += 1
            continue

        # Pre-filter
        passes, reason = _pre_filter(title, job.get("location", ""))
        if not passes:
            print(f"  [filter] {title[:60]} — {reason}")
            filtered_count += 1
            _mark_seen(url, title, job.get("company", ""), status="filtered")
            continue

        # Passes — queue it
        print(f"  [queue] {title[:60]} ({job.get('company', 'unknown')}) [{job.get('source','')}]")
        queued.append(job)

        if not dry_run:
            _add_to_queue(job)
            _mark_seen(url, title, job.get("company", ""), status="queued")

    print(f"\n[Intake] Summary:")
    print(f"  Raw listings:    {len(all_raw)}")
    print(f"  Filtered out:    {filtered_count}")
    print(f"  Already seen:    {dedup_count}")
    print(f"  Newly queued:    {len(queued)}")

    return queued


# ---------------------------------------------------------------------------
# Single-job pipeline execution
# ---------------------------------------------------------------------------

def _run_single_job(
    jd_text: str,
    job_url: str,
    candidate_profile,
    results_accumulator: list,
    output_dir: str = OUTPUT_DIR,
):
    """
    Run the full pipeline for one job.
    Appends FitScoreResult + metadata to results_accumulator.
    Generates interview kit for Apply/Conditional Apply jobs.
    """
    os.makedirs(output_dir, exist_ok=True)
    job_title = "Unknown"
    company   = "Unknown"

    try:
        # ── Step 1: Job Analyst ──────────────────────────────────────────
        print(f"\n  [1/5] Job Analyst: parsing JD...")
        job_record = parse_job(jd_text, job_url)

        if getattr(job_record, "status", "ok") == "rejected":
            print(f"  [!] Job Analyst rejected — {getattr(job_record, 'reason', '')}")
            return

        job_title = getattr(getattr(job_record, "job_title", None), "value", "Unknown") or "Unknown"
        company   = getattr(getattr(job_record, "company_name", None), "value", "Unknown") or "Unknown"
        print(f"  ✓  Parsed: {job_title} @ {company}")

        # ── Step 2: Fit Scorer ───────────────────────────────────────────
        print(f"  [2/5] Fit Scorer: scoring...")
        fit_score = score_job(job_record, candidate_profile)

        decision = getattr(fit_score, "decision", "skip")
        score_val = getattr(fit_score, "total_score", 0)
        condition = getattr(fit_score, "condition", "") or ""
        print(f"  ✓  Score: {score_val:.0f}% | Decision: {decision.upper()} {('— ' + condition) if condition else ''}")

        # Attach display meta for Presenter
        fit_score = attach_display_meta(fit_score, job_title, company, "Canada")
        results_accumulator.append(fit_score)

        # Mark job as scored
        if job_url:
            seen = _load_seen_jobs()
            if job_url in seen:
                seen[job_url]["status"] = f"scored:{decision}"
                with open(SEEN_JOBS_FILE, "w") as f:
                    json.dump(seen, f, indent=2)

        # ── Stop here if below threshold ─────────────────────────────────
        if DECISION_RANK.get(decision, 0) < DECISION_RANK.get(MIN_DECISION_FOR_DOWNSTREAM, 3):
            print(f"  ↩  Decision '{decision}' below threshold — skipping downstream agents")
            return

        # ── Step 3: Company Researcher ───────────────────────────────────
        print(f"  [3/5] Company Researcher: researching {company}...")
        company_brief = research_company(
            company_name=company,
            decision_level=decision,
            job_record=job_record,
        )

        if getattr(company_brief, "status", "ok") == "rejected":
            print(f"  [!] Company Researcher rejected — {getattr(company_brief, 'reason', '')}")
            company_brief = None
        else:
            cache_hit = getattr(company_brief, "cache_hit", False)
            ai_class  = getattr(getattr(company_brief, "ai_signal", None), "classification", "")
            print(f"  ✓  Brief ready {'(cached)' if cache_hit else '(fresh)'} | AI: {ai_class}")

        # ── Step 4: Interview Prep Coach ─────────────────────────────────
        print(f"  [4/5] Interview Prep Coach: building prep pack...")
        prep_pack = prep_interview(
            job_record=job_record,
            fit_score=fit_score,
            company_brief=company_brief,
            profile_path=PROFILE_PATH,
            knowledge_base_path=KB_PATH,
        )

        if getattr(prep_pack, "status", "ok") == "rejected":
            print(f"  [!] Interview Prep Coach rejected — {getattr(prep_pack, 'reason', '')}")
            prep_pack = None
        else:
            q_count = len(getattr(prep_pack, "question_bank", []) or [])
            t_count = len(getattr(prep_pack, "technical_questions", []) or [])
            print(f"  ✓  Prep pack: {q_count} questions, {t_count} technical")

        # ── Step 5: Interview Kit ─────────────────────────────────────────
        if prep_pack:
            print(f"  [5/5] Interview Kit: generating HTML...")
            safe_name = re.sub(r"[^\w]", "_", company.lower())[:25]
            kit_path  = os.path.join(output_dir, f"{safe_name}_interview_kit.html")
            generate_interview_kit(
                prep_pack=prep_pack,
                company_brief=company_brief,
                fit_score=fit_score,
                output_path=kit_path,
            )
            print(f"  ✓  Kit saved: {kit_path}")

        # ── Resume Strategist (optional — only for Apply) ─────────────────
        if decision == "apply":
            print(f"  [+] Resume Strategist: generating targeted rewrites...")
            try:
                strategy = strategize(fit_score, job_record, profile_path=PROFILE_PATH)
                if getattr(strategy, "status", "ok") != "rejected":
                    strat_path = os.path.join(output_dir, f"{safe_name}_resume_diff.json")
                    with open(strat_path, "w") as f:
                        json.dump(strategy.model_dump(), f, indent=2)
                    print(f"  ✓  Resume diff saved: {strat_path}")
            except Exception as e:
                print(f"  [!] Resume Strategist failed: {e}")

    except Exception as e:
        print(f"  [✗] Pipeline error for '{job_title}': {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Entry Point 1: Manual
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Profile fallback — build CandidateProfile from profile.yaml directly
# Used when resume.pdf is missing or invalid
# ---------------------------------------------------------------------------

def _build_profile_from_yaml(profile_path: str = PROFILE_PATH, prefs=None):
    """
    Build a minimal CandidateProfile from profile.yaml as fallback.
    Used when resume.pdf is missing or Profile Builder rejects the PDF.
    """
    try:
        with open(profile_path, "r") as f:
            data = yaml.safe_load(f) or {}

        identity = data.get("identity", {})
        skills_raw = data.get("skills", {})

        # Flatten all skills
        all_skills = []
        for category, skill_list in skills_raw.items():
            if isinstance(skill_list, list):
                all_skills.extend(skill_list)

        ai_skills = skills_raw.get("ai_genai", []) or []

        # Build a minimal CandidateProfile-compatible object
        from profile_builder import CandidateProfile, PreferencesConfig, TargetComp
        prefs = prefs or PreferencesConfig(
            work_arrangement_preference="remote",
            target_compensation=TargetComp(min=120000, max=150000, currency="CAD"),
        )

        # Map current_role to valid CandidateProfile enum value
        years_pm = identity.get("years_pm", 0)
        role = identity.get("current_role", "").lower()
        if "principal" in role or years_pm >= 10:
            level = "principal_pm"
        elif "staff" in role or years_pm >= 8:
            level = "staff_pm"
        elif "senior" in role or years_pm >= 5:
            level = "senior_pm"
        elif years_pm >= 2:
            level = "mid_pm"
        elif years_pm >= 1:
            level = "junior_pm"
        else:
            level = "unclassified"

        profile = CandidateProfile(
            candidate_name=identity.get("name", "Candidate"),
            current_level=level,
            years_total=identity.get("years_total_tech", 0),
            years_as_pm=float(identity.get("years_pm", 0)),
            skills=all_skills,
            ai_skills=ai_skills,
            unclassified_skills=[],
            industry_background=identity.get("target_roles", []),
            location=identity.get("location", ""),
            certifications=[c for c in data.get("education_and_certs", {}).get("certifications", [])],
            parse_warnings=[],
            work_arrangement_preference=prefs.work_arrangement_preference if prefs else "remote",
            target_compensation=prefs.target_compensation if prefs else None,
        )
        print(f"[Pipeline] Profile loaded from profile.yaml: {profile.candidate_name} | Level: {profile.current_level}")
        return profile
    except Exception as e:
        print(f"[Pipeline] Warning: profile.yaml fallback failed: {e}")
        return None


def run_manual(
    jd_text: str,
    job_url: str = "",
    profile_path: str = PROFILE_PATH,
):
    """
    Run the pipeline on a single manually-provided JD.

    Args:
        jd_text:      Full JD text (paste from the job posting).
        job_url:      Job URL (LinkedIn, Indeed, etc.). Optional but recommended.
        profile_path: Path to profile.yaml.
    """
    print("\n" + "="*60)
    print("JOBHUNTER AI CREW — Manual Run")
    print("="*60)

    if not jd_text or not jd_text.strip():
        print("[Error] jd_text is required for manual mode.")
        return

    # Load profile once — try PDF first, fall back to profile.yaml
    print("\n[Pipeline] Loading candidate profile...")
    prefs = PreferencesConfig(
        work_arrangement_preference="remote",
        target_compensation=TargetComp(min=120000, max=150000, currency="CAD"),
    )
    candidate_profile = None
    if os.path.exists(RESUME_PDF_PATH):
        candidate_profile = build_profile(RESUME_PDF_PATH, prefs)
        if getattr(candidate_profile, "status", "ok") == "rejected":
            print(f"[Pipeline] PDF profile rejected ({getattr(candidate_profile, 'reason', '')}), falling back to profile.yaml")
            candidate_profile = None
    if candidate_profile is None:
        candidate_profile = _build_profile_from_yaml(PROFILE_PATH, prefs)
    if candidate_profile is None:
        print("[Pipeline] Could not build candidate profile. Check profile.yaml exists.")
        return

    results = []
    _run_single_job(jd_text, job_url, candidate_profile, results)

    # Generate batch report (even for single job)
    if results:
        print(f"\n[Pipeline] Generating batch report...")
        generate_report(results, output_path=BATCH_REPORT)
        print(f"[Pipeline] Report saved: {BATCH_REPORT}")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


# ---------------------------------------------------------------------------
# Entry Point 2: Automated
# ---------------------------------------------------------------------------

def run_automated(
    profile_path: str = PROFILE_PATH,
    sources: list[str] = None,
    search_terms: list[str] = None,
    location: str = DEFAULT_LOCATION,
    process_queue: bool = True,
):
    """
    Run intake from Gmail alerts + Indeed RSS, then process queued jobs.

    Args:
        profile_path:  Path to profile.yaml.
        sources:       ["gmail", "indeed_rss"]. Default: both.
        search_terms:  Job title search terms. Default: DEFAULT_SEARCH_TERMS.
        location:      Location filter. Default: "Canada".
        process_queue: If True, also process the existing queue (not just new jobs).
    """
    print("\n" + "="*60)
    print("JOBHUNTER AI CREW — Automated Run")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    sources = sources or ["gmail", "indeed_rss"]
    search_terms = search_terms or DEFAULT_SEARCH_TERMS

    # ── Step 1: Intake ────────────────────────────────────────────────────
    print(f"\n[Intake] Sources: {', '.join(sources)}")
    print(f"[Intake] Search terms: {', '.join(search_terms)}")
    print(f"[Intake] Location: {location}")

    newly_queued = run_intake(sources, search_terms, location)

    # ── Step 2: Load full queue (new + previously unprocessed) ────────────
    queue = _load_queue()
    if not queue:
        print("\n[Pipeline] Queue is empty — nothing to process.")
        return

    print(f"\n[Pipeline] Processing {len(queue)} jobs from queue...")

    # ── Step 3: Load profile once ─────────────────────────────────────────
    prefs = PreferencesConfig(
        work_arrangement_preference="remote",
        target_compensation=TargetComp(min=120000, max=150000, currency="CAD"),
    )
    candidate_profile = build_profile(RESUME_PDF_PATH, prefs)
    print(f"[Pipeline] Profile: {getattr(candidate_profile, 'name', '')} loaded")

    # ── Step 4: Process each queued job ───────────────────────────────────
    results = []
    processed = []

    for i, job in enumerate(queue):
        title   = job.get("title", "unknown")
        url     = job.get("url", "")
        desc    = job.get("description", "")
        source  = job.get("source", "")

        print(f"\n[{i+1}/{len(queue)}] {title}")
        print(f"         URL: {url[:70]}...")

        if not desc and not url:
            print("  [!] No JD text or URL — skipping")
            continue

        # Use description as JD text if no separate text
        jd_text = desc if desc else f"Job Title: {title}\nURL: {url}"

        _run_single_job(jd_text, url, candidate_profile, results)
        processed.append(url)

        # Small delay between jobs to avoid rate limits
        if i < len(queue) - 1:
            time.sleep(2)

    # ── Step 5: Clear processed jobs from queue ───────────────────────────
    remaining = [j for j in queue if j.get("url") not in processed]
    _save_queue(remaining)
    print(f"\n[Pipeline] Queue: {len(processed)} processed, {len(remaining)} remaining")

    # ── Step 6: Batch report ──────────────────────────────────────────────
    if results:
        print(f"\n[Pipeline] Generating batch report for {len(results)} scored jobs...")
        generate_report(results, output_path=BATCH_REPORT)
        print(f"[Pipeline] Report saved: {BATCH_REPORT}")

    print("\n" + "="*60)
    print(f"DONE — {len(results)} jobs scored, {len([r for r in results if getattr(r, 'decision', '') in ('apply', 'conditional_apply')])} worth pursuing")
    print("="*60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JobHunter AI Crew — run the full job search pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Manual: paste a JD
  python run_pipeline.py --manual --jd "Senior AI PM at Stripe..." --url "https://..."

  # Manual: read JD from file
  python run_pipeline.py --manual --jd-file jd.txt --url "https://..."

  # Automated: run intake + pipeline
  python run_pipeline.py --auto

  # Automated: just intake, don't process yet
  python run_pipeline.py --auto --intake-only

  # Automated: dry run (show what would be queued)
  python run_pipeline.py --auto --dry-run
        """
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manual", action="store_true", help="Manual mode: provide a JD")
    mode.add_argument("--auto",   action="store_true", help="Automated mode: read alerts + process queue")

    # Manual options
    parser.add_argument("--jd",      type=str, help="JD text (for --manual)")
    parser.add_argument("--jd-file", type=str, help="Path to .txt file with JD (for --manual)")
    parser.add_argument("--url",     type=str, default="", help="Job URL (for --manual)")

    # Automated options
    parser.add_argument("--sources",      nargs="+", default=["gmail", "indeed_rss"],
                        help="Intake sources: gmail, indeed_rss (default: both)")
    parser.add_argument("--search-terms", nargs="+", default=DEFAULT_SEARCH_TERMS,
                        help="Job title search terms for RSS")
    parser.add_argument("--location",     default=DEFAULT_LOCATION,
                        help="Location filter for RSS")
    parser.add_argument("--intake-only",  action="store_true",
                        help="Only run intake, don't process pipeline")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Show what would be queued without queuing")

    # Common options
    parser.add_argument("--profile", default=PROFILE_PATH, help="Path to profile.yaml")
    parser.add_argument("--resume", default=RESUME_PDF_PATH, help="Path to resume PDF for Profile Builder")

    args = parser.parse_args()

    if args.manual:
        # Get JD text
        jd_text = ""
        if args.jd_file:
            with open(args.jd_file, "r") as f:
                jd_text = f.read()
        elif args.jd:
            jd_text = args.jd
        else:
            print("Error: --manual requires --jd or --jd-file")
            sys.exit(1)

        RESUME_PDF_PATH = args.resume
        run_manual(jd_text=jd_text, job_url=args.url or "", profile_path=args.profile)

    elif args.auto:
        if args.intake_only or args.dry_run:
            run_intake(
                sources=args.sources,
                search_terms=args.search_terms,
                location=args.location,
                dry_run=args.dry_run,
            )
        else:
            run_automated(
                profile_path=args.profile,
                sources=args.sources,
                search_terms=args.search_terms,
                location=args.location,
            )
