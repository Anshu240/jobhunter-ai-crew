"""
Interview Kit — JobHunter AI Crew.

Generates a per-job interview kit: one clean HTML document you open the
night before the interview. Four tabs:
  1. Company Brief  — history, AI signal, culture, PM talking points, smart questions
  2. Question Bank  — all predicted questions with model answers, time budgets, follow-ups
  3. Technical      — technical questions routed by gap type
  4. Executive      — director/vp_cpo/ceo prep (only when detected)

Design: clean editorial — readable at 11pm, not a data dashboard.
Fonts: Lora (headings) + Karla (body). Warm white background.
Amber highlights for must-prepare questions. Color-coded gap badges.

Usage:
    from interview_kit import generate_interview_kit

    kit_path = generate_interview_kit(
        prep_pack=pack,
        company_brief=brief,
        fit_score=score,
        output_path="babylist_interview_kit.html",
    )
    print(f"Kit ready: {kit_path}")
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import html
import os
from datetime import datetime
from typing import Optional

try:
    from interview_prep_coach import InterviewPrepPack, PrepQuestion, TechnicalQuestion
except ImportError:
    InterviewPrepPack = None  # type: ignore

try:
    from company_researcher import CompanyBrief
except ImportError:
    CompanyBrief = None  # type: ignore

try:
    from fit_scorer import FitScoreResult
except ImportError:
    FitScoreResult = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(s or ""), quote=True)


def _likelihood_badge(likelihood: str) -> str:
    colors = {"high": "#2D6A4F", "likely": "#1D4ED8", "possible": "#6B7280"}
    color = colors.get(likelihood, "#6B7280")
    return f'<span class="badge" style="background:{color}20;color:{color};border:1px solid {color}40">{_esc(likelihood)}</span>'


def _gap_badge(gap_type: str) -> str:
    if gap_type == "hard_gap":
        return '<span class="badge" style="background:#FEE2E2;color:#DC2626;border:1px solid #FCA5A5">hard gap</span>'
    if gap_type == "soft_gap":
        return '<span class="badge" style="background:#FEF3C7;color:#D97706;border:1px solid #FCD34D">soft gap</span>'
    return '<span class="badge" style="background:#D1FAE5;color:#065F46;border:1px solid #6EE7B7">experience</span>'


def _ai_signal_badge(classification: str) -> str:
    configs = {
        "ai_core":    ("#7C3AED", "🧠 AI Core"),
        "ai_enabled": ("#1D4ED8", "⚡ AI Enabled"),
        "non_ai":     ("#6B7280", "○ Non-AI"),
        "unknown":    ("#9CA3AF", "? Unknown"),
    }
    color, label = configs.get(classification, ("#9CA3AF", classification))
    return f'<span class="ai-badge" style="background:{color}15;color:{color};border:2px solid {color}40">{label}</span>'


def _decision_badge(decision: str) -> str:
    configs = {
        "apply":            ("#2D6A4F", "#D1FAE5", "✓ Apply"),
        "conditional_apply": ("#B45309", "#FEF3C7", "⚡ Conditional Apply"),
        "maybe":            ("#4338CA", "#EEF2FF", "~ Maybe"),
        "skip":             ("#6B7280", "#F3F4F6", "✗ Skip"),
    }
    color, bg, label = configs.get(decision, ("#6B7280", "#F3F4F6", decision))
    return f'<span class="decision-badge" style="background:{bg};color:{color};border:1px solid {color}40">{label}</span>'


def _category_icon(category: str) -> str:
    return {
        "behavioral":       "💬",
        "behavioral_ai":    "🤖",
        "product_sense":    "🏗️",
        "metrics":          "📊",
        "technical":        "⚙️",
        "company_specific": "🏢",
        "executive":        "👔",
    }.get(category, "❓")


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_company_tab(brief) -> str:
    if brief is None:
        return """
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <h3>No Company Brief Available</h3>
            <p>Run the Company Researcher agent before generating this kit.</p>
            <code>brief = research_company(company_name, decision_level="apply")</code>
        </div>"""

    sections = getattr(brief, "sections", None)
    ai_sig   = getattr(brief, "ai_signal", None)
    pts      = getattr(brief, "pm_talking_points", []) or []
    smart_qs = getattr(brief, "smart_questions", []) or []
    warnings = getattr(brief, "research_warnings", []) or []
    coverage = getattr(brief, "platform_coverage", None)
    cache    = getattr(brief, "cache_hit", False)
    conf     = getattr(brief, "confidence_overall", "")
    res_date = getattr(brief, "research_date", "")

    ai_classification = getattr(ai_sig, "classification", "unknown") if ai_sig else "unknown"
    evidence_list     = getattr(ai_sig, "evidence", []) if ai_sig else []

    parts = []

    # AI signal banner
    parts.append(f"""
    <div class="ai-signal-banner">
        {_ai_signal_badge(ai_classification)}
        <span class="conf-label">Confidence: <strong>{_esc(conf)}</strong></span>
        {"<span class='cache-label'>📦 Cached " + _esc(res_date) + "</span>" if cache else ""}
    </div>""")

    # Warnings
    if warnings:
        warn_html = " · ".join(f'<span class="warn-tag">{_esc(w)}</span>' for w in warnings)
        parts.append(f'<div class="warnings-bar">⚠️ {warn_html}</div>')

    # AI evidence
    if evidence_list:
        ev_items = ""
        type_order = {"product_delivered": 0, "product_planned": 1, "hiring_signal": 2, "engineering_blog": 3, "marketing_copy": 4}
        sorted_ev = sorted(evidence_list, key=lambda e: type_order.get(getattr(e, "type", ""), 5))
        for ev in sorted_ev[:5]:
            ev_type = getattr(ev, "type", "")
            ev_signal = getattr(ev, "signal", "")
            ev_source = getattr(ev, "source", "")
            ev_date   = getattr(ev, "date", "")
            type_colors = {"product_delivered": "#2D6A4F", "product_planned": "#1D4ED8",
                           "hiring_signal": "#7C3AED", "engineering_blog": "#0891B2", "marketing_copy": "#9CA3AF"}
            tc = type_colors.get(ev_type, "#6B7280")
            source_link = f'<a href="{_esc(ev_source)}" target="_blank" class="source-link">↗</a>' if ev_source else ""
            ev_items += f"""
            <div class="ev-item">
                <span class="ev-type" style="color:{tc}">{_esc(ev_type.replace('_',' '))}</span>
                <span class="ev-signal">{_esc(ev_signal)}</span>
                {source_link}
                {"<span class='ev-date'>" + _esc(ev_date) + "</span>" if ev_date else ""}
            </div>"""
        parts.append(f'<div class="section-card"><h3>AI Signals</h3><div class="ev-list">{ev_items}</div></div>')

    # Brief sections
    section_configs = [
        ("history",           "📖 History"),
        ("culture_and_people","👥 Culture & People"),
        ("product_and_technology", "⚙️ Product & Technology"),
        ("recent_developments", "📰 Recent Developments"),
        ("current_stance",    "🎯 Current Stance"),
    ]
    sections_html = ""
    for attr, label in section_configs:
        sec = getattr(sections, attr, None) if sections else None
        if not sec or not getattr(sec, "stated", False):
            continue
        para   = getattr(sec, "paragraph", "") or ""
        claims = getattr(sec, "key_claims", []) or []
        claim_items = ""
        for c in claims[:3]:
            c_text   = getattr(c, "claim", "")
            c_source = getattr(c, "source", "")
            c_date   = getattr(c, "date", "")
            src_link = f'<a href="{_esc(c_source)}" target="_blank" class="source-link">↗</a>' if c_source else ""
            claim_items += f'<li class="claim-item">{_esc(c_text)}{src_link}{"<span class=\'claim-date\'>" + _esc(c_date) + "</span>" if c_date else ""}</li>'

        # Glassdoor
        extra = ""
        if attr == "culture_and_people":
            rating = getattr(sec, "glassdoor_rating", "")
            pos    = getattr(sec, "glassdoor_positives", "")
            neg    = getattr(sec, "glassdoor_negatives", "")
            gs     = getattr(sec, "glassdoor_source", "")
            if rating or pos or neg:
                glink = f'<a href="{_esc(gs)}" target="_blank" class="source-link">Glassdoor ↗</a>' if gs else "Glassdoor"
                extra = f"""<div class="glassdoor-row">
                    {f'<span class="gd-rating">⭐ {_esc(rating)}</span>' if rating else ""}
                    {f'<span class="gd-pos">✓ {_esc(pos)}</span>' if pos else ""}
                    {f'<span class="gd-neg">⚠ {_esc(neg)}</span>' if neg else ""}
                    <span class="gd-source">{glink}</span>
                </div>"""

        # GitHub signal
        if attr == "product_and_technology":
            gh_sig = getattr(sec, "github_signal", "")
            gh_src = getattr(sec, "github_source", "")
            if gh_sig:
                gh_link = f'<a href="{_esc(gh_src)}" target="_blank" class="source-link">GitHub ↗</a>' if gh_src else ""
                extra += f'<div class="github-row">⚡ {_esc(gh_sig)} {gh_link}</div>'

        sections_html += f"""
        <div class="brief-section">
            <h4>{label}</h4>
            <p>{_esc(para)}</p>
            {extra}
            {"<ul class='claim-list'>" + claim_items + "</ul>" if claim_items else ""}
        </div>"""

    if sections_html:
        parts.append(f'<div class="sections-grid">{sections_html}</div>')

    # PM Talking Points
    if pts:
        pt_items = "".join(f'<li class="pt-item"><span class="pt-num">{i+1}</span>{_esc(p)}</li>' for i, p in enumerate(pts))
        parts.append(f'<div class="section-card amber-card"><h3>💡 PM Talking Points</h3><ul class="pt-list">{pt_items}</ul></div>')

    # Smart questions
    if smart_qs:
        sq_items = "".join(f'<li class="sq-item">"{_esc(q)}"</li>' for q in smart_qs)
        parts.append(f'<div class="section-card"><h3>🙋 Smart Questions to Ask Them</h3><ul class="sq-list">{sq_items}</ul></div>')

    return "\n".join(parts)


def _render_question_card(q: "PrepQuestion", idx: int, highlight: bool = False) -> str:
    border = "border-left: 3px solid #E8A045;" if highlight else ""
    story_warn = ""
    if getattr(q, "story_missing", False):
        story_warn = '<div class="story-warn">⚠️ No story mapped — use placeholder and find a real experience before the interview.</div>'

    ma = getattr(q, "model_answer", None)
    answer_html = ""
    if ma:
        for label, attr in [("Situation", "situation"), ("Action", "action"), ("Result", "result"), ("Demonstrates", "demonstrates")]:
            val = getattr(ma, attr, "")
            if val:
                answer_html += f'<div class="answer-row"><span class="answer-label">{label}</span><span class="answer-val">{_esc(val)}</span></div>'

    followups = getattr(q, "anticipated_followups", []) or []
    fu_html = ""
    if followups:
        fu_items = "".join(f'<li class="fu-item">"{_esc(f)}"</li>' for f in followups)
        fu_html = f'<div class="followups"><h5>Anticipated Follow-ups</h5><ul>{fu_items}</ul></div>'

    coaching = getattr(q, "coaching_note", "") or ""
    framework = getattr(q, "framework", "") or ""
    framework_hint = getattr(q, "framework_hint", "") or ""
    time_budget = getattr(q, "time_budget_minutes", "") or ""
    category = getattr(q, "category", "") or ""
    likelihood = getattr(q, "likelihood", "") or ""
    rec_story = getattr(q, "recommended_story", "") or ""
    alt_story  = getattr(q, "alternative_story", "") or ""

    story_badge = f'<span class="story-badge">📖 {_esc(rec_story)}</span>' if rec_story else ""
    alt_badge   = f'<span class="story-badge alt">📖 alt: {_esc(alt_story)}</span>' if alt_story else ""

    return f"""
    <div class="q-card" id="q-{idx}" style="{border}">
        <div class="q-header" onclick="toggleQ('q-{idx}')">
            <div class="q-meta">
                {_likelihood_badge(likelihood)}
                <span class="cat-badge">{_category_icon(category)} {_esc(category.replace('_',' '))}</span>
                {"<span class='time-badge'>⏱ " + _esc(time_budget) + "</span>" if time_budget else ""}
            </div>
            <div class="q-text">{_esc(getattr(q, 'question', ''))}</div>
            <div class="q-footer-meta">
                {story_badge}{alt_badge}
                {"<span class='framework-tag'>📐 " + _esc(framework) + "</span>" if framework else ""}
            </div>
        </div>
        <div class="q-body" style="display:none">
            {story_warn}
            {"<div class='framework-hint'>📐 " + _esc(framework_hint) + "</div>" if framework_hint else ""}
            <div class="model-answer">{answer_html}</div>
            {fu_html}
            <div class="coaching-note">🎯 {_esc(coaching)}</div>
        </div>
    </div>"""


def _render_questions_tab(pack: "InterviewPrepPack") -> str:
    must_ids = set(getattr(pack, "must_prepare_first", []) or [])
    all_qs   = getattr(pack, "question_bank", []) or []
    comp_qs  = getattr(pack, "company_specific_questions", []) or []
    smart_qs = getattr(pack, "smart_questions_to_ask", []) or []

    parts = []

    # Must prepare banner
    if must_ids:
        ids_str = " · ".join(f"<strong>{_esc(i)}</strong>" for i in list(must_ids)[:5])
        parts.append(f'<div class="must-prepare-banner">⭐ Must Prepare First: {ids_str}</div>')

    # Estimated time
    est = getattr(pack, "estimated_prep_time_hours", 0)
    if est:
        parts.append(f'<div class="prep-time-bar">⏱ Estimated prep time: <strong>{est} hours</strong></div>')

    # Questions
    if not all_qs:
        parts.append('<div class="empty-state"><p>No questions generated.</p></div>')
    else:
        for i, q in enumerate(all_qs):
            qid      = getattr(q, "id", str(i))
            highlight = qid in must_ids
            parts.append(_render_question_card(q, i, highlight))

    # Company specific
    if comp_qs:
        parts.append('<h3 class="section-divider">🏢 Company-Specific Questions</h3>')
        for i, q in enumerate(comp_qs):
            parts.append(_render_question_card(q, 1000 + i))

    # Smart questions to ask
    if smart_qs:
        sq_items = "".join(f'<li class="sq-item">"{_esc(q)}"</li>' for q in smart_qs)
        parts.append(f'<div class="section-card"><h3>🙋 Smart Questions to Ask Them</h3><ul class="sq-list">{sq_items}</ul></div>')

    return "\n".join(parts)


def _render_technical_tab(pack: "InterviewPrepPack") -> str:
    tech_qs = getattr(pack, "technical_questions", []) or []
    if not tech_qs:
        return '<div class="empty-state"><div class="empty-icon">⚙️</div><p>No technical questions generated for this role.</p></div>'

    grouped = {"none": [], "soft_gap": [], "hard_gap": []}
    for q in tech_qs:
        gt = getattr(q, "gap_type", "none") or "none"
        grouped.setdefault(gt, []).append(q)

    parts = []
    section_configs = [
        ("none",      "✓ Your Experience",     "#D1FAE5", "#065F46"),
        ("soft_gap",  "⚡ Soft Gaps",           "#FEF3C7", "#B45309"),
        ("hard_gap",  "⚠️ Hard Gaps — Be Honest", "#FEE2E2", "#DC2626"),
    ]

    for gt, label, bg, color in section_configs:
        qs = grouped.get(gt, [])
        if not qs:
            continue

        cards = ""
        for i, q in enumerate(qs):
            question   = getattr(q, "question", "")
            likelihood = getattr(q, "likelihood", "")
            your_exp   = getattr(q, "your_experience", "") or ""
            concept    = getattr(q, "concept_framework", "") or ""
            bridge     = getattr(q, "bridge_answer", "") or ""
            coaching   = getattr(q, "coaching_note", "") or ""
            time_bud   = getattr(q, "time_budget_minutes", "") or ""
            followups  = getattr(q, "anticipated_followups", []) or []

            fu_html = ""
            if followups:
                fu_items = "".join(f'<li class="fu-item">"{_esc(f)}"</li>' for f in followups)
                fu_html = f'<div class="followups"><h5>Anticipated Follow-ups</h5><ul>{fu_items}</ul></div>'

            cards += f"""
            <div class="t-card" id="tq-{gt}-{i}">
                <div class="q-header" onclick="toggleQ('tq-{gt}-{i}')">
                    <div class="q-meta">
                        {_likelihood_badge(likelihood)}
                        {_gap_badge(gt)}
                        {"<span class='time-badge'>⏱ " + _esc(time_bud) + "</span>" if time_bud else ""}
                    </div>
                    <div class="q-text">{_esc(question)}</div>
                </div>
                <div class="q-body" style="display:none">
                    {"<div class='tech-exp'><strong>Your Experience:</strong> " + _esc(your_exp) + "</div>" if your_exp else ""}
                    {"<div class='tech-concept'><strong>Concept:</strong> " + _esc(concept) + "</div>" if concept else ""}
                    {"<div class='tech-bridge'><strong>Bridge Answer:</strong> " + _esc(bridge) + "</div>" if bridge else ""}
                    {fu_html}
                    <div class="coaching-note">🎯 {_esc(coaching)}</div>
                </div>
            </div>"""

        parts.append(f"""
        <div class="tech-group" style="border-left:3px solid {color}">
            <h3 class="tech-group-label" style="color:{color}">{label}</h3>
            {cards}
        </div>""")

    return "\n".join(parts)


def _render_executive_tab(pack: "InterviewPrepPack") -> str:
    exec_prep = getattr(pack, "executive_prep", None)
    if not exec_prep:
        return '<div class="empty-state"><p>No executive prep for this role.</p></div>'

    director = getattr(exec_prep, "director", []) or []
    vp_cpo   = getattr(exec_prep, "vp_cpo", []) or []
    ceo      = getattr(exec_prep, "ceo", []) or []

    if not any([director, vp_cpo, ceo]):
        return '<div class="empty-state"><div class="empty-icon">👔</div><p>No senior-level interviewers detected for this role.</p></div>'

    parts = ['<div class="exec-header">💡 Executive rounds: crisp strategic answers only — no long SPAR structures. Lead with conviction.</div>']

    sections = [
        (director, "Director",  "#1D4ED8"),
        (vp_cpo,   "VP / CPO",  "#7C3AED"),
        (ceo,      "CEO / CTO", "#DC2626"),
    ]

    for qs, label, color in sections:
        if not qs:
            continue
        cards = ""
        for i, q in enumerate(qs):
            question  = getattr(q, "question", "")
            structure = getattr(q, "answer_structure", "") or ""
            coaching  = getattr(q, "coaching_note", "") or ""
            cards += f"""
            <div class="e-card" id="eq-{label.replace(' ','')}-{i}">
                <div class="q-header" onclick="toggleQ('eq-{label.replace(' ','')}-{i}')">
                    <div class="q-text">{_esc(question)}</div>
                </div>
                <div class="q-body" style="display:none">
                    {"<div class='exec-structure'><strong>Answer Structure:</strong><br>" + _esc(structure) + "</div>" if structure else ""}
                    <div class="coaching-note">🎯 {_esc(coaching)}</div>
                </div>
            </div>"""
        parts.append(f'<div class="exec-group" style="border-top:3px solid {color}"><h3 style="color:{color}">{label}</h3>{cards}</div>')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main HTML generator
# ---------------------------------------------------------------------------

def _generate_html(pack, brief, score) -> str:
    job_title = _esc(getattr(pack, "job_title", "Job Title") if pack else "Job Title")
    company   = _esc(getattr(pack, "company", "Company") if pack else "Company")
    decision  = getattr(pack, "decision_level", "") if pack else ""
    warnings  = getattr(pack, "prep_warnings", []) if pack else []

    score_val  = getattr(score, "total_score", 0) if score else 0
    score_html = f'<span class="score-val">{score_val:.0f}%</span>' if score_val else ""

    company_html  = _render_company_tab(brief)
    questions_html = _render_questions_tab(pack) if pack else ""
    technical_html = _render_technical_tab(pack) if pack else ""
    executive_html = _render_executive_tab(pack) if pack else ""

    warn_html = ""
    if warnings:
        warn_items = " · ".join(f'<span class="warn-tag">{_esc(w)}</span>' for w in warnings)
        warn_html  = f'<div class="top-warnings">⚠️ {warn_items}</div>'

    gen_date = datetime.now().strftime("%B %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{company} — Interview Kit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Karla:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:        #FAFAF8;
  --card:      #FFFFFF;
  --border:    #E5E0D8;
  --text:      #1C1917;
  --muted:     #78716C;
  --accent:    #1A1A2E;
  --amber:     #E8A045;
  --amber-bg:  #FFFBF0;
  --green:     #2D6A4F;
  --red:       #DC2626;
  --radius:    8px;
  --shadow:    0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Karla', sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 15px;
  line-height: 1.65;
}}
a {{ color: var(--accent); }}
a:hover {{ opacity: .7; }}

/* ── Header ── */
.kit-header {{
  background: var(--accent);
  color: white;
  padding: 28px 32px 24px;
  position: sticky; top: 0; z-index: 100;
}}
.kit-header h1 {{
  font-family: 'Lora', serif;
  font-size: 1.6rem;
  font-weight: 600;
  letter-spacing: -.01em;
}}
.kit-header h2 {{
  font-family: 'Karla', sans-serif;
  font-size: .95rem;
  font-weight: 400;
  opacity: .75;
  margin-top: 2px;
}}
.header-row {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}}
.header-meta {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 10px; }}
.score-val {{ font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 500; color: var(--amber); }}
.gen-date {{ font-size: .8rem; opacity: .55; margin-top: 4px; }}
.decision-badge {{
  display: inline-flex; align-items: center;
  padding: 3px 10px; border-radius: 99px;
  font-size: .8rem; font-weight: 600;
}}

/* ── Tab navigation ── */
.tabs {{
  background: white;
  border-bottom: 1px solid var(--border);
  display: flex;
  padding: 0 24px;
  position: sticky; top: 0; z-index: 99;
  box-shadow: var(--shadow);
}}
.tab-btn {{
  background: none; border: none; cursor: pointer;
  font-family: 'Karla', sans-serif;
  font-size: .9rem; font-weight: 500;
  color: var(--muted);
  padding: 14px 18px;
  border-bottom: 2px solid transparent;
  transition: all .15s;
}}
.tab-btn:hover {{ color: var(--text); }}
.tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}

/* ── Content ── */
.tab-content {{ display: none; padding: 28px 32px; max-width: 920px; margin: 0 auto; }}
.tab-content.active {{ display: block; }}

/* ── Cards & sections ── */
.section-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}}
.amber-card {{ background: var(--amber-bg); border-color: #F6D490; }}
.sections-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
@media (max-width: 680px) {{ .sections-grid {{ grid-template-columns: 1fr; }} }}
.brief-section {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  box-shadow: var(--shadow);
}}
.brief-section h4 {{
  font-family: 'Lora', serif;
  font-size: .95rem;
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--accent);
}}
.brief-section p {{ font-size: .9rem; color: var(--muted); line-height: 1.6; }}

/* ── AI Signal ── */
.ai-signal-banner {{
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 16px;
}}
.ai-badge {{
  padding: 5px 14px; border-radius: 99px;
  font-size: .85rem; font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
}}
.conf-label {{ font-size: .85rem; color: var(--muted); }}
.cache-label {{ font-size: .8rem; color: var(--muted); margin-left: auto; }}
.warnings-bar {{
  background: #FEF3C7; border: 1px solid #FCD34D;
  border-radius: var(--radius); padding: 8px 14px;
  margin-bottom: 16px; font-size: .85rem; color: #B45309;
}}
.warn-tag {{
  background: #FDE68A; padding: 1px 7px;
  border-radius: 4px; font-size: .8rem; margin: 0 2px;
}}

/* ── Evidence list ── */
.ev-list {{ display: flex; flex-direction: column; gap: 6px; }}
.ev-item {{ display: flex; align-items: center; gap: 8px; font-size: .88rem; }}
.ev-type {{ font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; min-width: 120px; }}
.ev-signal {{ flex: 1; }}
.ev-date {{ font-size: .78rem; color: var(--muted); }}
.source-link {{ font-size: .8rem; color: var(--accent); text-decoration: none; margin-left: 4px; }}

/* ── Claims / Glassdoor ── */
.claim-list {{ margin-top: 8px; padding-left: 0; list-style: none; }}
.claim-item {{ font-size: .85rem; color: var(--muted); padding: 3px 0; border-bottom: 1px solid #F5F0E8; }}
.claim-date {{ font-size: .78rem; color: #B0A898; margin-left: 6px; }}
.glassdoor-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; font-size: .82rem; }}
.gd-rating {{ color: #B45309; font-weight: 600; }}
.gd-pos {{ color: var(--green); }}
.gd-neg {{ color: var(--red); }}
.gd-source {{ margin-left: auto; }}
.github-row {{ margin-top: 8px; font-size: .85rem; color: #4338CA; }}

/* ── PM Talking Points ── */
.pt-list {{ list-style: none; display: flex; flex-direction: column; gap: 8px; }}
.pt-item {{ display: flex; align-items: flex-start; gap: 12px; font-size: .92rem; }}
.pt-num {{ background: var(--amber); color: white; border-radius: 50%; width: 22px; height: 22px;
           display: flex; align-items: center; justify-content: center; font-size: .75rem; font-weight: 700; flex-shrink: 0; }}
.sq-list {{ list-style: none; }}
.sq-item {{ padding: 8px 0; border-bottom: 1px solid var(--border); font-style: italic; font-size: .92rem; color: var(--muted); }}

/* ── Question cards ── */
.must-prepare-banner {{
  background: linear-gradient(135deg, #FFF8E8, #FFFBF4);
  border: 1px solid var(--amber);
  border-radius: var(--radius);
  padding: 12px 18px;
  margin-bottom: 20px;
  font-size: .9rem;
  color: #92400E;
}}
.prep-time-bar {{
  font-size: .88rem; color: var(--muted);
  margin-bottom: 16px; padding: 8px 0;
  border-bottom: 1px solid var(--border);
}}
.q-card, .t-card, .e-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 10px;
  box-shadow: var(--shadow);
  overflow: hidden;
}}
.q-header {{
  padding: 14px 18px;
  cursor: pointer;
  transition: background .1s;
}}
.q-header:hover {{ background: #F9F8F6; }}
.q-meta {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }}
.q-text {{ font-weight: 500; font-size: .95rem; line-height: 1.4; }}
.q-footer-meta {{ display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap; }}
.q-body {{ padding: 0 18px 16px; border-top: 1px solid var(--border); }}
.badge {{
  display: inline-flex; align-items: center;
  padding: 2px 9px; border-radius: 99px;
  font-size: .75rem; font-weight: 600;
}}
.cat-badge {{ background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; padding: 2px 9px; border-radius: 99px; font-size: .75rem; font-weight: 500; }}
.time-badge {{ background: #F0FDF4; color: #166534; border: 1px solid #BBF7D0; padding: 2px 9px; border-radius: 99px; font-size: .75rem; }}
.story-badge {{ background: #F5F3FF; color: #6D28D9; font-size: .75rem; padding: 2px 8px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; }}
.story-badge.alt {{ background: #FDF4FF; color: #9333EA; }}
.framework-tag {{ background: #FFF7ED; color: #C2410C; font-size: .75rem; padding: 2px 8px; border-radius: 4px; }}
.story-warn {{ background: #FEF3C7; border: 1px solid #FCD34D; border-radius: 4px; padding: 8px 12px; margin: 12px 0 8px; font-size: .85rem; color: #92400E; }}
.framework-hint {{ background: #FFF7ED; border-radius: 4px; padding: 8px 12px; margin: 12px 0 8px; font-size: .85rem; color: #C2410C; }}
.model-answer {{ display: flex; flex-direction: column; gap: 6px; margin: 12px 0; }}
.answer-row {{ display: flex; gap: 10px; font-size: .9rem; }}
.answer-label {{ font-weight: 600; min-width: 90px; color: var(--accent); font-size: .82rem; text-transform: uppercase; letter-spacing: .04em; padding-top: 2px; }}
.answer-val {{ flex: 1; color: var(--text); line-height: 1.5; }}
.followups {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }}
.followups h5 {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin-bottom: 6px; }}
.fu-item {{ font-style: italic; font-size: .88rem; color: var(--muted); padding: 3px 0; list-style: none; }}
.coaching-note {{
  margin-top: 12px; padding: 10px 14px;
  background: #F8F7F5; border-radius: 6px;
  font-size: .88rem; color: #44403C;
  border-left: 3px solid var(--amber);
}}

/* ── Technical ── */
.tech-group {{ padding: 0 0 8px 16px; margin-bottom: 20px; }}
.tech-group-label {{ font-family: 'Lora', serif; font-size: 1rem; margin-bottom: 12px; }}
.tech-exp, .tech-concept, .tech-bridge {{ font-size: .88rem; margin: 8px 0; padding: 8px 12px; border-radius: 4px; }}
.tech-exp {{ background: #D1FAE520; border-left: 2px solid var(--green); }}
.tech-concept {{ background: #EEF2FF; border-left: 2px solid #4338CA; }}
.tech-bridge {{ background: #FEF3C7; border-left: 2px solid #D97706; }}

/* ── Executive ── */
.exec-header {{
  background: #1A1A2E10; border: 1px solid #1A1A2E30;
  border-radius: var(--radius); padding: 12px 16px;
  margin-bottom: 20px; font-size: .9rem; color: var(--accent);
}}
.exec-group {{ padding-top: 16px; margin-bottom: 24px; }}
.exec-group h3 {{ font-family: 'Lora', serif; font-size: 1.05rem; margin-bottom: 12px; }}
.exec-structure {{ font-size: .9rem; margin: 10px 0; padding: 10px 14px; background: #F8F7F5; border-radius: 4px; }}

/* ── Section dividers ── */
.section-divider {{ font-family: 'Lora', serif; font-size: 1rem; color: var(--accent); margin: 24px 0 12px; padding-top: 16px; border-top: 1px solid var(--border); }}

/* ── Empty state ── */
.empty-state {{ text-align: center; padding: 48px 24px; color: var(--muted); }}
.empty-icon {{ font-size: 2.5rem; margin-bottom: 12px; }}
.empty-state h3 {{ font-family: 'Lora', serif; color: var(--text); margin-bottom: 8px; }}
.empty-state code {{ background: #F3F4F6; padding: 8px 14px; border-radius: 4px; font-size: .85rem; display: inline-block; margin-top: 12px; }}

/* ── Top warnings ── */
.top-warnings {{
  background: #FEF3C7; border-bottom: 1px solid #FCD34D;
  padding: 8px 32px; font-size: .85rem; color: #92400E;
}}
</style>
</head>
<body>

<header class="kit-header">
  <div class="header-row">
    <div>
      <h1>{company}</h1>
      <h2>{job_title}</h2>
      <div class="header-meta">
        {_decision_badge(decision)}
        {score_html}
      </div>
    </div>
    <div style="text-align:right">
      <div class="gen-date">Generated {gen_date}</div>
    </div>
  </div>
</header>

{warn_html}

<nav class="tabs">
  <button class="tab-btn active" onclick="showTab('company')">🏢 Company Brief</button>
  <button class="tab-btn" onclick="showTab('questions')">💬 Question Bank</button>
  <button class="tab-btn" onclick="showTab('technical')">⚙️ Technical</button>
  <button class="tab-btn" onclick="showTab('executive')">👔 Executive</button>
</nav>

<div id="tab-company" class="tab-content active">
  {company_html}
</div>

<div id="tab-questions" class="tab-content">
  {questions_html}
</div>

<div id="tab-technical" class="tab-content">
  {technical_html}
</div>

<div id="tab-executive" class="tab-content">
  {executive_html}
</div>

<script>
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.currentTarget.classList.add('active');
}}

function toggleQ(id) {{
  const card = document.getElementById(id);
  const body = card.querySelector('.q-body');
  body.style.display = body.style.display === 'none' ? 'block' : 'none';
}}

// Open must-prepare questions by default
document.addEventListener('DOMContentLoaded', () => {{
  document.querySelectorAll('.q-card').forEach((card, i) => {{
    if (i < 3) {{
      const body = card.querySelector('.q-body');
      if (body) body.style.display = 'block';
    }}
  }});
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_interview_kit(
    prep_pack=None,
    company_brief=None,
    fit_score=None,
    output_path: str = "",
) -> str:
    """
    Generate a per-job interview kit HTML file.

    Args:
        prep_pack:      InterviewPrepPack from Interview Prep Coach.
        company_brief:  CompanyBrief from Company Researcher (optional).
        fit_score:      FitScoreResult from Fit Scorer (optional).
        output_path:    Where to save the HTML. Defaults to "{company}_{job}_kit.html".

    Returns:
        Path to the generated HTML file.
    """
    if not output_path:
        company  = getattr(prep_pack, "company", "company") if prep_pack else "company"
        job      = getattr(prep_pack, "job_title", "job") if prep_pack else "job"
        safe     = lambda s: "".join(c if c.isalnum() else "_" for c in s.lower())[:30]
        output_path = f"{safe(company)}_{safe(job)}_interview_kit.html"

    html_content = _generate_html(prep_pack, company_brief, fit_score)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[InterviewKit] Generating demo kit with placeholder data...")
    path = generate_interview_kit(output_path="/mnt/user-data/outputs/demo_interview_kit.html")
    print(f"[InterviewKit] Demo kit saved: {path}")
    print("[InterviewKit] Pass prep_pack + company_brief + fit_score for a real kit.")
