"""
Presenter — JobHunter AI Crew.
Dark analytics dashboard edition.

No LLM calls. No API cost. Pure Python -> standalone HTML.
Animated score rings, glowing decision colors, dark navy aesthetic.

Keep in the same directory as fit_scorer.py.

Usage:
    from presenter import generate_report, attach_display_meta
    from fit_scorer import score_job

    results = [score_job(job, profile) for job in job_records if job.status == "ok"]
    scored  = [r for r in results if r.status == "ok"]
    generate_report(scored, output_path="report.html", run_label="June 4 2026")
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import html, math
from datetime import datetime
from typing import Optional

try:
    from fit_scorer import FitScoreResult, DimensionScores, DimensionScore
except ImportError:
    raise ImportError(
        "Could not import from fit_scorer.py. "
        "Make sure presenter.py and fit_scorer.py are in the same directory."
    )

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

DECISION_STYLES: dict[str, dict] = {
    "apply": {
        "color":   "#34D399",
        "glow":    "rgba(52,211,153,0.18)",
        "glow_strong": "rgba(52,211,153,0.35)",
        "border":  "rgba(52,211,153,0.35)",
        "label":   "APPLY",
        "symbol":  "✓",
        "dim":     "rgba(52,211,153,0.10)",
    },
    "conditional_apply": {
        "color":   "#FBBF24",
        "glow":    "rgba(251,191,36,0.15)",
        "glow_strong": "rgba(251,191,36,0.30)",
        "border":  "rgba(251,191,36,0.30)",
        "label":   "CONDITIONAL",
        "symbol":  "◐",
        "dim":     "rgba(251,191,36,0.08)",
    },
    "maybe": {
        "color":   "#818CF8",
        "glow":    "rgba(129,140,248,0.15)",
        "glow_strong": "rgba(129,140,248,0.30)",
        "border":  "rgba(129,140,248,0.30)",
        "label":   "MAYBE",
        "symbol":  "?",
        "dim":     "rgba(129,140,248,0.08)",
    },
    "skip": {
        "color":   "#4B5563",
        "glow":    "rgba(75,85,99,0.08)",
        "glow_strong": "rgba(75,85,99,0.15)",
        "border":  "rgba(75,85,99,0.25)",
        "label":   "SKIP",
        "symbol":  "✗",
        "dim":     "rgba(75,85,99,0.06)",
    },
}

DIMENSION_LABELS: dict[str, str] = {
    "required_skills":    "Required Skills",
    "ai_genai_depth":     "AI / GenAI Depth",
    "seniority_fit":      "Seniority Fit",
    "nice_to_have":       "Nice-to-Have",
    "work_arrangement":   "Work Arrangement",
    "industry_alignment": "Industry Alignment",
    "compensation":       "Compensation",
}

# ---------------------------------------------------------------------------
# CSS — dark analytics dashboard
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:          #060A14;
  --bg-card:     #0C1427;
  --bg-card-alt: #0F1A30;
  --bg-hover:    #111E38;
  --border:      rgba(148,163,184,0.08);
  --border-mid:  rgba(148,163,184,0.14);
  --text:        #CDD5E0;
  --text-bright: #E8EEF7;
  --text-mid:    #8896A8;
  --text-muted:  #4A5568;
  --navy-accent: #1E3A5F;
  --radius:      12px;
  --radius-sm:   8px;
  font-family: 'Outfit', sans-serif;
  font-size: 15px;
  color: var(--text);
  background: var(--bg);
  line-height: 1.6;
}

body { padding-bottom: 80px; }

/* ---- Page header ---- */
.report-header {
  background: linear-gradient(135deg, #060E22 0%, #0A1530 60%, #0D1E3A 100%);
  border-bottom: 1px solid var(--border-mid);
  padding: 36px 48px 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 44px;
  position: relative;
  overflow: hidden;
}
.report-header::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(52,211,153,0.06) 0%, transparent 70%);
  pointer-events: none;
}
.report-header h1 {
  font-family: 'Syne', sans-serif;
  font-size: 26px;
  font-weight: 800;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text-bright);
}
.header-sub {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 5px;
  letter-spacing: 0.5px;
  font-weight: 300;
}
.header-stats {
  display: flex;
  gap: 24px;
  text-align: center;
}
.stat-box { }
.stat-num {
  font-family: 'Syne', sans-serif;
  font-size: 22px;
  font-weight: 700;
  line-height: 1;
}
.stat-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--text-muted);
  margin-top: 3px;
  font-weight: 400;
}

/* ---- Section wrappers ---- */
.section { max-width: 1040px; margin: 0 auto 52px; padding: 0 24px; }
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(to right, var(--border), transparent);
}

/* ---- Dashboard grid ---- */
.dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }

/* Apply bucket — full width, hero treatment */
.dash-bucket-apply {
  grid-column: 1 / -1;
  background: linear-gradient(135deg, rgba(52,211,153,0.07) 0%, var(--bg-card) 60%);
  border: 1px solid rgba(52,211,153,0.25);
  border-radius: var(--radius);
  padding: 24px 28px;
  box-shadow: 0 0 40px rgba(52,211,153,0.06), inset 0 1px 0 rgba(52,211,153,0.08);
}
.dash-bucket {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
}

.dash-bucket-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
}
.dash-bucket-label {
  font-family: 'Syne', sans-serif;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.dash-count {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 20px;
  border: 1px solid;
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
}
.dash-empty {
  font-size: 13px;
  color: var(--text-muted);
  font-style: italic;
  padding: 4px 0;
}

/* Apply feature cards grid */
.apply-cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 12px;
}
.apply-feature-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(52,211,153,0.15);
  border-radius: var(--radius-sm);
  padding: 16px;
  display: flex;
  align-items: center;
  gap: 14px;
  transition: background 0.2s, border-color 0.2s, transform 0.2s;
}
.apply-feature-card:hover {
  background: rgba(52,211,153,0.06);
  border-color: rgba(52,211,153,0.30);
  transform: translateY(-1px);
}
.apply-card-ring { flex-shrink: 0; }
.apply-card-info { flex: 1; min-width: 0; }
.apply-card-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--text-bright);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 2px;
}
.apply-card-company {
  font-size: 12px;
  color: var(--text-muted);
}
.apply-card-links {
  display: flex;
  gap: 6px;
  margin-top: 8px;
  flex-wrap: wrap;
}

/* Compact job rows (non-Apply buckets) */
.dash-job {
  display: flex;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
  gap: 10px;
}
.dash-job:last-child { border-bottom: none; }
.dash-job-info { flex: 1; min-width: 0; }
.dash-job-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-bright);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.dash-job-company { font-size: 11.5px; color: var(--text-muted); margin-top: 1px; }
.dash-job-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  font-weight: 600;
  flex-shrink: 0;
}

/* Small link buttons */
.link-btn {
  display: inline-block;
  font-size: 10.5px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
  text-decoration: none;
  border: 1px solid;
  letter-spacing: 0.3px;
  transition: opacity 0.15s;
}
.link-btn:hover { opacity: 0.75; }
.link-btn-ghost {
  font-size: 11px;
  color: var(--text-muted);
  text-decoration: none;
  flex-shrink: 0;
}
.link-btn-ghost:hover { color: var(--text); }

/* ---- Score ring ---- */
.ring-wrap {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.ring-track { fill: none; stroke: rgba(255,255,255,0.05); }
.ring-progress {
  fill: none;
  stroke-linecap: round;
  stroke-dashoffset: var(--ring-max);
  animation: ring-draw 1.4s cubic-bezier(0.4,0,0.2,1) forwards;
  animation-delay: var(--ring-delay, 0.2s);
}
@keyframes ring-draw {
  from { stroke-dashoffset: var(--ring-max); }
  to   { stroke-dashoffset: var(--ring-offset); }
}
.ring-score-text {
  position: absolute;
  text-align: center;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
}
.ring-score-num {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  line-height: 1;
  display: block;
}
.ring-score-denom {
  font-size: 9px;
  color: var(--text-muted);
  display: block;
  margin-top: 1px;
}

/* ---- Decision badge ---- */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 20px;
  font-family: 'Syne', sans-serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  border: 1px solid;
}

/* ---- Mini-report cards ---- */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 20px;
  overflow: hidden;
  transition: border-color 0.2s, box-shadow 0.2s;
  animation: fadeUp 0.5s ease both;
}
.card:hover {
  border-color: var(--border-mid);
  box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.card-header {
  padding: 22px 28px 18px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 20px;
}
.card-title-block { flex: 1; min-width: 0; }
.card-title {
  font-family: 'Syne', sans-serif;
  font-size: 17px;
  font-weight: 700;
  color: var(--text-bright);
  line-height: 1.2;
  margin-bottom: 4px;
}
.card-company {
  font-size: 13px;
  color: var(--text-muted);
}
.card-company a {
  color: #60A5FA;
  text-decoration: none;
  font-weight: 500;
}
.card-company a:hover { text-decoration: underline; }
.card-meta {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 6px;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
}

/* Card body */
.card-body { padding: 22px 28px; }
.card-section { margin-bottom: 22px; }
.card-section:last-child { margin-bottom: 0; }
.card-section-title {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 10px;
}

/* Reasoning */
.reasoning {
  font-size: 14px;
  color: var(--text-mid);
  line-height: 1.75;
  font-style: italic;
  border-left: 2px solid var(--navy-accent);
  padding-left: 14px;
}

/* Condition */
.condition {
  font-size: 13px;
  background: rgba(251,191,36,0.08);
  border: 1px solid rgba(251,191,36,0.25);
  color: #FBBF24;
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  font-weight: 500;
}

/* Dimension score table */
.dim-table { width: 100%; border-collapse: collapse; }
.dim-table tr { border-bottom: 1px solid var(--border); }
.dim-table tr:last-child { border-bottom: none; }
.dim-table td { padding: 7px 0; vertical-align: middle; }
.dim-name { font-size: 12.5px; color: var(--text-mid); width: 170px; font-weight: 400; }
.dim-bar-wrap { padding: 0 14px; }
.dim-bar-bg { background: rgba(255,255,255,0.05); border-radius: 4px; height: 5px; overflow: hidden; }
.dim-bar-fill { height: 100%; border-radius: 4px; background: var(--bar-color, #34D399); }
.dim-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11.5px;
  font-weight: 600;
  text-align: right;
  width: 48px;
  color: var(--text);
}
.dim-detail { font-size: 11px; color: var(--text-muted); padding-left: 10px; font-style: italic; }

/* Skill pills */
.pills { display: flex; flex-wrap: wrap; gap: 5px; }
.pill {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 20px;
  font-size: 11.5px;
  font-weight: 500;
}
.pill-matched  { background: rgba(52,211,153,0.1);  color: #34D399; border: 1px solid rgba(52,211,153,0.25); }
.pill-missing  { background: rgba(239,68,68,0.08);  color: #F87171; border: 1px solid rgba(239,68,68,0.20); }
.pill-near     { background: rgba(129,140,248,0.1); color: #818CF8; border: 1px solid rgba(129,140,248,0.25); }
.pill-nth-ok   { background: rgba(52,211,153,0.07); color: #6EE7B7; border: 1px solid rgba(52,211,153,0.15); }
.pill-nth-miss { background: rgba(255,255,255,0.04); color: var(--text-muted); border: 1px solid var(--border); }

/* Flags */
.flags { display: flex; flex-wrap: wrap; gap: 5px; }
.flag {
  font-size: 11px;
  padding: 3px 9px;
  border-radius: 4px;
  background: rgba(251,191,36,0.07);
  color: #FBBF24;
  border: 1px solid rgba(251,191,36,0.20);
  font-weight: 500;
}

/* Divider */
.section-divider {
  max-width: 1040px;
  margin: 0 auto 44px;
  padding: 0 24px;
}
.section-divider hr { border: none; border-top: 1px solid var(--border); }

/* Empty state */
.empty-report { max-width: 500px; margin: 140px auto; text-align: center; }
.empty-report h2 { font-family:'Syne',sans-serif; font-size:20px; color:var(--text-mid); margin-bottom:8px; }
.empty-report p { color: var(--text-muted); font-size: 14px; }

/* Responsive */
@media (max-width: 720px) {
  .dashboard { grid-template-columns: 1fr; }
  .dash-bucket-apply { grid-column: auto; }
  .report-header { flex-direction: column; gap: 20px; align-items: flex-start; }
  .header-stats { gap: 16px; }
  .card-header { flex-direction: column; }
  .apply-cards-grid { grid-template-columns: 1fr; }
}
"""

# Inline JS for nothing — rings use pure CSS animation via custom properties
JS = """
// Stagger card animations
document.querySelectorAll('.card').forEach(function(el, i) {
  el.style.animationDelay = (i * 0.06) + 's';
});
"""

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _e(text: str) -> str:
    return html.escape(str(text or ""))

def _badge_html(decision: str) -> str:
    s = DECISION_STYLES.get(decision, DECISION_STYLES["skip"])
    return (
        f'<span class="badge" style="color:{s["color"]};'
        f'background:{s["glow"]};border-color:{s["border"]}">'
        f'{s["symbol"]} {s["label"]}</span>'
    )

def _score_ring_svg(
    score: float,
    color: str,
    size: int = 72,
    stroke_w: int = 5,
    delay: str = "0.2s",
) -> str:
    r = (size / 2) - stroke_w - 2
    cx = cy = size / 2
    circ = 2 * math.pi * r
    offset = circ * (1 - max(0, min(100, score)) / 100)
    num_size = max(13, size // 5)
    return (
        f'<div class="ring-wrap" style="width:{size}px;height:{size}px">'
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'style="transform:rotate(-90deg)">'
        f'<circle class="ring-track" cx="{cx}" cy="{cy}" r="{r:.1f}" '
        f'stroke-width="{stroke_w}"/>'
        f'<circle class="ring-progress" cx="{cx}" cy="{cy}" r="{r:.1f}" '
        f'stroke="{color}" stroke-width="{stroke_w}" '
        f'stroke-dasharray="{circ:.1f}" '
        f'style="--ring-max:{circ:.1f};--ring-offset:{offset:.1f};--ring-delay:{delay}"/>'
        f'</svg>'
        f'<div class="ring-score-text">'
        f'<span class="ring-score-num" style="color:{color};font-size:{num_size}px">'
        f'{score:.0f}</span>'
        f'<span class="ring-score-denom">/ 100</span>'
        f'</div>'
        f'</div>'
    )

def _score_bar_html(score: float, max_score: float, color: str) -> str:
    pct = min(100, round((score / max_score) * 100)) if max_score > 0 else 0
    return (
        f'<div class="dim-bar-bg">'
        f'<div class="dim-bar-fill" style="width:{pct}%;--bar-color:{color}"></div>'
        f'</div>'
    )

def _pills_html(skills: list[str], css_class: str) -> str:
    if not skills:
        return '<span style="font-size:12px;color:var(--text-muted);font-style:italic">None</span>'
    return "".join(f'<span class="pill {css_class}">{_e(s)}</span>' for s in skills)

def _flags_html(flags: list[str]) -> str:
    if not flags:
        return ""
    items = "".join(
        f'<span class="flag">{_e(f.replace("_"," "))}</span>' for f in flags
    )
    return (
        f'<div class="card-section">'
        f'<div class="card-section-title">Scoring Flags</div>'
        f'<div class="flags">{items}</div>'
        f'</div>'
    )

def _dim_rows_html(dim_scores: DimensionScores, decision_color: str) -> str:
    rows = ""
    for field, label in DIMENSION_LABELS.items():
        dim: DimensionScore = getattr(dim_scores, field, None)
        if dim is None:
            continue
        bar = _score_bar_html(dim.score, dim.max, decision_color)
        rows += (
            f"<tr>"
            f'<td class="dim-name">{_e(label)}</td>'
            f'<td class="dim-bar-wrap">{bar}</td>'
            f'<td class="dim-score">{dim.score:.0f}/{dim.max:.0f}</td>'
            f'<td class="dim-detail">{_e(dim.detail)}</td>'
            f"</tr>"
        )
    return f'<table class="dim-table">{rows}</table>'

def _link_buttons(job: FitScoreResult, color: str, border: str) -> str:
    if not job.job_id:
        return ""
    li_url = f"https://www.linkedin.com/jobs/view/{_e(job.job_id)}/"
    anchor  = f"#job-{_e(job.job_id)}"
    return (
        f'<a href="{li_url}" target="_blank" class="link-btn" '
        f'style="color:{color};border-color:{border};background:rgba(0,0,0,0.2)">'
        f'LinkedIn ↗</a>'
        f'<a href="{anchor}" class="link-btn-ghost">report ↓</a>'
    )

# ---------------------------------------------------------------------------
# Apply feature card (dashboard)
# ---------------------------------------------------------------------------

def _apply_feature_card_html(job: FitScoreResult, delay: str) -> str:
    s = DECISION_STYLES["apply"]
    ring = _score_ring_svg(job.total_score, s["color"], size=60, stroke_w=4, delay=delay)
    links = _link_buttons(job, s["color"], s["border"])
    return (
        f'<div class="apply-feature-card">'
        f'<div class="apply-card-ring">{ring}</div>'
        f'<div class="apply-card-info">'
        f'<div class="apply-card-title">{_e(_get_title(job))}</div>'
        f'<div class="apply-card-company">{_e(_get_company(job))}</div>'
        f'<div class="apply-card-links">{links}</div>'
        f'</div>'
        f'</div>'
    )

# ---------------------------------------------------------------------------
# Compact job row (non-Apply buckets dashboard)
# ---------------------------------------------------------------------------

def _compact_job_row_html(job: FitScoreResult, color: str, border: str) -> str:
    if job.job_id:
        li_url = f"https://www.linkedin.com/jobs/view/{_e(job.job_id)}/"
        anchor  = f"#job-{_e(job.job_id)}"
        link_html = (
            f'<a href="{li_url}" target="_blank" class="link-btn" '
            f'style="color:{color};border-color:{border};background:rgba(0,0,0,0.2)">'
            f'↗</a>'
            f'<a href="{anchor}" class="link-btn-ghost" style="font-size:11px">↓</a>'
        )
    else:
        link_html = ""
    return (
        f'<div class="dash-job">'
        f'<div class="dash-job-info">'
        f'<div class="dash-job-title">{_e(_get_title(job))}</div>'
        f'<div class="dash-job-company">{_e(_get_company(job))}</div>'
        f'</div>'
        f'<div class="dash-job-score" style="color:{color}">{job.total_score:.0f}</div>'
        f'<div style="display:flex;gap:6px;flex-shrink:0">{link_html}</div>'
        f'</div>'
    )

# ---------------------------------------------------------------------------
# Dashboard bucket
# ---------------------------------------------------------------------------

def _dashboard_bucket_html(decision: str, jobs: list[FitScoreResult]) -> str:
    s = DECISION_STYLES[decision]
    is_apply = decision == "apply"

    count_pill = (
        f'<span class="dash-count" style="color:{s["color"]};'
        f'border-color:{s["border"]};background:{s["glow"]}">'
        f'{len(jobs)}</span>'
    )
    label_html = (
        f'<span class="dash-bucket-label" style="color:{s["color"]}">'
        f'{s["symbol"]} {s["label"]}</span>'
    )

    if not jobs:
        content = '<div class="dash-empty">None</div>'
    elif is_apply:
        cards = ""
        for i, job in enumerate(jobs):
            delay = f"{0.2 + i * 0.1:.1f}s"
            cards += _apply_feature_card_html(job, delay)
        content = f'<div class="apply-cards-grid">{cards}</div>'
    else:
        rows = "".join(
            _compact_job_row_html(job, s["color"], s["border"]) for job in jobs
        )
        content = rows

    bucket_class = "dash-bucket-apply" if is_apply else "dash-bucket"
    return (
        f'<div class="{bucket_class}">'
        f'<div class="dash-bucket-header">{label_html}{count_pill}</div>'
        f'{content}'
        f'</div>'
    )

# ---------------------------------------------------------------------------
# Mini-report card
# ---------------------------------------------------------------------------

def _mini_report_html(job: FitScoreResult, card_index: int) -> str:
    s = DECISION_STYLES.get(job.decision, DECISION_STYLES["skip"])
    badge  = _badge_html(job.decision)
    ring   = _score_ring_svg(
        job.total_score, s["color"],
        size=80, stroke_w=5,
        delay=f"{0.1 + card_index * 0.06:.2f}s"
    )

    # LinkedIn link in header
    li_link = ""
    if job.job_id:
        li_url = f"https://www.linkedin.com/jobs/view/{_e(job.job_id)}/"
        li_link = (
            f'<a href="{li_url}" target="_blank" class="link-btn" '
            f'style="color:{s["color"]};border-color:{s["border"]};'
            f'background:{s["glow"]};font-size:10.5px">LinkedIn ↗</a>'
        )

    location_str = _e(getattr(job, "_location", "") or "")
    meta_html = ""
    if location_str:
        meta_html = f'<div class="card-meta">{location_str}</div>'

    # Condition
    cond_html = ""
    if job.condition:
        cond_html = (
            f'<div class="card-section">'
            f'<div class="card-section-title">Condition</div>'
            f'<div class="condition">⚠ {_e(job.condition)}</div>'
            f'</div>'
        )

    # Reasoning
    reason_html = ""
    if job.reasoning:
        reason_html = (
            f'<div class="card-section">'
            f'<div class="reasoning">{_e(job.reasoning)}</div>'
            f'</div>'
        )

    # Dimensions
    dim_html = ""
    if job.dimension_scores:
        dim_html = (
            f'<div class="card-section">'
            f'<div class="card-section-title">Score Breakdown</div>'
            f'{_dim_rows_html(job.dimension_scores, s["color"])}'
            f'</div>'
        )

    # Skills
    skills_html = ""
    has_skills = job.matched_skills or job.missing_skills or job.near_match_skills
    if has_skills:
        near = ""
        if job.near_match_skills:
            near = (
                f'<div style="margin-top:6px">'
                f'<span style="font-size:10px;color:var(--text-muted);font-weight:700;'
                f'letter-spacing:1px;text-transform:uppercase;margin-right:6px">Near Match</span>'
                f'<span class="pills" style="display:inline-flex">'
                f'{_pills_html(job.near_match_skills, "pill-near")}</span></div>'
            )
        skills_html = (
            f'<div class="card-section">'
            f'<div class="card-section-title">Required Skills</div>'
            f'<div style="margin-bottom:7px">'
            f'<span style="font-size:10px;color:var(--text-muted);font-weight:700;'
            f'letter-spacing:1px;text-transform:uppercase;margin-right:6px">Matched</span>'
            f'<span class="pills" style="display:inline-flex">'
            f'{_pills_html(job.matched_skills, "pill-matched")}</span></div>'
            f'<div>'
            f'<span style="font-size:10px;color:var(--text-muted);font-weight:700;'
            f'letter-spacing:1px;text-transform:uppercase;margin-right:6px">Missing</span>'
            f'<span class="pills" style="display:inline-flex">'
            f'{_pills_html(job.missing_skills, "pill-missing")}</span></div>'
            f'{near}'
            f'</div>'
        )

    # Nice-to-haves
    nth_html = ""
    if job.matched_nice_to_have or job.missing_nice_to_have:
        nth_html = (
            f'<div class="card-section">'
            f'<div class="card-section-title">Nice-to-Have</div>'
            f'<div style="margin-bottom:6px">'
            f'<span style="font-size:10px;color:var(--text-muted);font-weight:700;'
            f'letter-spacing:1px;text-transform:uppercase;margin-right:6px">Have</span>'
            f'<span class="pills" style="display:inline-flex">'
            f'{_pills_html(job.matched_nice_to_have, "pill-nth-ok")}</span></div>'
            f'<div>'
            f'<span style="font-size:10px;color:var(--text-muted);font-weight:700;'
            f'letter-spacing:1px;text-transform:uppercase;margin-right:6px">Don\'t Have</span>'
            f'<span class="pills" style="display:inline-flex">'
            f'{_pills_html(job.missing_nice_to_have, "pill-nth-miss")}</span></div>'
            f'</div>'
        )

    flags = _flags_html(job.scoring_flags or [])

    header_bg = f"linear-gradient(135deg, {s['glow']} 0%, transparent 60%)"

    return f"""
<div class="card" id="job-{_e(job.job_id or 'x')}">
  <div class="card-header" style="background:{header_bg}">
    <div class="card-title-block">
      <div class="card-title">{_e(_get_title(job))}</div>
      <div class="card-company">{_e(_get_company(job))}</div>
      {meta_html}
      <div style="margin-top:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        {badge}
        {li_link}
      </div>
    </div>
    {ring}
  </div>
  <div class="card-body">
    {reason_html}
    {cond_html}
    {dim_html}
    {skills_html}
    {nth_html}
    {flags}
  </div>
</div>"""

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _get_title(job: FitScoreResult) -> str:
    return getattr(job, "_job_title", None) or f"Job {job.job_id or '—'}"

def _get_company(job: FitScoreResult) -> str:
    return getattr(job, "_company_name", None) or ""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    scored_jobs: list[FitScoreResult],
    output_path: str = "job_hunt_report.html",
    run_label: str = "",
) -> str:
    """
    Generate an HTML report from scored FitScoreResult objects.
    Pass only status == "ok" records; filter rejected before calling.

    Args:
        scored_jobs:  List of FitScoreResult objects.
        output_path:  File path for the HTML output.
        run_label:    Batch label shown in the header (e.g. "June 4 2026").

    Returns:
        The generated HTML string.
    """
    timestamp = run_label or datetime.now().strftime("%B %d, %Y")
    total = len(scored_jobs)

    if total == 0:
        html_out = _empty_report_html(timestamp)
        _write(html_out, output_path)
        return html_out

    sorted_jobs = sorted(scored_jobs, key=lambda j: j.total_score, reverse=True)

    buckets: dict[str, list[FitScoreResult]] = {
        "apply": [], "conditional_apply": [], "maybe": [], "skip": [],
    }
    for job in sorted_jobs:
        key = job.decision if job.decision in buckets else "skip"
        buckets[key].append(job)

    # Dashboard
    dashboard_html = '<div class="dashboard">'
    for d in ("apply", "conditional_apply", "maybe", "skip"):
        dashboard_html += _dashboard_bucket_html(d, buckets[d])
    dashboard_html += "</div>"

    # Mini-reports
    card_index = 0
    mini_html = ""
    for d in ("apply", "conditional_apply", "maybe", "skip"):
        for job in buckets[d]:
            mini_html += _mini_report_html(job, card_index)
            card_index += 1

    # Header stats
    def stat(num: int, label: str, color: str) -> str:
        return (
            f'<div class="stat-box">'
            f'<div class="stat-num" style="color:{color}">{num}</div>'
            f'<div class="stat-label">{label}</div>'
            f'</div>'
        )

    stats_html = (
        stat(len(buckets["apply"]),             "Apply",       "#34D399") +
        stat(len(buckets["conditional_apply"]), "Conditional", "#FBBF24") +
        stat(len(buckets["maybe"]),             "Maybe",       "#818CF8") +
        stat(len(buckets["skip"]),              "Skip",        "#4B5563")
    )

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JobHunter AI — {_e(timestamp)}</title>
  <style>{CSS}</style>
</head>
<body>

<header class="report-header">
  <div>
    <h1>JobHunter AI</h1>
    <div class="header-sub">FIT SCORE REPORT &nbsp;·&nbsp; {_e(timestamp)} &nbsp;·&nbsp; {total} jobs</div>
  </div>
  <div class="header-stats">{stats_html}</div>
</header>

<div class="section">
  <div class="section-title">Decision Dashboard</div>
  {dashboard_html}
</div>

<div class="section-divider"><hr/></div>

<div class="section">
  <div class="section-title">Full Reports</div>
  {mini_html}
</div>

<script>{JS}</script>
</body>
</html>"""

    _write(html_out, output_path)
    return html_out


def _empty_report_html(timestamp: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>JobHunter AI</title><style>{CSS}</style></head>
<body>
<header class="report-header">
  <div><h1>JobHunter AI</h1>
  <div class="header-sub">FIT SCORE REPORT · {_e(timestamp)}</div></div>
</header>
<div class="empty-report">
  <h2>No jobs to display</h2>
  <p>Run the Job Analyst and Fit Scorer on a batch of job postings first.</p>
</div>
</body></html>"""


def _write(html_str: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"Report written → {path}")


def attach_display_meta(
    result: FitScoreResult,
    job_title: str,
    company_name: str,
    location: str = "",
) -> FitScoreResult:
    """
    Attach display fields to a FitScoreResult for the report.
    Call between score_job() and generate_report().

    Example:
        result = attach_display_meta(
            score_job(job_record, profile),
            job_title=job_record.job_title.value,
            company_name=job_record.company_name.value,
            location=job_record.location.value,
        )
    """
    result._job_title    = job_title
    result._company_name = company_name
    result._location     = location
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fit_scorer import DimensionScores, DimensionScore

    def _mock(job_id, title, company, score, decision, condition="", reasoning=""):
        dims = DimensionScores(
            required_skills    = DimensionScore(score=round(score*0.25,1), max=25, detail=f"{round(score*0.25/25*9)}/9 matched"),
            ai_genai_depth     = DimensionScore(score=round(score*0.20,1), max=20, detail="AI-core role" if score > 70 else "Non-AI role"),
            seniority_fit      = DimensionScore(score=round(score*0.10,1), max=10, detail="Exact Senior match"),
            nice_to_have       = DimensionScore(score=round(score*0.05,1), max=5,  detail="2/3 matched"),
            work_arrangement   = DimensionScore(score=round(score*0.15,1), max=15, detail="Remote — preferred"),
            industry_alignment = DimensionScore(score=round(score*0.15,1), max=15, detail="AI-core company"),
            compensation       = DimensionScore(score=round(score*0.10,1), max=10, detail="Within CAD target"),
        )
        r = FitScoreResult(
            job_id=job_id, candidate_name="Anshu Joshi",
            total_score=score, decision=decision,
            condition=condition,
            reasoning=reasoning or f"Strong overall fit for {title} with well-matched AI skills. Minor gaps exist in a few secondary dimensions. Apply with confidence and lead with your production AI experience.",
            ai_classification="ai_core" if score > 70 else "non_ai",
            dimension_scores=dims,
            matched_skills=["RAG Pipelines","Prompt Engineering","Stakeholder Management","Product Strategy"],
            missing_skills=["AWS SageMaker"] if score < 85 else [],
            near_match_skills=["AI platforms ≈ Claude API, RAG pipelines"],
            matched_nice_to_have=["Figma","Agile"],
            missing_nice_to_have=["PMP"],
            scoring_flags=[] if score > 60 else ["salary_not_stated"],
        )
        return attach_display_meta(r, title, company, "Canada (Remote)")

    sample = [
        _mock("4409361708", "Senior AI PM, AI Builder",  "Babylist",     88.0, "apply",
              reasoning="Exceptional skills match — 8 of 9 required skills matched and AI-core classification aligns perfectly with your RAG and agentic workflows background. Remote-Canada and compensation both solid fits. Apply immediately and lead with your MyTravelWallet production deployment as the centrepiece."),
        _mock("4417367952", "PM, Support Experience",    "Stripe",        82.0, "apply",
              reasoning="Strong technical PM match with 5+ years requirement aligned to your experience and AI-enabled conversational agent role well-suited to your Zoe AI background. Hybrid work arrangement and USD compensation both score well. Apply and highlight your production conversational AI work in the cover letter."),
        _mock("4409957797", "Strategy & Product Owner",  "TELUS Digital", 68.0, "conditional_apply",
              condition="AI/GenAI Depth — AI-enabled inside telecom (14/20)",
              reasoning="Solid PM fundamentals and Dialogflow CX signals are a direct match. Score limited by the AI-enabled classification — this is conversational AI inside a large telecom, not an AI-core product. Apply, but frame your background around enterprise AI delivery rather than AI product building."),
        _mock("4371068977", "Product Manager",           "Guidepoint",    71.0, "conditional_apply",
              condition="AI/GenAI Depth — non-AI role (5/20)",
              reasoning="Strong seniority fit and compensation match at CAD $120–140K. Score held back by the absence of AI signals — Guidepoint is a research platform, not an AI product company. Worth applying if open to non-AI roles; lead with PM fundamentals and analytics credentials."),
        _mock("AR001",      "AI Product Manager",        "Arize AI",      52.0, "maybe",
              reasoning="AI-core domain is a perfect conceptual fit and compensation is strong at $150–220K USD. Experience gap is the limiting factor — 2.5 years PM title at the lower boundary of the 2–3 year requirement. Monitor this one and revisit after another 6 months of PM title experience."),
        _mock("SK001",      "Associate PM",              "Generic Corp",  30.0, "skip",
              reasoning="Level mismatch is disqualifying — Associate PM is two levels below your Senior target. Compensation also falls below the CAD $120K floor. Skip and redirect energy toward Senior-level AI PM roles where your background is a genuine fit."),
    ]

    generate_report(
        scored_jobs=sample,
        output_path="/mnt/user-data/outputs/sample_report.html",
        run_label="June 4, 2026 — Demo Run",
    )
