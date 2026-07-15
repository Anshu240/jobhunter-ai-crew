JobHunter AI Crew

A multi-agent job search intelligence system built with the Anthropic SDK and Pydantic structured outputs.

Paste a job description. Get a fit score, tailored resume bullets, company research, and a full interview prep kit in one pipeline run.

What It Does

JobHunter AI Crew runs a crew of specialized agents, each responsible for a narrowly defined task. Agents hand off structured outputs to the next stage rather than sharing a single context window.

| Agent | Responsibility |
|---|---|
| Job Analyst | Extracts required skills, seniority signals, and role context from a JD |
| Profile Builder | Loads candidate profile and maps experience to JD requirements |
| Fit Scorer| Scores candidate fit (0–100) with decision: Apply / Conditional / Maybe / Skip |
| Company Researcher | Pulls company context, recent news, culture signals, and interview intel |
| Resume Strategist | Generates tailored resume bullets aligned to the specific JD |
| Interview Prep Coach | Builds role-specific prep — likely questions, talking points, red flags |
| Presenter| Formats and saves the full output package for review |

Pipeline also includes `interview_kit.py` for single-job deep prep ("what do I say tomorrow?") and `run_evals.py` for evaluating output quality across multiple JDs.

Architecture Decisions

Fit Scorer as a gate. If the Fit Scorer returns `decision: skip`, downstream agents (Company Researcher, Resume Strategist, Interview Prep Coach) do not run. No tokens wasted on jobs not worth pursuing.

**Specialized agents over one large agent.** Each agent owns a narrow responsibility. This makes outputs easier to evaluate, debug, and improve independently  and makes it easier to reason about what each agent can do versus what it should do.

Two distinct output modes.
- `presenter.py` — batch view across multiple JDs ("which of these should I pursue?")
- `interview_kit.py` — single-job deep prep ("what do I say in the interview tomorrow?")

Graceful degradation. Company Researcher failure (e.g. limited public data on smaller companies) does not halt the pipeline. Downstream agents proceed with available context.

Pydantic structured outputs. Every agent produces typed, validated output. No free-text parsing between agent handoffs.

Tech Stack

- Anthropic SDK (claude-sonnet-4-6) — raw API, no framework abstraction
- Pydantic — structured output validation between agents
- Python 3.12
- YAML — agent prompts and knowledge base configuration


Project Structure

```
jobhunter-ai-crew/
├── agents/                  # Agent definitions
├── prompts/                 # System prompts per agent
├── tools/                   # Shared utilities
├── job_analyst.py           # JD analysis agent
├── fit_scorer.py            # Fit scoring + decision gate
├── company_researcher.py    # Company research agent
├── resume_strategist.py     # Resume tailoring agent
├── interview_prep_coach.py  # Interview prep agent
├── presenter.py             # Batch output formatter
├── profile_builder.py       # Candidate profile loader
├── interview_kit.py         # Single-job deep prep
├── run_pipeline.py          # Orchestrator (manual + auto modes)
├── run_evals.py             # Evaluation harness
└── knowledge_base.yaml      # Agent knowledge configuration
```

 Running the Pipeline

Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Anthropic API key to a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

Manual mode** (paste a JD directly)
```bash
python3 run_pipeline.py --manual
```

Full pipeline with all agents**
```bash
python3 run_pipeline.py --manual --mode full
```

Fit check only** (Job Analyst + Fit Scorer)
```bash
python3 run_pipeline.py --manual --mode fit
```

Interview kit** (single job deep prep)
```bash
python3 interview_kit.py
```

Run evals
```bash
python3 run_evals.py
```

Status

Active development. Evals in progress across multiple JD types.

- [x] 7-agent pipeline operational
- [x] Fit Scorer gate logic validated
- [x] Structured output evaluation harness built
- [ ] Automated intake via Gmail MCP (scaffolded, OAuth setup in progress)
- [ ] Eval results finalized

About

Built by [Anshu Joshi](https://www.linkedin.com/in/anshujoshi240) — AI Product Manager.  
Portfolio: [anshujoshi-aipm-portfolio.netlify.app](https://anshujoshi-aipm-portfolio.netlify.app)
