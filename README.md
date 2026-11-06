# InsightTrace

**Understand what changed. Discover why.**

InsightTrace is an AI business-intelligence app. You give it a sales table. It finds the unusual changes in your KPIs, splits each change into the groups that contributed, tests whether any group is a real cause, and explains the result with the evidence behind it.

Its main rule: **numbers come from code, not from the AI.** Every figure is calculated in Python and SQL. The AI only picks which calculation to run and rewords the result, and a checker rejects any sentence that contains a number it cannot find in the evidence. When the evidence is weak, InsightTrace says **UNKNOWN** instead of guessing.

> Live demo: _add your deployed link here_
> Video / screenshots: _add here_

---

## What you can do with it

| Tab | What it does |
|---|---|
| **Overview** | KPI cards, a chart with the range the detector *expected*, flagged unusual periods, and a list of alerts. |
| **Analyst** | Ask questions in plain English ("Which region had the highest revenue?", "Why did revenue fall in autumn 2024?"). Answers cite their evidence ([E1], [E2]...). Quick tools run the same calculations without the AI. |
| **Root Cause** | Pick an alert. See who contributed, then five checks per candidate. Verdict: **CAUSE**, **NOT A CAUSE** or **UNKNOWN**. |
| **Forecast** | Next periods with an 80% range, plus a back-test: did the range hold, and is it better than repeating last year? |
| **Voice** | Ask by voice and hear the answer read back (uses the browser's speech tools). |

### The five checks

A group is called a **CAUSE** only if it passes every check:

1. It is big enough and distinctive (not just "everything moved together").
2. It differs from the rest of the business.
3. The change started inside the window, not before it.
4. It is not a yearly repeat (seasonal pattern).
5. It is unusual for its own history.

If evidence is missing or mixed, the answer is **UNKNOWN**, and the app suggests what data to collect next. Correlation is never presented as proof.

---

## Architecture

### 1. System overview

The app has four layers. The **engine** calculates every number. The **LLM layer** only chooses which calculation to run and rewords the result. The **API** connects the browser to both. The **frontend** shows the answers.

```mermaid
flowchart TB
  U([User in browser]):::user
  subgraph FE["Frontend: React + Vite"]
    direction LR
    T1[Overview]:::fe
    T2[Analyst]:::fe
    T3[Root Cause]:::fe
    T4[Forecast]:::fe
    T5[Voice]:::fe
  end
  subgraph API["Backend: FastAPI"]
    R[Routes in api/main.py<br/>upload, demo, summary, alerts, why, tool, ask]:::api
    S[Service in api/service.py<br/>sessions, rate limits, upload size guard]:::api
  end
  subgraph ENG["Engine: all numbers are computed here"]
    direction LR
    E1[ingest<br/>CSV/Excel to DuckDB]:::eng
    E2[kpi]:::eng
    E3[detect<br/>seasonal baseline]:::eng
    E4[decompose<br/>exact contributions]:::eng
    E5[verify<br/>5 checks + placebo]:::eng
    E6[evidence<br/>records E1..En]:::eng
    E7[forecast<br/>conformal + back-test]:::eng
    E1 --> E2 --> E3 --> E4 --> E5 --> E6
    E2 --> E7
  end
  subgraph LLM["LLM layer: words only, never numbers"]
    direction LR
    L1[Planner<br/>picks 1 of 8 tools]:::llm
    L2[Tools + SQL guard<br/>run the calculation]:::llm
    L3[Explainer<br/>rewords the evidence]:::llm
    L4[Validator<br/>rejects unsupported text]:::llm
    L5[Plain fallback<br/>written by code]:::llm
    L1 --> L2 --> L3 --> L4
    L4 -- rejected twice --> L5
  end
  G[(Gemini API)]:::ext
  D[(DuckDB<br/>in memory)]:::db
  U --> FE
  FE -- "HTTPS /api" --> R --> S
  S --> ENG
  S --> LLM
  E1 --- D
  L2 --> ENG
  L1 -.-> G
  L3 -.-> G
  classDef user fill:#2b2350,stroke:#a78bfa,color:#fff
  classDef fe fill:#1e3a8a,stroke:#60a5fa,color:#fff
  classDef api fill:#4c1d95,stroke:#a78bfa,color:#fff
  classDef eng fill:#064e3b,stroke:#34d399,color:#fff
  classDef llm fill:#831843,stroke:#f472b6,color:#fff
  classDef ext fill:#333,stroke:#aaa,color:#fff
  classDef db fill:#333,stroke:#aaa,color:#fff
```

### 2. Data pipeline (runs once when a file is loaded)

```
CSV / Excel ──► ingest ──► detect changes ──► decompose by group ──► verify (5 checks) ──► evidence records E1..En
                 (DuckDB)   (seasonal baseline)  (exact contributions)   (incl. placebo test)
```

### 3. What happens when you ask a question

```mermaid
sequenceDiagram
  autonumber
  actor U as User
  participant FE as React app
  participant API as FastAPI + Service
  participant PL as Planner (Gemini)
  participant TL as Tool (Python/SQL)
  participant EX as Explainer (Gemini)
  participant VA as Validator
  U->>FE: "Why did revenue fall in autumn 2024?"
  FE->>API: POST /api/sessions/{id}/ask
  API->>API: rate limit check (per visitor and global)
  API->>PL: question + list of 8 allowed tools
  PL-->>API: tool = why, arguments
  API->>TL: run the tool
  TL-->>API: numbers + evidence records E1..En
  API->>EX: evidence only
  EX-->>API: draft answer with [E#] citations
  API->>VA: check every number, date, citation
  alt draft is supported by evidence
    VA-->>API: accepted
  else rejected
    API->>EX: retry once with the reasons
    EX-->>API: second draft
    VA-->>API: accepted, or fall back to plain text from code
  end
  API-->>FE: answer + evidence + status
  FE-->>U: answer card with evidence chips
```

### 4. Why it is built this way

- **Detection** compares each period with what the same time of year usually brings, so ordinary seasons are not alerts.
- **Decomposition** is exact: group contributions add up to the total change.
- **Verification** includes a placebo test: the same procedure is run on periods where nothing happened, to see how often it would blame something by chance.
- **Evidence records** are the only source the AI may use. Each has an id, facts with units, the checks that were run and the SQL used.
- **Validator** rejects wording that contains numbers or dates not in the cited records, overclaims ("proves"), stray citations, or sign mistakes like "fell by -5". It retries once, then falls back to a plain summary written by code.
- **Analyst** can only use 8 fixed tools (KPI value, breakdown, trend, contributors, alerts, why, forecast, read-only SQL). The SQL tool accepts a single `SELECT` on one table and blocks everything else.
- **Forecast** uses seasonal methods with conformal ranges and walk-forward validation, and reports `LOW_CONFIDENCE` when it does not beat a simple baseline.

---

## Run it on your computer (Windows)

You need Python 3.12 and Node.js 20 or newer.

**Backend**

```powershell
cd InsightTrace
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then open .env and paste your Gemini key
uvicorn api.main:app --port 8000
```

**Frontend** (in a second terminal)

```powershell
cd InsightTrace\frontend
npm install
npm run dev
```

Open http://localhost:5173. Use **Synthetic business** or **Superstore** on the Home page to load a demo without any file.

The AI is optional. Without `GEMINI_API_KEY`, Overview, Root Cause, Forecast and the Analyst's Quick tools still work. Typed questions and Voice need the key. Get a free key from Google AI Studio and keep it in `.env` only. `.env` is ignored by git.

**Tests**

```powershell
pytest tests/ -q              # backend and engine
cd frontend; npm test         # frontend logic
python -m bench.run_bench --seeds 1,2,3 --no-demo     # accuracy check on generated data
```

---

## Settings (environment variables)

| Name | Where | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | backend | Key for the AI. Never commit it. |
| `GEMINI_MODEL` | backend | Model id to use (see `python -m llm.client --list-models`). |
| `LLM_MODE=off` | backend | Turn the AI off completely. |
| `CORS_ORIGINS` | backend | Comma-separated website addresses allowed to call the API (your frontend's address). |
| `MAX_UPLOAD_MB` | backend | Largest upload accepted (default 25). |
| `VITE_API_URL` | frontend | Address of the backend when deployed. Empty in development (the dev server forwards `/api`). |

---

## Project layout

```
api/        FastAPI layer (service.py = logic, main.py = routes)
engine/     ingest, KPIs, detect, decompose, verify, evidence, forecast
llm/        Gemini client, prompts, validator, analyst tools, SQL guard
bench/      synthetic data generator and the accuracy benchmark
data/       demo files (synthetic business, Superstore) and the answer key
tests/      backend tests
frontend/   React + Vite app (src/lib = tested logic, src/components, src/pages)
docs/       design notes
```

---

## Built with

Python, pandas, DuckDB, FastAPI, React, Vite, Gemini (backend only), pytest.
