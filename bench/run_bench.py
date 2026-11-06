"""Benchmark: score the WHOLE pipeline against the planted answer key.

Usage (from the project folder):
    python -m bench.run_bench                       # data/synthetic_demo.csv only (about 1 minute)
    python -m bench.run_bench --seeds 1,2,3         # also freshly generated datasets (new random data)
    python -m bench.run_bench --llm                 # also measure the LLM (needs GEMINI_API_KEY + GEMINI_MODEL in .env)
    python -m bench.run_bench --out bench/results.json

What is scored (each is a yes/no per dataset; the answer key is data/synthetic_ground_truth.json)
-------------------------------------------------------------------------------------------------
  A   profit Mar-Apr 2024            a verified cause = Region West + Sub-Category Phones
  B   revenue Jun-Aug 2024           verified causes are the 5 planted churned Central customers
                                     (pass: found >= 3 of the 5, and >= 80% of the customers named are planted ones)
  C   revenue 15 Oct-30 Nov 2024     a verified cause involves Phones (Sub-Category) or Phones Max (Product)
  U   revenue Sep 2024               conclusion UNKNOWN (a uniform dip has no cause in the data)
  D1  revenue Mar 2024               East Chairs (a yearly March dip) is NOT reported as a cause
  D2  revenue Jan 2024               conclusion UNKNOWN (normal January slump)
  CALM revenue Jul-Aug 2023          conclusion UNKNOWN (nothing planted: a false-positive check)
  FORECAST revenue, monthly          status, and how often the 80% band held in past walk-forward checks

With --llm it also measures, with your real model: how often the explanation passed the number
checker (first try / after one retry / fell back to the plain text), and whether the question planner
picked the right tool for a few sample questions.

Honest limits
-------------
  - Everything is scored on SYNTHETIC data that I (the builder) generated and tuned the thresholds on.
    A good score here shows the pipeline does what it was designed to do; it does NOT show it will
    find the real causes in a real company's data. The Superstore file has no answer key.
  - Yes/no per scenario over a handful of datasets is a small sample: do not read 3/3 as 100% accuracy.
  - The LLM section depends on your model, key and the free-tier quota; it is skipped without a key.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from engine.evidence import build_evidence  # noqa: E402
from engine.forecast import forecast_kpi  # noqa: E402
from engine.ingest import load_dataset  # noqa: E402
from engine.verify import VerifyConfig, to_plain, verify_window  # noqa: E402

DEMO = ROOT / "data" / "synthetic_demo.csv"
TRUTH = ROOT / "data" / "synthetic_ground_truth.json"

SCENARIOS = [  # id, kpi, window, what a correct engine says
    ("A", "profit", ("2024-03-01", "2024-04-30"), "cause: West + Phones"),
    ("B", "revenue", ("2024-06-01", "2024-08-31"), "cause: the churned Central customers"),
    ("C", "revenue", ("2024-10-15", "2024-11-30"), "cause involving Phones / Phones Max"),
    ("U", "revenue", ("2024-09-01", "2024-09-30"), "UNKNOWN"),
    ("D1", "revenue", ("2024-03-01", "2024-03-31"), "East Chairs is not a cause"),
    ("D2", "revenue", ("2024-01-01", "2024-01-31"), "UNKNOWN"),
    ("CALM", "revenue", ("2023-07-01", "2023-08-31"), "UNKNOWN"),
]

QUESTIONS = [  # question, the tool a sensible planner should pick ("none" = cannot be answered from the file)
    ("What was our revenue in 2024?", "kpi_value"),
    ("Which region has the lowest profit margin?", "breakdown"),
    ("Why did profit drop in March and April 2024?", "why"),
    ("What will revenue look like over the next few months?", "forecast"),
    ("Show me the unusual changes in the data", "alerts"),
    ("How has revenue moved month by month?", "trend"),
    ("Who contributed most to the fall in revenue between June and December 2024?", "contributors"),
    ("Should I fire my sales team?", "none"),
]


def churned_customer_names(truth: dict) -> set[str]:
    """'CU-0242' in the answer key is 'Customer 242' in the loaded data (the ID column is dropped)."""
    ev = next(e for e in truth["events"] if e["id"].startswith("B"))
    return {f"Customer {int(c.split('-')[1])}" for c in ev["where"]["Customer ID"]}


def _parse_description(text: str) -> dict:
    """'Sub-Category = Phones + Region = West' -> {'Sub-Category': ['Phones'], 'Region': ['West']}"""
    out = {}
    for part in text.split(" + "):
        if " = " in part:
            dim, vals = part.split(" = ", 1)
            out[dim.strip()] = [v.strip() for v in vals.split(", ")]
    return out


def descriptions(res) -> list[dict]:
    """Every way the engine describes its causes: the main slice AND its equivalent descriptions
    (the same rows can be described by region, by city or by the customers; the engine lists one first)."""
    out = []
    for c in res.causes:
        out.append({k: list(v) for k, v in c.filters.items()})
        out += [_parse_description(a) for a in c.also_described_as]
    return out


def _members(res, dim: str) -> set[str]:
    return {m for d in descriptions(res) for m in d.get(dim, [])}


def _describe(res) -> list[str]:
    return [c.describe() + (f"  (also: {'; '.join(c.also_described_as[:3])})" if c.also_described_as else "")
            for c in res.causes]


def judge(sid: str, res, truth: dict) -> tuple[bool, str]:
    """(passed, one-line detail) for one scenario."""
    if sid == "A":
        ok = any("West" in d.get("Region", []) and "Phones" in d.get("Sub-Category", []) for d in descriptions(res))
        return ok, "; ".join(_describe(res)) or "UNKNOWN"
    if sid == "B":
        planted = churned_customer_names(truth)
        named = _members(res, "Customer Name")
        hit = named & planted
        ok = len(hit) >= 3 and len(hit) / max(1, len(named)) >= 0.8
        return ok, f"found {len(hit)} of {len(planted)} planted customers, named {len(named)} in total"
    if sid == "C":
        ok = any("Phones" in d.get("Sub-Category", []) or any("Phones Max" in m for m in d.get("Product Name", []))
                 for d in descriptions(res))
        return ok, "; ".join(_describe(res)) or "UNKNOWN"
    if sid == "D1":
        bad = [x for x in _describe(res) if "Chairs" in x]
        return not bad, ("reported as a cause: " + "; ".join(bad)) if bad else "not reported as a cause"
    ok = res.conclusion == "UNKNOWN"          # U, D2, CALM
    return ok, res.conclusion if ok else "reported causes: " + "; ".join(_describe(res))


def score_dataset(ds, truth: dict, keep_bundles: bool = False, config: VerifyConfig | None = None) -> dict:
    """Run every scenario and the forecast on one loaded dataset."""
    out = {"scenarios": [], "bundles": {}}
    for sid, kpi, window, expect in SCENARIOS:
        t0 = time.time()
        res = verify_window(ds, kpi, window, config=config)
        passed, detail = judge(sid, res, truth)
        out["scenarios"].append({"id": sid, "kpi": kpi, "window": list(window), "expected": expect,
                                 "passed": bool(passed), "detail": detail, "conclusion": res.conclusion,
                                 "seconds": round(time.time() - t0, 1)})
        if keep_bundles:
            out["bundles"][sid] = build_evidence(res)
    fc = forecast_kpi(ds, "revenue", "M")
    v = fc.validation
    cov, n = v.get("coverage"), v.get("n_checks")
    out["forecast"] = {"status": fc.status, "coverage": cov, "n_checks": n, "level": fc.level,
                       "gap_to_target": (abs(cov - fc.level) if cov is not None and n else None)}
    if keep_bundles:
        out["forecast_obj"] = fc
    return out


# ----------------------------------------------------------------------------
# LLM measurements
# ----------------------------------------------------------------------------
def score_llm(ds, llm, bundles: dict, forecast) -> dict:
    """How often the real model's explanations pass the validator, and planner accuracy."""
    from llm.analyst import answer_question
    from llm.explain import explain_evidence, explain_forecast
    rows = []
    for sid, b in bundles.items():
        e = explain_evidence(b, llm)
        rows.append({"what": sid, "source": e.source, "attempts": e.attempts, "problems": e.problems[:3],
                     "reason": e.reason})
    e = explain_forecast(forecast, llm)
    rows.append({"what": "forecast", "source": e.source, "attempts": e.attempts, "problems": e.problems[:3],
                 "reason": e.reason})
    planner = []
    for question, want in QUESTIONS:
        a = answer_question(ds, question, llm)
        got = a.tool if a.tool else a.status
        ok = (a.status == "unsupported") if want == "none" else (a.status == "ok" and a.tool == want)
        planner.append({"question": question, "wanted": want, "got": got, "status": a.status, "ok": bool(ok),
                        "explanation_source": a.explanation.source if a.explanation else None})
    n = len(rows)
    return {"model": getattr(llm, "model", getattr(llm, "name", "?")), "explanations": rows,
            "accepted_first_try": sum(r["source"] == "llm" and r["attempts"] == 1 for r in rows),
            "accepted_after_retry": sum(r["source"] == "llm" and r["attempts"] > 1 for r in rows),
            "fell_back": sum(r["source"] != "llm" for r in rows), "n_explanations": n,
            "planner": planner, "planner_ok": sum(p["ok"] for p in planner), "n_questions": len(planner)}


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def report(results: list[dict]) -> str:
    lines = []
    ids = [s[0] for s in SCENARIOS]
    lines.append("SCENARIOS (pass = the engine says what the answer key says)")
    for r in results:
        cells = {s["id"]: s for s in r["scenarios"]}
        row = "  ".join(f"{i}:{'PASS' if cells[i]['passed'] else 'FAIL'}" for i in ids)
        lines.append(f"  {r['name']:<22}{row}")
    lines.append("")
    tot = {i: sum(1 for r in results for s in r["scenarios"] if s["id"] == i and s["passed"]) for i in ids}
    lines.append("TOTAL  " + "  ".join(f"{i} {tot[i]}/{len(results)}" for i in ids))
    allp = sum(tot.values())
    lines.append(f"       {allp} of {len(ids) * len(results)} scenario checks passed")
    fails = [(r["name"], s) for r in results for s in r["scenarios"] if not s["passed"]]
    if fails:
        lines.append("\nFAILED CHECKS")
        for name, s in fails:
            lines.append(f"  {name} {s['id']} ({s['kpi']} {s['window'][0]}..{s['window'][1]}): "
                         f"wanted {s['expected']}; got {s['detail']}")
    lines.append("\nFORECAST (revenue, monthly)")
    for r in results:
        f = r["forecast"]
        cov = f"{f['coverage']:.0%} of {f['n_checks']} past checks inside the {f['level']:.0%} band" \
            if f["n_checks"] else "no back-test (too little history)"
        lines.append(f"  {r['name']:<22}{f['status']:<15}{cov}")
    for r in results:
        if r.get("llm"):
            m = r["llm"]
            lines.append(f"\nLLM ({m['model']}) on {r['name']}")
            lines.append(f"  explanations accepted on the first try: {m['accepted_first_try']}/{m['n_explanations']}, "
                         f"after one retry: {m['accepted_after_retry']}, fell back to plain text: {m['fell_back']}")
            for e in m["explanations"]:
                if e["source"] != "llm" or e["attempts"] > 1:
                    lines.append(f"    {e['what']}: {e['source']} (attempts {e['attempts']}) {'; '.join(e['problems'])[:200]}")
            lines.append(f"  planner picked the right tool: {m['planner_ok']}/{m['n_questions']}")
            for p in m["planner"]:
                if not p["ok"]:
                    lines.append(f"    '{p['question']}' wanted {p['wanted']}, got {p['got']} ({p['status']})")
    lines.append("\nRemember: synthetic data that the thresholds were tuned on; a small sample; not a guarantee on real data.")
    return "\n".join(lines)


def _config(args) -> VerifyConfig | None:
    """The verification settings for this run (default settings unless an experiment option was given)."""
    kw = {"auto_growth": False} if args.no_auto_growth else {}
    for item in args.set:
        name, _, value = item.partition("=")
        if name not in VerifyConfig.__dataclass_fields__ or name in ("growth_clip",):
            raise SystemExit(f"Unknown or unsupported setting: {name}")
        kw[name] = (value.lower() == "true") if isinstance(getattr(VerifyConfig(), name), bool) else float(value)
        if isinstance(getattr(VerifyConfig(), name), int) and not isinstance(getattr(VerifyConfig(), name), bool):
            kw[name] = int(float(value))
    return VerifyConfig(**kw) if kw else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score the InsightTrace pipeline against the planted answer key.")
    ap.add_argument("--seeds", default="", help="comma-separated seeds to generate extra datasets, e.g. 1,2,3")
    ap.add_argument("--no-demo", action="store_true", help="skip data/synthetic_demo.csv")
    ap.add_argument("--llm", action="store_true", help="also measure the LLM from .env (uses your API quota)")
    ap.add_argument("--no-auto-growth", action="store_true", help="compare with last year directly (old behaviour)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a VerifyConfig number for an experiment, e.g. --set min_lift=1.8")
    ap.add_argument("--out", default="", help="write full results as JSON to this path")
    args = ap.parse_args(argv)

    llm = None
    if args.llm:
        from llm.client import get_llm, llm_status
        llm = get_llm()
        if llm is None:
            print("LLM skipped:", llm_status()["reason"])
    datasets = []
    if not args.no_demo:
        datasets.append(("synthetic_demo.csv", load_dataset(DEMO), json.loads(TRUTH.read_text())))
    for s in [int(x) for x in args.seeds.split(",") if x.strip()]:
        from bench.make_synthetic import generate
        df, truth = generate(s)
        datasets.append((f"generated seed {s}", load_dataset(df.to_csv(index=False).encode(), filename=f"seed{s}.csv"),
                         truth))
    if not datasets:
        print("Nothing to run.")
        return 1
    results = []
    for name, ds, truth in datasets:
        print(f"Scoring {name} ...", flush=True)
        r = score_dataset(ds, truth, keep_bundles=llm is not None,
                          config=_config(args))
        r["name"] = name
        if llm is not None:
            r["llm"] = score_llm(ds, llm, r.pop("bundles"), r.pop("forecast_obj"))
        r.pop("bundles", None)
        r.pop("forecast_obj", None)
        results.append(r)
    print()
    print(report(results))
    if args.out:
        Path(args.out).write_text(json.dumps(to_plain(results), indent=1))
        print(f"\nFull results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
