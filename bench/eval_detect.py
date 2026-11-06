"""Scores engine/detect.py on several random synthetic datasets.
Usage (from project folder): python bench/eval_detect.py "{}"   [SEEDS=11,12,... to change seeds]
Prints, per seed, which planted events (A,B,U,C) were found and any clean-period false alarms."""
import sys, io, time, warnings
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from concurrent.futures import ProcessPoolExecutor
from bench.make_synthetic import generate
from engine.ingest import load_dataset
from engine.detect import detect_all, DetectConfig

VOL = {"revenue", "orders", "quantity"}
PROF = {"profit", "profit_margin"}

def overlaps(e, a, b):
    return e.start <= pd.Timestamp(b) and e.end >= pd.Timestamp(a)

def run(seed, cfg_kwargs):
    df, _ = generate(seed)
    ds = load_dataset(df.to_csv(index=False).encode(), filename=f"s{seed}.csv")
    events, _ = detect_all(ds, "M", config=DetectConfig(**cfg_kwargs))
    down = [e for e in events if e.direction == "down"]
    res = {
        "A": any(e.kpi in PROF and overlaps(e, "2024-03-01", "2024-04-30") for e in down),
        "B": any(e.kpi in VOL and overlaps(e, "2024-06-01", "2024-08-31") for e in down),
        "U": any(e.kpi in VOL and overlaps(e, "2024-09-01", "2024-09-30") for e in down),
        "C": any(e.kpi in VOL and overlaps(e, "2024-10-01", "2024-11-30") for e in down),
        "false": [f"{e.kpi}:{e.start:%Y-%m}" for e in events if e.start < pd.Timestamp("2024-03-01")],
    }
    return seed, res

if __name__ == "__main__":
    kw = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
    import os; seeds = [int(x) for x in os.environ.get("SEEDS","1,2,3,4,5,6,7,8").split(",")]
    t0 = time.time()
    with ProcessPoolExecutor(4) as ex:
        out = list(ex.map(run, seeds, [kw] * len(seeds)))
    tot = {k: 0 for k in "ABUC"}
    nfalse = 0
    for seed, r in sorted(out):
        for k in "ABUC": tot[k] += r[k]
        nfalse += len(r["false"])
        print(seed, {k: int(r[k]) for k in "ABUC"}, "false alarms in clean period:", r["false"])
    print("RECALL out of", len(seeds), tot, "| total clean-period false alarms:", nfalse, f"| {time.time()-t0:.0f}s")
