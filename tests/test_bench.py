"""Tests for bench/run_bench.py (Step 10): the scoring rules themselves, then one real run.

    pytest tests/test_bench.py -v

The scoring helpers are tested with hand-made results so a wrong rule (for example one that passes a
false alarm) is caught. The last test runs the whole benchmark on data/synthetic_demo.csv (about a minute).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.run_bench import (QUESTIONS, SCENARIOS, _parse_description, churned_customer_names,  # noqa: E402
                             descriptions, judge, main, report, score_dataset)
from engine.ingest import load_dataset  # noqa: E402
from tests.test_evidence import make_cause, make_result  # noqa: E402

SYNTHETIC = ROOT / "data" / "synthetic_demo.csv"
TRUTH = ROOT / "data" / "synthetic_ground_truth.json"
needs_synthetic = pytest.mark.skipif(not SYNTHETIC.exists(), reason="synthetic file missing")


@pytest.fixture(scope="module")
def synthetic():
    return load_dataset(SYNTHETIC)


def truth():
    return json.loads(TRUTH.read_text())


def res_with(filters, also=(), conclusion=None):
    c = make_cause(filters)
    c.also_described_as = list(also)
    return make_result([c], conclusion=conclusion)


def test_planted_customer_ids_become_the_names_used_in_the_data():
    assert churned_customer_names(truth()) == {"Customer 242", "Customer 286", "Customer 312",
                                               "Customer 317", "Customer 330"}


def test_descriptions_are_parsed_including_equivalent_ones():
    assert _parse_description("Sub-Category = Phones + Region = West") == {"Sub-Category": ["Phones"], "Region": ["West"]}
    assert _parse_description("Customer Name = Customer 1, Customer 2") == {"Customer Name": ["Customer 1", "Customer 2"]}
    assert _parse_description("nonsense") == {}
    r = res_with({"City": ["Ohio Metro"]}, ["Customer Name = Customer 242, Customer 286"])
    assert descriptions(r) == [{"City": ["Ohio Metro"]}, {"Customer Name": ["Customer 242", "Customer 286"]}]


def test_scenario_A_needs_west_and_phones_together():
    both = res_with({"Region": ["West"], "Sub-Category": ["Phones"]})
    assert judge("A", both, truth())[0]
    via_alias = res_with({"City": ["X Metro"]}, ["Sub-Category = Phones + Region = West"])
    assert judge("A", via_alias, truth())[0]
    only_west = res_with({"Region": ["West"]})
    only_phones = res_with({"Sub-Category": ["Phones"]})
    assert not judge("A", only_west, truth())[0] and not judge("A", only_phones, truth())[0]
    assert not judge("A", make_result([]), truth())[0]


def test_scenario_B_needs_most_planted_customers_and_few_wrong_ones():
    planted = ["Customer 242", "Customer 286", "Customer 312", "Customer 317", "Customer 330"]
    assert judge("B", res_with({"Customer Name": planted}), truth())[0]
    assert judge("B", res_with({"Region": ["Central"]}, ["Customer Name = " + ", ".join(planted[:3])]), truth())[0]
    assert not judge("B", res_with({"Customer Name": planted[:2]}), truth())[0]                    # too few found
    noisy = planted[:3] + ["Customer 1", "Customer 2", "Customer 3"]
    assert not judge("B", res_with({"Customer Name": noisy}), truth())[0]                          # too many wrong names
    assert not judge("B", res_with({"Region": ["Central"]}), truth())[0]                           # region alone is not enough


def test_scenario_C_accepts_phones_or_phones_max_only():
    assert judge("C", res_with({"Sub-Category": ["Phones"]}), truth())[0]
    assert judge("C", res_with({"Product Name": ["Phones Max"]}), truth())[0]
    assert not judge("C", res_with({"Sub-Category": ["Chairs"]}), truth())[0]
    assert not judge("C", make_result([]), truth())[0]


def test_decoys_and_unknown_scenarios_fail_when_a_cause_is_reported():
    for sid in ("U", "D2", "CALM"):
        assert judge(sid, make_result([]), truth())[0], sid
        assert not judge(sid, res_with({"Ship Mode": ["First Class"]}), truth())[0], sid
    assert judge("D1", res_with({"Region": ["West"]}), truth())[0]
    assert not judge("D1", res_with({"Region": ["East"], "Sub-Category": ["Chairs"]}), truth())[0]
    assert not judge("D1", res_with({"City": ["X"]}, ["Sub-Category = Chairs"]), truth())[0]


def test_scenarios_and_questions_are_well_formed():
    ids = [s[0] for s in SCENARIOS]
    assert ids == ["A", "B", "C", "U", "D1", "D2", "CALM"]
    from llm.tools import TOOLS
    assert all(w in TOOLS or w == "none" for _, w in QUESTIONS)
    assert len({q for q, _ in QUESTIONS}) == len(QUESTIONS)


def test_report_lists_failures_and_never_hides_them():
    row = {"name": "demo", "scenarios": [
        {"id": i, "kpi": "revenue", "window": ["2024-01-01", "2024-01-31"], "expected": "x", "passed": i != "B",
         "detail": "found 0 of 5", "conclusion": "UNKNOWN", "seconds": 1.0} for i, *_ in SCENARIOS],
        "forecast": {"status": "LOW_CONFIDENCE", "coverage": 0.35, "n_checks": 69, "level": 0.8, "gap_to_target": 0.45}}
    text = report([row])
    assert "B:FAIL" in text and "A:PASS" in text and "6 of 7" in text
    assert "FAILED CHECKS" in text and "found 0 of 5" in text
    assert "35% of 69" in text and "LOW_CONFIDENCE" in text and "not a guarantee" in text


def test_main_with_nothing_to_run_says_so():
    assert main(["--no-demo"]) == 1


@needs_synthetic
def test_full_benchmark_on_the_demo_file_scores_every_scenario(synthetic):
    r = score_dataset(synthetic, truth())
    assert [s["id"] for s in r["scenarios"]] == [s[0] for s in SCENARIOS]
    failed = [(s["id"], s["detail"]) for s in r["scenarios"] if not s["passed"]]
    assert not failed, failed
    f = r["forecast"]
    assert f["status"] in ("OK", "LOW_CONFIDENCE", "UNKNOWN") and f["n_checks"] > 0
    json.dumps(r)
