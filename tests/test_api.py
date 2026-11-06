"""Tests for api/main.py (Step 11): every URL, with FastAPI's TestClient.

    pytest tests/test_api.py -v

Needs:  pip install fastapi httpx python-multipart   (they are in requirements.txt).
The logic behind the URLs is tested in tests/test_api_service.py; here we check the web layer:
status codes, JSON shapes, upload, validation errors, CORS and that unexpected errors show no internals.
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.main import create_app  # noqa: E402
from api.service import Service  # noqa: E402
from engine.ingest import load_dataset  # noqa: E402
from llm.client import FakeLLM  # noqa: E402

SYNTHETIC = ROOT / "data" / "synthetic_demo.csv"
needs_synthetic = pytest.mark.skipif(not SYNTHETIC.exists(), reason="synthetic file missing")
CSV = (b"Order Date,Sales,Profit,Region,Order ID\n" +
       b"".join(f"2023-{(i % 12) + 1:02d}-{(i % 27) + 1:02d},{100 + i},{10 + i % 5},{'East' if i % 2 else 'West'},O{i}\n".encode()
                for i in range(60)))


def make_client(llm=None, **kw):
    svc = Service(llm=llm, loader=load_dataset, demos={"synthetic": SYNTHETIC}, **kw)
    return TestClient(create_app(svc), raise_server_exceptions=False), svc


def test_health_and_tools_listing():
    c, _ = make_client()
    h = c.get("/api/health")
    assert h.status_code == 200 and h.json()["ok"] is True and h.json()["llm"]["configured"] is False
    t = c.get("/api/tools").json()
    assert len(t) == 8 and {"name", "help", "params", "required"} <= set(t[0])


def test_upload_summary_headlines_and_delete():
    c, _ = make_client()
    r = c.post("/api/upload", files={"file": ("small.csv", CSV, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    sid = body["session_id"]
    assert body["rows"] == 60 and body["kpis"][0]["key"] == "revenue"
    assert c.get(f"/api/sessions/{sid}/summary").json()["name"] == "small.csv"
    h = c.get(f"/api/sessions/{sid}/headlines", params={"freq": "M"})
    assert h.status_code == 200 and h.json()["headlines"]
    assert c.delete(f"/api/sessions/{sid}").json() == {"deleted": True}
    assert c.get(f"/api/sessions/{sid}/summary").status_code == 404


def test_bad_uploads_and_unknown_things_have_clear_status_codes():
    c, _ = make_client(max_upload_bytes=500)
    assert c.post("/api/upload", files={"file": ("a.exe", b"x", "application/octet-stream")}).status_code == 400
    assert c.post("/api/upload", files={"file": ("big.csv", CSV, "text/csv")}).status_code == 413
    assert c.post("/api/upload", files={"file": ("empty.csv", b"", "text/csv")}).status_code == 400
    assert c.post("/api/upload").status_code == 422                       # no file at all
    r = c.get("/api/sessions/nope/summary")
    assert r.status_code == 404 and "expired" in r.json()["detail"]
    assert c.post("/api/demo", json={"name": "nope"}).status_code == 404
    assert c.get("/api/nothing").status_code == 404


@needs_synthetic
def test_demo_alerts_and_alert_verification():
    c, _ = make_client()
    sid = c.post("/api/demo", json={"name": "synthetic"}).json()["session_id"]
    a = c.get(f"/api/sessions/{sid}/alerts", params={"kpi": "profit_margin", "top_n": 5})
    assert a.status_code == 200 and a.json()["events"][0]["kpi"] == "profit_margin"
    assert c.get(f"/api/sessions/{sid}/alerts", params={"top_n": 0}).status_code == 422
    assert c.get(f"/api/sessions/{sid}/alerts", params={"sensitivity": "extreme"}).status_code == 400
    sr = c.get(f"/api/sessions/{sid}/series", params={"kpi": "revenue", "freq": "M"})
    assert sr.status_code == 200 and len(sr.json()["points"]) == 36
    assert c.get(f"/api/sessions/{sid}/series", params={"kpi": "nope"}).status_code == 400
    ev = a.json()["events"][0]
    w = c.post(f"/api/sessions/{sid}/alerts/why", json={"kpi": ev["kpi"], "start": ev["start"]})
    assert w.status_code == 200 and w.json()["evidence"]["records"][0]["id"] == "E1"


@needs_synthetic
def test_tool_and_ask_endpoints():
    llm = FakeLLM([json.dumps({"tool": "kpi_value", "args": {"kpi": "revenue", "start": "2024", "end": "2024"}})])
    c, _ = make_client(llm=llm)
    sid = c.post("/api/demo", json={}).json()["session_id"]                # default demo name
    ok = c.post(f"/api/sessions/{sid}/tool", json={"tool": "kpi_value", "args": {"kpi": "revenue"}, "use_llm": False})
    assert ok.status_code == 200 and ok.json()["tool"] == "kpi_value"
    bad = c.post(f"/api/sessions/{sid}/tool", json={"tool": "sql", "args": {"query": "DROP TABLE sales"}, "use_llm": False})
    assert bad.status_code == 400 and "rejected" in bad.json()["detail"]
    assert c.post(f"/api/sessions/{sid}/tool", json={"args": {}}).status_code == 422       # tool missing
    asked = c.post(f"/api/sessions/{sid}/ask", json={"question": "What was revenue in 2024?"})
    assert asked.status_code == 200 and asked.json()["status"] == "ok" and asked.json()["evidence"]["records"]
    assert c.post(f"/api/sessions/{sid}/ask", json={"question": "hi", "use_llm": False}).json()["status"] == "no_llm"
    assert c.post(f"/api/sessions/{sid}/ask", json={}).status_code == 422


def test_cors_allows_the_react_dev_server_and_not_other_sites():
    c, _ = make_client()
    ok = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    evil = c.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
    pre = c.options("/api/sessions/x/ask", headers={"Origin": "http://localhost:5173",
                                                    "Access-Control-Request-Method": "POST",
                                                    "Access-Control-Request-Headers": "content-type"})
    assert pre.status_code == 200


def test_unexpected_errors_return_a_generic_500():
    c, svc = make_client()

    def boom():
        raise RuntimeError("secret internal detail C:\\Users\\me\\.env")
    svc.health = boom
    r = c.get("/api/health")
    assert r.status_code == 500
    assert "secret" not in r.text and "RuntimeError" not in r.text and "Something went wrong" in r.text
