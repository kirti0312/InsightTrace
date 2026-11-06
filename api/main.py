"""FastAPI web layer: URLs -> api.service.Service. No logic lives here.

Run (from the project folder, with the virtual environment active):
    uvicorn api.main:app --reload --port 8000
Then open http://127.0.0.1:8000/docs for the automatic, clickable API documentation.

Settings (environment variables or .env):
    GEMINI_API_KEY, GEMINI_MODEL, LLM_MODE=off   see llm/client.py
    CORS_ORIGINS   comma-separated website addresses allowed to call this API
                   (default: the local React dev server). Add your deployed frontend address here.
    MAX_UPLOAD_MB  largest upload in megabytes (default 25)

Note: this file uses FastAPI features that were NOT run by the person who wrote it (the build sandbox
had no FastAPI). tests/test_api.py exercises every route with FastAPI's TestClient; run it.
"""
import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.service import ApiError, Service
from llm.client import get_llm

try:                                   # read .env (the API key) if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class DemoBody(BaseModel):
    name: str = "synthetic"


class ToolBody(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    use_llm: bool = True


class AskBody(BaseModel):
    question: str
    use_llm: bool = True


class AlertWhyBody(BaseModel):
    kpi: str
    start: str
    freq: str = "M"
    use_llm: bool = True


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ApiError as err:
        raise HTTPException(status_code=err.status, detail=err.message)


def create_app(service: Service | None = None) -> FastAPI:
    max_mb = float(os.environ.get("MAX_UPLOAD_MB", "25"))
    svc = service or Service(llm=get_llm(), max_upload_bytes=int(max_mb * 1024 * 1024))
    app = FastAPI(title="InsightTrace API", version="0.1")
    origins = [o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["*"])
    app.state.service = svc

    @app.exception_handler(Exception)
    async def unexpected(request, exc):                      # never show internals to the browser
        return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server."})

    @app.get("/api/health")
    def health():
        return svc.health()

    @app.get("/api/tools")
    def tools():
        return svc.tools()

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read(svc.max_upload_bytes + 1)
        return await run_in_threadpool(_run, svc.upload, data, file.filename or "")   # parsing can take seconds

    @app.post("/api/demo")
    def demo(body: DemoBody):
        return _run(svc.demo, body.name)

    @app.get("/api/sessions/{sid}/summary")
    def summary(sid: str):
        return _run(svc.summary, sid)

    @app.delete("/api/sessions/{sid}")
    def delete(sid: str):
        return _run(svc.delete, sid)

    @app.get("/api/sessions/{sid}/headlines")
    def headlines(sid: str, freq: str = "M"):
        return _run(svc.headlines, sid, freq)

    @app.get("/api/sessions/{sid}/alerts")
    def alerts(sid: str, freq: str = "M", kpi: str | None = None, primary_only: bool = True,
               sensitivity: str = "normal", top_n: int = Query(20, ge=1, le=50)):
        return _run(svc.alerts, sid, freq, kpi, primary_only, sensitivity, top_n)

    @app.get("/api/sessions/{sid}/series")
    def series(sid: str, kpi: str = "revenue", freq: str = "M", sensitivity: str = "normal"):
        return _run(svc.series, sid, kpi, freq, sensitivity)

    @app.post("/api/sessions/{sid}/alerts/why")
    def alert_why(sid: str, body: AlertWhyBody):
        return _run(svc.alert_why, sid, body.kpi, body.start, body.freq, body.use_llm)

    @app.post("/api/sessions/{sid}/tool")
    def tool(sid: str, body: ToolBody):
        return _run(svc.tool, sid, body.tool, body.args, body.use_llm)

    @app.post("/api/sessions/{sid}/ask")
    def ask(sid: str, body: AskBody):
        return _run(svc.ask, sid, body.question, body.use_llm)

    return app


app = create_app()
