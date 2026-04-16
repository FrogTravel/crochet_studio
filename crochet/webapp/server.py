"""FastAPI backend for the Crochet Instruction Studio.

Endpoints:

* ``GET  /``                         - the single-page app.
* ``GET  /api/capabilities``         - which optional features are live.
* ``POST /api/generate``             - start a job, return ``{job_id}``.
* ``GET  /api/events/{job_id}``      - Server-Sent Events stream of progress.
* ``GET  /api/result/{job_id}/html`` - the styled tutorial HTML.
* ``GET  /api/result/{job_id}/md``   - the plain-markdown fallback.
* ``GET  /files/...``                - generated images / meshes (static).

The router runs on a background thread; each callback pushes an event
onto an ``asyncio.Queue`` via ``run_coroutine_threadsafe`` so the SSE
generator can forward it to the browser unchanged.
"""

from __future__ import annotations

import os

# Force a non-interactive matplotlib backend BEFORE anything imports pyplot.
# The router renders figures on a worker thread; the default macOS backend
# refuses to create a FigureManager off the main thread, so we lock in Agg.
os.environ.setdefault("MPLBACKEND", "Agg")
try:  # pragma: no cover - matplotlib may not be installed in every env
    import matplotlib  # type: ignore

    matplotlib.use("Agg", force=True)
except Exception:
    pass

import asyncio
import json
import shutil
import sys
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles


# ---------------------------------------------------------------------------
# Paths & dotenv bootstrapping
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STATIC_DIR = Path(__file__).resolve().parent / "static"
WORK_DIR = REPO_ROOT / "notebooks_output" / "webapp"
WORK_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> Path | None:
    for candidate in (REPO_ROOT / "notebooks" / ".env", REPO_ROOT / ".env"):
        if candidate.is_file():
            for raw in candidate.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return candidate
    return None


ENV_PATH = _load_dotenv()

from crochet.palette import PaletteColor  # noqa: E402
from crochet.routing import capabilities, run_router  # noqa: E402


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Crochet Instruction Studio",
    description="Turn a prompt or photo into a professionally-styled crochet tutorial.",
    version="1.0.0",
)

app.mount("/files", StaticFiles(directory=WORK_DIR, html=False), name="files")
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static")


# ---------------------------------------------------------------------------
# In-memory job registry
# ---------------------------------------------------------------------------

JOBS: dict[str, dict[str, Any]] = {}


def _new_job(prompt: str, image_path: Path | None) -> str:
    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = {
        "queue":      asyncio.Queue(maxsize=0),
        "work_dir":   job_dir,
        "image_path": image_path,
        "prompt":     prompt,
        "started":    False,
        "result":     None,
        "error":      None,
    }
    return job_id


def _to_public_url(path: Path, job_dir: Path, job_id: str) -> str:
    """Resolve a Path to a /files/ URL; copy outside files in if needed."""
    p = Path(path)
    try:
        rel = p.resolve().relative_to(WORK_DIR.resolve())
        return f"/files/{rel.as_posix()}"
    except Exception:
        dest = job_dir / p.name
        if p.is_file() and not dest.exists():
            try:
                shutil.copyfile(p, dest)
            except Exception:
                return ""
        return f"/files/{job_id}/{dest.name}"


def _start_worker(job_id: str) -> None:
    """Kick off the router on a background thread for ``job_id``."""
    job = JOBS[job_id]
    if job["started"]:
        return
    job["started"] = True

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = job["queue"]
    job_dir: Path = job["work_dir"]

    def publish(event: str, data: dict[str, Any]) -> None:
        try:
            asyncio.run_coroutine_threadsafe(q.put((event, data)), loop)
        except RuntimeError:
            # Loop already closed; nothing we can do.
            pass

    def worker() -> None:
        try:
            def on_phase(name: str, pct: float) -> None:
                publish("phase", {"name": name, "pct": float(pct)})

            def on_image(path: Path, label: str) -> None:
                url = _to_public_url(path, job_dir, job_id)
                if url:
                    publish("image", {"url": url, "label": label or Path(path).name})

            def on_palette(colors: list[PaletteColor]) -> None:
                publish(
                    "palette",
                    {"colors": [c.as_dict() for c in colors]},
                )

            def on_log(msg: str) -> None:
                publish("log", {"msg": msg})

            result = run_router(
                prompt=job["prompt"],
                image_path=job["image_path"],
                work_dir=job_dir,
                on_phase=on_phase,
                on_image=on_image,
                on_palette=on_palette,
                on_log=on_log,
            )
            job["result"] = result
            rows = result.get("amigurumi_rows") or []
            publish(
                "result",
                {
                    "job_id":         job_id,
                    "path":           result.get("path"),
                    "label":          result.get("label"),
                    "scores":         result.get("scores", {}),
                    "num_detections": int(result.get("num_detections", 0)),
                    "rounds":         len(rows),
                    "counts":         result.get("counts", {}),
                    "palette":        result.get("palette", []),
                    "note":           result.get("note"),
                    "has_html":       bool(result.get("instruction_html")),
                    "has_md":         bool(result.get("instruction")),
                    "mesh_source":    result.get("mesh_source"),
                },
            )
        except Exception as exc:  # noqa: BLE001 - surface everything
            job["error"] = f"{type(exc).__name__}: {exc}"
            publish("error", {"message": job["error"]})
        finally:
            publish("done", {})

    threading.Thread(target=worker, name=f"job-{job_id}", daemon=True).start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>Frontend missing.</h1>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/capabilities")
async def api_capabilities() -> JSONResponse:
    caps = capabilities()
    return JSONResponse({
        **caps,
        "env_path": str(ENV_PATH.relative_to(REPO_ROOT)) if ENV_PATH else None,
    })


@app.post("/api/generate")
async def api_generate(
    prompt: str = Form(""),
    image: UploadFile | None = File(None),
) -> JSONResponse:
    prompt = (prompt or "").strip()
    if not prompt and image is None:
        raise HTTPException(status_code=400, detail="Provide a prompt or an image.")

    # Persist upload into a new job dir.
    image_path: Path | None = None
    tmp_job_dir = WORK_DIR / f"_pending-{uuid.uuid4().hex[:8]}"
    if image is not None and image.filename:
        tmp_job_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(image.filename).suffix or ".png"
        image_path = tmp_job_dir / f"upload{suffix}"
        image_path.write_bytes(await image.read())

    job_id = _new_job(prompt, image_path)

    # Move the upload under the real job dir so the served URL is stable.
    if image_path is not None:
        dest = JOBS[job_id]["work_dir"] / image_path.name
        shutil.move(str(image_path), dest)
        JOBS[job_id]["image_path"] = dest
        try:
            tmp_job_dir.rmdir()
        except OSError:
            pass

    return JSONResponse({"job_id": job_id})


@app.get("/api/events/{job_id}")
async def api_events(job_id: str, request: Request) -> StreamingResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")

    _start_worker(job_id)

    async def stream():
        q: asyncio.Queue = job["queue"]
        # Initial hello so the client shows "connected" immediately.
        yield "event: hello\ndata: {}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                event, data = await asyncio.wait_for(q.get(), timeout=10.0)
            except asyncio.TimeoutError:
                # keepalive
                yield ": ping\n\n"
                continue
            payload = json.dumps(data, default=str)
            yield f"event: {event}\ndata: {payload}\n\n"
            if event == "done":
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/result/{job_id}/html", response_class=HTMLResponse)
async def api_result_html(job_id: str) -> HTMLResponse:
    job = JOBS.get(job_id)
    if job is None or job.get("result") is None:
        raise HTTPException(status_code=404, detail="no result")
    html = job["result"].get("instruction_html") or ""
    return HTMLResponse(html)


@app.get("/api/result/{job_id}/md", response_class=PlainTextResponse)
async def api_result_md(job_id: str) -> PlainTextResponse:
    job = JOBS.get(job_id)
    if job is None or job.get("result") is None:
        raise HTTPException(status_code=404, detail="no result")
    md = job["result"].get("instruction") or ""
    return PlainTextResponse(md)


@app.get("/api/result/{job_id}/download/{kind}")
async def api_result_download(job_id: str, kind: str) -> Response:
    job = JOBS.get(job_id)
    if job is None or job.get("result") is None:
        raise HTTPException(status_code=404, detail="no result")
    r = job["result"]
    if kind == "html":
        body = (r.get("instruction_html") or "").encode("utf-8")
        return Response(
            content=body,
            media_type="text/html",
            headers={"Content-Disposition": 'attachment; filename="tutorial.html"'},
        )
    if kind == "md":
        body = (r.get("instruction") or "").encode("utf-8")
        return Response(
            content=body,
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="tutorial.md"'},
        )
    if kind == "glb":
        glb_path = r.get("glb_path")
        if glb_path and Path(glb_path).is_file():
            return FileResponse(
                glb_path,
                media_type="model/gltf-binary",
                filename="mesh.glb",
            )
        raise HTTPException(status_code=404, detail="no mesh")
    raise HTTPException(status_code=400, detail=f"unknown kind: {kind}")


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "jobs": len(JOBS)})
