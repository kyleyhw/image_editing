"""Studio backend (PROJECT_PLAN Phases 13-15).

Local-only FastAPI app. The browser renders every slider change itself with a
WebGL port of ``GlobalRenderer``, so interaction never waits on Python. The
server predicts parameters, renders full-resolution exports, and runs
style-learning jobs in the background.

    uv run python -m studio.server          # http://127.0.0.1:8765

Endpoints
---------
GET  /                              single-page app
GET  /api/styles                    style cards
POST /api/upload                    multipart file(s) -> photo ids
GET  /api/photos                    uploaded photos
GET  /api/photo/{id}?edge=1600      sRGB preview JPEG
POST /api/predict                   {id, style} -> EditParams (+ curves, derived, ood)
POST /api/batch_predict             {ids, style, consistency} -> [EditParams] (series consistency)
POST /api/export                    {id, params, format: jpg|cube|xmp|json} -> file
POST /api/learn                     multipart: name, mode=paired|unpaired, files -> job id
GET  /api/jobs/{job}                learning progress
POST /api/remember                  {id, style, params} -> store a user correction
POST /api/personalise               {style} -> fine-tune on remembered corrections (job)
"""

from __future__ import annotations

import copy
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image

from photostyle.engine import EditParams, Engine, save_stylepack
from photostyle.io import load_image, save_image, to_pil, to_tensor
from photostyle.train import learn_paired, learn_unpaired

ROOT = Path(os.environ.get("PHOTOSTYLE_STUDIO_DATA", "data/studio"))
UPLOADS = ROOT / "uploads"
STYLE_ROOT = Path(os.environ.get("PHOTOSTYLE_STYLES", "stylepacks"))
STATIC = Path(__file__).parent / "static"
DIST = Path(__file__).parent / "dist"

app = FastAPI(title="photostyle Studio")
engine = Engine(roots=[STYLE_ROOT])
jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _safe_name(name: str) -> str:
    """Style names become folder names: letters, digits, '-' and '_' only."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name or ""):
        raise HTTPException(400, "style name: 1-64 letters, digits, '-' or '_'")
    return name


def _photo_dir(pid: str) -> Path:
    if not pid.isalnum():
        raise HTTPException(400, "bad photo id")
    d = UPLOADS / pid
    if not d.exists():
        raise HTTPException(404, f"unknown photo {pid}")
    return d


def _params_payload(p: EditParams) -> dict:
    return {**json.loads(p.to_json()), "explain": explain(p)}


def explain(p: EditParams) -> str:
    """Plain-language summary of an edit (UI 'Explain' panel)."""
    c = p.curves()
    d = p.derived()
    parts = []
    black = min(c["r"][0], c["g"][0], c["b"][0])
    if black < -0.005 or c["r"][0] < 0.001 and c["g"][0] < 0.001 and c["b"][0] < 0.001 and black < 0:
        parts.append("deepened blacks")
    lift = [c[k][0] for k in "rgb"]
    if max(lift) > 0.01:
        parts.append("lifted shadows" + (" (blue)" if lift[2] > max(lift[0], lift[1]) + 0.005 else ""))
    mid = sum(c[k][len(c[k]) // 2] for k in "rgb") / 3
    if abs(mid - 0.5) > 0.02:
        parts.append(f"{'brightened' if mid > 0.5 else 'darkened'} midtones ({(mid - 0.5) * 200:+.0f} %)")
    if abs(d["warmth"]) > 0.01:
        parts.append(f"{'warmer' if d['warmth'] > 0 else 'cooler'} colour balance")
    if abs(d["saturation"] - 1) > 0.03:
        parts.append(f"{'more' if d['saturation'] > 1 else 'less'} saturated ({(d['saturation'] - 1) * 100:+.0f} %)")
    if d["vignette"] > 0.03:
        parts.append("light vignette")
    return (", ".join(parts) or "a very light touch").capitalize() + "."


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Studio v2 (Svelte, built into studio/dist); falls back to the legacy page if not built."""
    page = DIST / "index.html"
    return (page if page.exists() else STATIC / "index.html").read_text()


@app.get("/legacy", response_class=HTMLResponse)
def legacy() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/assets/{name}")
def assets(name: str):
    f = (DIST / "assets" / name).resolve()
    if not f.is_relative_to((DIST / "assets").resolve()) or not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/static/{name}")
def static(name: str):
    f = STATIC / name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/styles")
def styles():
    return engine.styles()


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    out = []
    for f in files:
        pid = uuid.uuid4().hex[:12]
        d = UPLOADS / pid
        d.mkdir(parents=True)
        raw = d / f"original{Path(f.filename or 'x.jpg').suffix.lower()}"
        raw.write_bytes(await f.read())
        img, info = load_image(raw)
        img.save(d / "srgb.jpg", quality=95)
        prev = img.copy()
        prev.thumbnail((1600, 1600))
        prev.save(d / "preview.jpg", quality=90)
        (d / "meta.json").write_text(json.dumps({"id": pid, "name": f.filename, "size": img.size,
                                                 "icc": info.get("icc")}))
        out.append({"id": pid, "name": f.filename, "size": img.size})
    return out


@app.get("/api/photos")
def photos():
    return [json.loads((d / "meta.json").read_text()) for d in sorted(UPLOADS.glob("*/")) if (d / "meta.json").exists()]


@app.get("/api/photo/{pid}")
def photo(pid: str, edge: int = 1600):
    d = _photo_dir(pid)
    img = Image.open(d / ("preview.jpg" if edge <= 1600 else "srgb.jpg"))
    img.thumbnail((edge, edge))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.post("/api/predict")
def predict(body: dict):
    d = _photo_dir(body["id"])
    p = engine.predict(Image.open(d / "srgb.jpg"), body["style"], strength=body.get("strength"))
    return _params_payload(p)


def _scene_maps(d: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth (nearness) and sky masks for the 1600 px preview, computed once per photo."""
    cache = d / "scene_maps.pt"
    if cache.exists():
        m = torch.load(cache)
        return m["near"], m["sky"]
    from photostyle.atmosphere import depth_map
    from photostyle.sky import sky_mask

    x = to_tensor(Image.open(d / "preview.jpg").convert("RGB"))[None]
    near, sky = depth_map(x), sky_mask(x)
    torch.save({"near": near, "sky": sky}, cache)
    return near, sky


@app.post("/api/scene")
def scene(body: dict):
    """Preview of the photo with scene tools applied (haze, clarity, sky light), before any grade.
    The browser renders the grade on top with WebGL, as for the plain photo."""
    from photostyle.atmosphere import SceneParams, apply_scene

    d = _photo_dir(body["id"])
    sp = SceneParams(**{k: float(v) for k, v in body.get("scene", {}).items() if k in SceneParams.__dataclass_fields__})
    img = Image.open(d / "preview.jpg").convert("RGB")
    if not sp.is_identity():
        near, sky = _scene_maps(d)
        img = to_pil(apply_scene(to_tensor(img)[None], sp, near=near, sky=sky)[0])
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.post("/api/batch_predict")
def batch_predict(body: dict):
    """Series consistency (Phase 15): shrink each photo's parameters toward the
    batch mean by ``consistency`` in [0, 1] (0 = fully per-photo, 1 = one shared edit)."""
    ps = [engine.predict(Image.open(_photo_dir(i) / "srgb.jpg"), body["style"]) for i in body["ids"]]
    c = float(body.get("consistency", 0.0))
    mean = torch.tensor([p.theta for p in ps]).mean(0)
    out = []
    for p in ps:
        th = (1 - c) * torch.tensor(p.theta) + c * mean
        p.theta = th.tolist()
        out.append(_params_payload(p))
    return out


@app.post("/api/export")
def export(body: dict):
    d = _photo_dir(body["id"])
    p = EditParams.from_json(json.dumps(body["params"]))
    fmt = body.get("format", "jpg")
    stem = Path(json.loads((d / "meta.json").read_text())["name"] or "photo").stem + f"_{p.style}"
    exp = d / "exports"
    exp.mkdir(exist_ok=True)
    if fmt == "jpg":
        src = next(d.glob("original.*"))
        img, info = load_image(src)
        path = exp / f"{stem}.jpg"
        save_image(engine.render(img, p), path, info)
    elif fmt == "cube":
        path = exp / f"{stem}.cube"
        p.to_cube(path)
    elif fmt == "xmp":
        path = exp / f"{stem}.xmp"
        p.to_xmp(path)
    else:
        path = exp / f"{stem}.json"
        p.to_json(path)
    return FileResponse(path, filename=path.name)


def _run_job(job_id: str, fn) -> None:
    def progress(step, total, v):
        with _lock:
            jobs[job_id].update(step=step, total=total, loss=v)
    try:
        fn(progress)
        with _lock:
            jobs[job_id]["status"] = "done"
        engine.refresh()
    except Exception as exc:  # surfaced to the UI
        with _lock:
            jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")


@app.post("/api/learn")
async def learn(name: str = Form(...), mode: str = Form(...), description: str = Form(""),
                before: list[UploadFile] = File(default=[]), after: list[UploadFile] = File(default=[]),
                examples: list[UploadFile] = File(default=[]), inputs: list[UploadFile] = File(default=[])):
    """Create-style wizard backend (Phase 14)."""
    name = _safe_name(name)
    async def tensors(files):
        out = []
        for f in files:
            img = Image.open(io.BytesIO(await f.read()))
            tmp = ROOT / "tmp" / (uuid.uuid4().hex + ".jpg")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            img.save(tmp, quality=95, icc_profile=img.info.get("icc_profile"))
            out.append((f.filename, to_tensor(load_image(tmp)[0])))
            tmp.unlink()
        return out

    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {"status": "running", "step": 0, "total": 1, "name": name, "mode": mode, "started": time.time()}
    if mode == "paired":
        b = {Path(n).stem: t for n, t in await tensors(before)}
        a = {Path(n).stem: t for n, t in await tensors(after)}
        pairs = [(b[k], a[k]) for k in sorted(set(b) & set(a))]
        jobs[job_id]["n"] = len(pairs)

        def fn(progress):
            head, r, feats, info = learn_paired(pairs, engine.fx, progress=progress)
            save_stylepack(STYLE_ROOT / name, name, head, r, feats,
                           {"source": "paired", "description": description, "train": info})
            jobs[job_id]["info"] = info
    else:
        ex = [t for _, t in await tensors(examples)]
        ins = [t for _, t in await tensors(inputs)]
        jobs[job_id]["n"] = len(ex)

        def fn(progress):
            head, r, feats, info = learn_unpaired(ex, ins, engine.fx, steps=600, progress=progress)
            save_stylepack(STYLE_ROOT / name, name, head, r, feats,
                           {"source": "unpaired", "description": description, "train": info})
            jobs[job_id]["info"] = info
    threading.Thread(target=_run_job, args=(job_id, fn), daemon=True).start()
    return {"job": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id]


# --------------------------------------------------------------------------- new-style pipeline
# The same steps as `photostyle style ...` (photostyle/newstyle.py), for Studio's Create-style dialog.


def _project(name: str):
    from photostyle import newstyle as ns

    return ns.Project.load(_safe_name(name))


def _project_payload(p) -> dict:
    from photostyle.newstyle import flagged

    return {"name": p.name, "description": p.description, "queries": p.queries,
            "candidates": [{"n": i + 1, "title": c.get("title"), "creator": c.get("creator"),
                            "license": c.get("license"), "art": flagged(p, i)} for i, c in enumerate(p.candidates)],
            "picks": [i + 1 for i in p.picks], "refs": [i + 1 for i in p.refs],
            "excluded": [i + 1 for i in p.excluded], "train": p.train}


@app.post("/api/style/new")
def style_new(body: dict):
    from photostyle import newstyle as ns

    queries = [q.strip() for q in body.get("queries", []) if q.strip()] or None
    try:
        p = ns.new(_safe_name(body["name"]).replace("-", "_"), body["describe"], queries, bool(body.get("mono")))
    except SystemExit as e:                      # e.g. the name is taken
        raise HTTPException(409, str(e)) from None
    return _project_payload(p)


@app.get("/api/style/{name}")
def style_get(name: str):
    return _project_payload(_project(name))


@app.get("/api/style/{name}/image/{n}")
def style_image(name: str, n: int, edge: int = 240):
    p = _project(name)
    if not 1 <= n <= len(p.candidates):
        raise HTTPException(404)
    img = Image.open(p.candidates[n - 1]["file"])
    img.thumbnail((edge, edge))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg")


def _job(fn) -> dict:
    job_id = uuid.uuid4().hex[:8]
    with _lock:
        jobs[job_id] = {"status": "running", "step": 0, "total": 1}
    threading.Thread(target=_run_job, args=(job_id, fn), daemon=True).start()
    return {"job": job_id}


@app.post("/api/style/search")
def style_search(body: dict):
    from photostyle import newstyle as ns

    name = _project(body["name"]).name
    return _job(lambda progress: ns.search(name, int(body.get("pages", 2)),
                                           progress=lambda i, n, c: progress(i, n, c)))


@app.post("/api/style/pick")
def style_pick(body: dict):
    from photostyle import newstyle as ns

    return _project_payload(ns.pick(_project(body["name"]).name, [int(n) for n in body["numbers"]],
                                    int(body.get("n_refs", 40))))


@app.post("/api/style/exclude")
def style_exclude(body: dict):
    from photostyle import newstyle as ns

    return _project_payload(ns.exclude(_project(body["name"]).name, [int(n) for n in body["numbers"]]))


@app.post("/api/style/restore")
def style_restore(body: dict):
    from photostyle import newstyle as ns

    return _project_payload(ns.restore(_project(body["name"]).name, [int(n) for n in body["numbers"]]))


@app.post("/api/style/train")
def style_train(body: dict):
    """Train, then write the style pack straight away so it appears in the style strip to try out."""
    from photostyle import newstyle as ns

    name, recipe = _project(body["name"]).name, body.get("recipe", "gentle")
    if recipe not in ("gentle", "strong", "instant"):
        raise HTTPException(400, "recipe: gentle, strong or instant (pairs: use 'My before/after edits')")

    def run(progress):
        ns.train(name, recipe, progress=progress)
        ns.pack(name, float(body.get("strength", 1.0)), STYLE_ROOT)
    return _job(run)


@app.post("/api/remember")
def remember(body: dict):
    """Store a user correction: (photo, final params) becomes a training example (Phase 15)."""
    d = _photo_dir(body["id"])
    corr = ROOT / "corrections" / body["style"]
    corr.mkdir(parents=True, exist_ok=True)
    (corr / f"{body['id']}.json").write_text(json.dumps({"id": body["id"], "photo": str(d / "srgb.jpg"),
                                                         "params": body["params"]}))
    return {"stored": len(list(corr.glob("*.json")))}


@app.post("/api/personalise")
def personalise(body: dict):
    """Fine-tune a style on remembered corrections (Phase 15).

    Each correction is a (photo, edited-by-user) pair rendered from the stored
    params; the style's head is fine-tuned on them with a small learning rate.
    """
    from photostyle.train import fit, paired_loss, shrink

    style = body["style"]
    corr = sorted((ROOT / "corrections" / style).glob("*.json"))
    if not corr:
        raise HTTPException(400, "no corrections stored for this style")
    pack = engine.style(style)
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {"status": "running", "step": 0, "total": 1, "name": style, "mode": "personalise"}

    def fn(progress):
        items = []
        for c in corr:
            rec = json.loads(c.read_text())
            src = to_tensor(Image.open(rec["photo"]))
            p = EditParams.from_json(json.dumps(rec["params"]))
            with torch.no_grad():
                tgt = p.make_renderer()(shrink(src, 256)[None], p.vector())[0]
            items.append({"feat": engine.fx(shrink(src, 512)), "src_t": shrink(src, 256), "tgt_t": tgt})
        head, r = copy.deepcopy(pack.head), pack.renderer  # never mutate the base style
        info = fit(head, r, items, items, paired_loss, steps=300, lr=5e-4, patience=6, progress=progress)
        feats = torch.stack([it["feat"] for it in items])
        name = style if style.endswith("_personal") else style + "_personal"
        save_stylepack(STYLE_ROOT / name, name, head, r, torch.cat([feats, pack.feat_mean[None]]),
                       {**pack.card, "name": name, "source": "personalised",
                        "description": f"{style} fine-tuned on {len(items)} of your corrections", "train": info})
        jobs[job_id]["info"] = info
    threading.Thread(target=_run_job, args=(job_id, fn), daemon=True).start()
    return {"job": job_id}


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    UPLOADS.mkdir(parents=True, exist_ok=True)
    print(f"photostyle Studio on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


@app.exception_handler(KeyError)
def _key_error(_req, exc):
    return JSONResponse({"detail": str(exc)}, status_code=404)


if __name__ == "__main__":
    run()
