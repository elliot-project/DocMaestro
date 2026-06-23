"""Task Visualizer – FastAPI server for SSL task inspection.

Discovers XML documents from output/, runs all SSL task composers on demand
using the central registry, and serves figure images and optional page renders
for the frontend.

Usage:
    uv run drafts/task_visualizer/server.py
    uv run drafts/task_visualizer/server.py --port 8767
"""

from __future__ import annotations

import argparse
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from src.task_composer import registry as task_registry  # noqa: E402

# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(ROOT / "output"))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--render-dpi", type=int, default=144)
    return p.parse_args()


ARGS = _parse_args()
OUTPUT_DIR = Path(ARGS.output_dir)
DATA_DIR = Path(ARGS.data_dir)
RENDER_DPI = ARGS.render_dpi
_SEED = 42

# ── Caches ─────────────────────────────────────────────────────────────────────

_registry: dict[str, dict] | None = None
_xml_cache: dict[str, ET.Element] = {}
_sample_cache: dict[str, list[dict]] = {}
_pdf_cache: dict[str, fitz.Document | None] = {}

# ── Discovery helpers ──────────────────────────────────────────────────────────


def _find_img_root(rel_folder: str) -> Path | None:
    """Return the parent of the best images/ directory for a doc folder."""
    search_base = DATA_DIR / rel_folder
    best: Path | None = None
    best_count = 0

    if search_base.exists():
        for d in search_base.rglob("images"):
            if d.is_dir():
                count = sum(1 for _ in d.glob("*"))
                if count > best_count:
                    best_count = count
                    best = d
        if best:
            return best.parent

    leaf = rel_folder.split("/")[-1]
    for d in DATA_DIR.rglob("images"):
        if d.is_dir() and leaf in str(d):
            count = sum(1 for _ in d.glob("*"))
            if count > best_count:
                best_count = count
                best = d
    return best.parent if best else None


def _find_pdf(rel_folder: str, doc_name: str) -> Path | None:
    for candidate in [
        DATA_DIR / "pdfs" / f"{doc_name}.pdf",
        DATA_DIR / rel_folder / f"{doc_name}.pdf",
    ]:
        if candidate.exists():
            return candidate
    search_root = DATA_DIR / rel_folder
    if search_root.exists():
        for p in search_root.rglob(f"{doc_name}.pdf"):
            return p
    for p in DATA_DIR.rglob(f"{doc_name}.pdf"):
        return p
    return None


def _doc_id(xml_path: Path) -> str:
    rel = xml_path.relative_to(OUTPUT_DIR).with_suffix("")
    return str(rel).replace("/", "__").replace("\\", "__")


def discover_docs() -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for xml_path in sorted(OUTPUT_DIR.rglob("*.xml")):
        try:
            root_el = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        if root_el.tag != "document":
            continue

        doc_name = root_el.attrib.get("name", xml_path.stem)
        rel_folder = str(xml_path.parent.relative_to(OUTPUT_DIR))
        did = _doc_id(xml_path)
        img_root = _find_img_root(rel_folder)
        pdf_path = _find_pdf(rel_folder, doc_name)

        docs[did] = {
            "doc_id": did,
            "doc_name": doc_name,
            "rel_folder": rel_folder,
            "xml_path": str(xml_path),
            "img_root": str(img_root) if img_root else None,
            "pdf_path": str(pdf_path) if pdf_path else None,
            "page_count": int(root_el.attrib.get("page_count", len(list(root_el)))),
        }
    return docs


def get_registry() -> dict[str, dict]:
    global _registry
    if _registry is None:
        _registry = discover_docs()
    return _registry


def _get_xml(doc_id: str) -> ET.Element:
    if doc_id not in _xml_cache:
        reg = get_registry()
        if doc_id not in reg:
            raise HTTPException(404, f"Doc '{doc_id}' not found")
        _xml_cache[doc_id] = ET.parse(reg[doc_id]["xml_path"]).getroot()
    return _xml_cache[doc_id]


def _get_pdf(doc_id: str) -> fitz.Document | None:
    if doc_id not in _pdf_cache:
        reg = get_registry()
        pdf_path = reg.get(doc_id, {}).get("pdf_path")
        _pdf_cache[doc_id] = fitz.open(pdf_path) if pdf_path else None
    return _pdf_cache[doc_id]


# ── Per-view-type serializers ──────────────────────────────────────────────────


def _img_url(doc_id: str, src: str) -> str:
    return f"/api/images/{doc_id}/{Path(src).name}"


def _find_img_src(xml_root: ET.Element, page_idx: int, element_tag: str, bbox: str) -> str | None:
    for page in xml_root:
        if page.attrib.get("idx") == str(page_idx):
            for elem in page:
                if elem.tag == element_tag and elem.attrib.get("bbox", "") == bbox:
                    img_el = elem.find("img")
                    return img_el.attrib.get("src") if img_el is not None else None
    return None


def _serialize_sample(
    sample, task_name: str, view_type: str, doc_id: str, xml_root: ET.Element
) -> dict:
    """Dispatch serialization based on view_type, returning a frontend-ready dict."""
    p = sample.payload
    meta = p.get("json", {})

    base = {
        "key": sample.key,
        "task_name": task_name,
        "view_type": view_type,
        "page_idx": meta.get("page_idx"),
    }

    if view_type == "image_grid":  # figure_selection
        label = p["target.json"]["label"]
        return {
            **base,
            "input_text": p.get("input.txt", ""),
            "label": label,
            "candidates": [
                {
                    "image_url": _img_url(doc_id, c["src"]),
                    "page_origin": c["page_origin"],
                    "is_correct": i == label,
                }
                for i, c in enumerate(meta.get("candidates", []))
            ],
        }

    elif view_type == "caption":  # caption_prediction
        img_src = _find_img_src(
            xml_root, meta.get("page_idx", -1), meta.get("element_tag", ""), meta.get("bbox", "")
        )
        return {
            **base,
            "element_tag": meta.get("element_tag"),
            "bbox": meta.get("bbox"),
            "preceding": p.get("input.json", {}).get("preceding", []),
            "following": p.get("input.json", {}).get("following", []),
            "gt_caption": p.get("target.txt", ""),
            "image_url": _img_url(doc_id, img_src) if img_src else None,
        }

    elif view_type == "reading_order":  # reading_order
        return {
            **base,
            "n_elements": meta.get("n_elements"),
            "shuffled": p.get("input.json", []),
            "target": p.get("target.json", []),
        }

    elif view_type == "text":  # basic_reading
        return {**base, "text": p.get("target.txt", "")}

    elif view_type == "bbox_overlay":  # ocr / layout
        return {**base, "blocks": p.get("target.json", [])}

    elif view_type == "markdown":  # markdown
        return {**base, "markdown": p.get("target.txt", "")}

    elif view_type == "xml_tree":  # page_tree
        return {**base, "xml": p.get("target.txt", "")}

    elif view_type == "xml_diff":  # dom_edition
        return {**base, "input_xml": p.get("input.txt", ""), "target_xml": p.get("target.txt", "")}

    elif view_type == "page_sequence":  # page_sorting
        return {
            **base,
            "n_pages": meta.get("n_pages"),
            "shuffled": p.get("input.json", []),
            "target": p.get("target.json", []),
        }

    elif view_type == "canvas_layout":  # doc_reconstruction
        return {**base, "elements": p.get("input.json", []), "target": p.get("target.json", [])}

    elif view_type == "figure_page_map":  # figure_ordering
        inp = p.get("input.json", {})
        return {
            **base,
            "pages": inp.get("pages", []),
            "figures": inp.get("figures", []),
            "target": p.get("target.json", []),
        }

    elif view_type == "kv_form":  # form_filling
        return {
            **base,
            "fields": p.get("target.json", {}).get("fields", []),
            "width": meta.get("width"),
            "height": meta.get("height"),
        }

    # Fallback: dump raw payload keys (excluding __key__)
    return {**base, "raw": {k: v for k, v in p.items() if k != "__key__"}}


def generate_samples(doc_id: str, task_name: str) -> list[dict]:
    cache_key = f"{doc_id}::{task_name}"
    if cache_key in _sample_cache:
        return _sample_cache[cache_key]

    reg = get_registry()
    if doc_id not in reg:
        raise HTTPException(404, f"Doc '{doc_id}' not found")

    if task_name not in task_registry.REGISTRY:
        raise HTTPException(400, f"Unknown task '{task_name}'")

    xml_root = _get_xml(doc_id)
    img_root = Path(reg[doc_id]["img_root"]) if reg[doc_id]["img_root"] else ROOT / "_nonexistent"
    spec = task_registry.REGISTRY[task_name]
    cfg = {**spec.default_config}
    if "seed" in cfg:
        cfg["seed"] = _SEED

    composer = spec.composer_factory(**cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_samples = composer.compose_document(xml_root, img_root)

    result = [
        _serialize_sample(s, task_name, spec.view_type, doc_id, xml_root) for s in raw_samples
    ]
    _sample_cache[cache_key] = result
    return result


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Task Visualizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HERE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(HERE)), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse((HERE / "index.html").read_text())


@app.get("/api/tasks")
def list_tasks():
    """Return all registered tasks with their view_type and description."""
    return [
        {"task_name": s.task_name, "view_type": s.view_type, "description": s.description}
        for s in task_registry.list_tasks()
    ]


@app.get("/api/docs")
def list_docs():
    reg = get_registry()
    return sorted(
        [
            {
                "doc_id": m["doc_id"],
                "doc_name": m["doc_name"],
                "rel_folder": m["rel_folder"],
                "page_count": m["page_count"],
                "has_images": m["img_root"] is not None,
                "has_pdf": m["pdf_path"] is not None,
            }
            for m in reg.values()
        ],
        key=lambda x: x["doc_name"],
    )


@app.get("/api/docs/{doc_id}/samples/{task_name}")
def get_samples(doc_id: str, task_name: str):
    samples = generate_samples(doc_id, task_name)
    return {"task_name": task_name, "total": len(samples), "samples": samples}


@app.get("/api/docs/{doc_id}/pages/{page_idx}/image")
def page_image(doc_id: str, page_idx: int):
    pdf = _get_pdf(doc_id)
    if pdf is None:
        raise HTTPException(404, "No PDF for this document")
    if page_idx < 0 or page_idx >= len(pdf):
        raise HTTPException(404, f"Page {page_idx} out of range")
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = pdf[page_idx].get_pixmap(matrix=mat)
    return Response(content=pix.tobytes("png"), media_type="image/png")


@app.get("/api/images/{doc_id}/{filename}")
def serve_image(doc_id: str, filename: str):
    reg = get_registry()
    img_root_str = reg.get(doc_id, {}).get("img_root")
    if not img_root_str:
        raise HTTPException(404, "No image root for this document")
    img_path = Path(img_root_str) / "images" / filename
    if not img_path.exists():
        raise HTTPException(404, f"Image not found: {filename}")
    suffix = img_path.suffix.lower()
    media_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return Response(content=img_path.read_bytes(), media_type=media_type)


if __name__ == "__main__":
    docs = get_registry()
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Data dir   : {DATA_DIR}")
    print(f"Documents  : {len(docs)} found")
    print(f"Tasks      : {len(task_registry.REGISTRY)} registered")
    for did, m in sorted(docs.items(), key=lambda x: x[1]["doc_name"]):
        imgs = "imgs=yes" if m["img_root"] else "imgs=no "
        print(
            f"  {m['doc_name']:42s} {m['page_count']:4d} pages  {imgs}  pdf={'yes' if m['pdf_path'] else 'no'}"
        )
    print(f"\nStarting at http://{ARGS.host}:{ARGS.port}")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
