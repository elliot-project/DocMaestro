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
import ast
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


# ── XML helpers ────────────────────────────────────────────────────────────────


def _parse_bbox(bbox_str: str | None) -> list[float] | None:
    """Parse a bbox string like '[x1, y1, x2, y2]' into a list of 4 floats."""
    if not bbox_str:
        return None
    try:
        v = ast.literal_eval(bbox_str)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return [float(x) for x in v]
    except Exception:
        pass
    return None


def _get_page_el(xml_root: ET.Element, page_idx: int) -> ET.Element | None:
    for page in xml_root:
        if page.attrib.get("idx") == str(page_idx):
            return page
    return None


def _page_dims(xml_root: ET.Element, page_idx: int | None) -> tuple[float, float]:
    if page_idx is None:
        return 0.0, 0.0
    page = _get_page_el(xml_root, page_idx)
    if page is None:
        return 0.0, 0.0
    return float(page.attrib.get("width", 0)), float(page.attrib.get("height", 0))


def _all_page_elements(xml_root: ET.Element, page_idx: int) -> list[dict]:
    """Return all direct child elements of a page that carry a bbox."""
    page = _get_page_el(xml_root, page_idx)
    if page is None:
        return []
    result = []
    for el in page:
        bbox = _parse_bbox(el.attrib.get("bbox"))
        if bbox:
            result.append(
                {
                    "tag": el.tag,
                    "bbox": bbox,
                    "text": (el.text or "").strip()[:80],
                }
            )
    return result


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


# ── Highlight bbox builders ────────────────────────────────────────────────────
# Each returns a list of {bbox: [x1,y1,x2,y2], color: str, label: str}


def _hl(bbox_str: str | None, color: str, label: str = "") -> dict | None:
    b = _parse_bbox(bbox_str)
    return {"bbox": b, "color": color, "label": label} if b else None


def _build_highlights(
    view_type: str, sample, p: dict, meta: dict, xml_root: ET.Element
) -> list[dict]:
    """Build the highlight_bboxes list for the page canvas overlay."""
    page_idx = meta.get("page_idx")
    highlights: list[dict] = []

    if view_type == "caption":
        h = _hl(meta.get("bbox"), "#50E3C2", meta.get("element_tag", ""))
        if h:
            highlights.append(h)

    elif view_type == "bbox_overlay":
        blocks = p.get("target.json", [])
        color = "#F5A623"  # OCR/layout = orange
        for b in blocks:
            h = _hl(b.get("bbox"), color, b.get("tag", ""))
            if h:
                highlights.append(h)

    elif view_type == "reading_order":
        # Shuffled items have {idx, tag, text} but no bbox.
        # Recover bboxes by matching tag+text against the XML page.
        if page_idx is not None:
            page = _get_page_el(xml_root, page_idx)
            if page is not None:
                shuffled = p.get("input.json", [])
                colors = [
                    "#4A90E2",
                    "#7ED321",
                    "#BD10E0",
                    "#F5A623",
                    "#50E3C2",
                    "#E91E8C",
                    "#FF7043",
                    "#00BCD4",
                    "#B388FF",
                    "#F5E642",
                ]
                for item in shuffled:
                    # Match by tag and truncated text
                    target_text = item.get("text", "")
                    for el in page:
                        if el.tag == item.get("tag"):
                            el_text = (el.text or "").strip()[:120]
                            if el_text == target_text and "bbox" in el.attrib:
                                col = colors[item["idx"] % len(colors)]
                                h = _hl(el.attrib["bbox"], col, str(item["idx"]))
                                if h:
                                    highlights.append(h)
                                break

    elif view_type == "image_grid":
        # Highlight all figures on the page; green = correct, dim = distractor
        if page_idx is not None:
            candidates = meta.get("candidates", [])
            page = _get_page_el(xml_root, page_idx)
            if page is not None:
                fig_els = [el for el in page if el.tag == "figure" and "bbox" in el.attrib]
                for i, el in enumerate(fig_els):
                    # Try to match with candidate by src
                    is_correct = i < len(candidates) and candidates[i].get("is_correct")
                    col = "#7ED321" if is_correct else "#78909C"
                    h = _hl(el.attrib["bbox"], col, "✓" if is_correct else "")
                    if h:
                        highlights.append(h)

    elif view_type == "canvas_layout":
        # Highlight target bboxes
        bboxes = p.get("target.json", [])
        colors = [
            "#4A90E2",
            "#7ED321",
            "#BD10E0",
            "#F5A623",
            "#50E3C2",
            "#E91E8C",
            "#FF7043",
            "#00BCD4",
        ]
        for i, bbox_str in enumerate(bboxes):
            h = _hl(bbox_str, colors[i % len(colors)], str(i))
            if h:
                highlights.append(h)

    elif view_type in ("text", "markdown", "xml_tree", "xml_diff"):
        # Light highlight of all elements — use one neutral colour
        if page_idx is not None:
            for el_info in _all_page_elements(xml_root, page_idx):
                highlights.append(
                    {
                        "bbox": el_info["bbox"],
                        "color": "#4A90E2",
                        "label": "",
                    }
                )

    return highlights


# ── Sample serialization ───────────────────────────────────────────────────────


def _serialize_sample(
    sample, task_name: str, view_type: str, doc_id: str, xml_root: ET.Element
) -> dict:
    p = sample.payload
    meta = p.get("json", {})
    page_idx = meta.get("page_idx")

    # Page geometry
    pw, ph = _page_dims(xml_root, page_idx)
    highlights = _build_highlights(view_type, sample, p, meta, xml_root)

    base = {
        "key": sample.key,
        "task_name": task_name,
        "view_type": view_type,
        "page_idx": page_idx,
        "page_width": pw,
        "page_height": ph,
        "highlight_bboxes": highlights,
        "has_page": page_idx is not None and pw > 0,
    }

    if view_type == "image_grid":
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

    elif view_type == "caption":
        img_src = _find_img_src(
            xml_root, page_idx or -1, meta.get("element_tag", ""), meta.get("bbox", "")
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

    elif view_type == "reading_order":
        return {
            **base,
            "n_elements": meta.get("n_elements"),
            "shuffled": p.get("input.json", []),
            "target": p.get("target.json", []),
        }

    elif view_type == "text":
        return {**base, "text": p.get("target.txt", "")}

    elif view_type == "bbox_overlay":
        return {**base, "blocks": p.get("target.json", [])}

    elif view_type == "markdown":
        return {**base, "markdown": p.get("target.txt", "")}

    elif view_type == "xml_tree":
        return {**base, "xml": p.get("target.txt", "")}

    elif view_type == "xml_diff":
        return {**base, "input_xml": p.get("input.txt", ""), "target_xml": p.get("target.txt", "")}

    elif view_type == "page_sequence":
        return {
            **base,
            "n_pages": meta.get("n_pages"),
            "shuffled": p.get("input.json", []),
            "target": p.get("target.json", []),
        }

    elif view_type == "canvas_layout":
        return {**base, "elements": p.get("input.json", []), "target": p.get("target.json", [])}

    elif view_type == "figure_page_map":
        inp = p.get("input.json", {})
        return {
            **base,
            "pages": inp.get("pages", []),
            "figures": inp.get("figures", []),
            "target": p.get("target.json", []),
        }

    elif view_type == "kv_form":
        return {
            **base,
            "fields": p.get("target.json", {}).get("fields", []),
            "width": meta.get("width"),
            "height": meta.get("height"),
        }

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


@app.get("/api/docs/{doc_id}/pages/{page_idx}/elements")
def page_elements(doc_id: str, page_idx: int):
    """Return all bbox-bearing elements on a page for the grey background layer."""
    xml_root = _get_xml(doc_id)
    elements = _all_page_elements(xml_root, page_idx)
    pw, ph = _page_dims(xml_root, page_idx)
    return {"page_idx": page_idx, "page_width": pw, "page_height": ph, "elements": elements}


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
