"""Draft utility — figure classification label viewer.

Reads a pipeline-produced XML file, finds all classified <figure> elements,
and generates a self-contained HTML page showing each figure image alongside
its SigLIP2 category label and confidence score.

Usage:
    uv run drafts/figure_label_viewer.py --xml output/macdonals/macdonals_hybrid-auto-engine.xml
    uv run drafts/figure_label_viewer.py --xml output/macdonals/macdonals_hybrid-auto-engine.xml \
        --img-root data/macdonals/hybrid-auto-engine/macdonals/hybrid_auto
    uv run drafts/figure_label_viewer.py --xml output/macdonals/macdonals_hybrid-auto-engine.xml \
        --out drafts/review.html --no-open
"""

from __future__ import annotations

import argparse
import base64
import webbrowser
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Label colour palette ───────────────────────────────────────────────────────

_LABEL_COLORS: dict[str, str] = {
    "natural-image":          "#22c55e",
    "diagram":                "#3b82f6",
    "scientific-medical-image": "#a855f7",
    "illustration-artwork":   "#f97316",
    "screenshot":             "#6b7280",
    "infographic":            "#14b8a6",
    "logo-icon":              "#eab308",
    "map":                    "#92400e",
    "signature-handwriting":  "#ec4899",
    "rasterized-table":       "#ef4444",
    "unknown":                "#374151",
}


# ── Data extraction ────────────────────────────────────────────────────────────

def _find_img_root(xml_path: Path) -> Path:
    """Guess img_root by walking from data/ given an output/ XML path.

    Heuristic: output/<doc>/<doc>_<backend>.xml  →  data/<doc>/<backend>/<doc>/hybrid_auto
    Falls back to searching the whole data/ directory for an images/ folder.
    """
    # Try to derive from output path structure
    doc_name = xml_path.stem.split("_")[0]
    data_root = xml_path.parents[2] / "data"
    if data_root.exists():
        candidates = sorted(
            data_root.rglob("images"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].parent
    return xml_path.parent


def _load_figures(xml_path: Path, img_root: Path) -> list[dict]:
    """Extract figure metadata from the XML file.

    Args:
        xml_path: Path to the classified XML file.
        img_root: Root directory for resolving <img src> paths.

    Returns:
        List of dicts with keys: page, bbox, category, confidence, caption, img_path.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    figures = []

    for page_el in root.findall("page"):
        page_idx = int(page_el.get("idx", 0))
        for fig in page_el.findall("figure"):
            img_el = fig.find("img")
            if img_el is None:
                continue

            src = img_el.get("src", "")
            img_path = img_root / src

            caption_el = fig.find("caption")
            figures.append({
                "page":       page_idx + 1,
                "bbox":       fig.get("bbox", ""),
                "category":   fig.get("category", "—"),
                "confidence": fig.get("category_confidence", ""),
                "caption":    caption_el.text.strip() if caption_el is not None and caption_el.text else "",
                "img_path":   img_path,
            })

    return figures


def _encode_image(img_path: Path) -> str | None:
    """Base64-encode an image file for embedding in HTML.

    Args:
        img_path: Path to the image file.

    Returns:
        data URI string, or None if the file cannot be read.
    """
    if not img_path.exists():
        return None
    ext = img_path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif"}.get(ext, "jpeg")
    data = base64.b64encode(img_path.read_bytes()).decode()
    return f"data:image/{mime};base64,{data}"


# ── HTML generation ────────────────────────────────────────────────────────────

def _confidence_bar(confidence: str) -> str:
    """Render a small inline confidence bar."""
    try:
        pct = round(float(confidence) * 100)
    except ValueError:
        return ""
    color = "#22c55e" if pct >= 70 else "#f97316" if pct >= 50 else "#ef4444"
    return f"""
      <div class="conf-bar-wrap" title="{pct}% confidence">
        <div class="conf-bar" style="width:{pct}%;background:{color}"></div>
        <span class="conf-text">{pct}%</span>
      </div>"""


def _figure_card(fig: dict) -> str:
    """Render a single figure card as HTML."""
    data_uri = _encode_image(fig["img_path"])
    img_tag = (
        f'<img src="{data_uri}" alt="{fig["category"]}">'
        if data_uri
        else '<div class="no-img">image not found</div>'
    )

    label = fig["category"]
    color = _LABEL_COLORS.get(label, "#374151")
    caption = f'<p class="caption">"{fig["caption"]}"</p>' if fig["caption"] else ""
    conf_bar = _confidence_bar(fig["confidence"])

    return f"""
    <div class="card" data-label="{label}">
      <div class="img-wrap">{img_tag}</div>
      <div class="meta">
        <span class="badge" style="background:{color}">{label}</span>
        {conf_bar}
        <p class="page">Page {fig["page"]}</p>
        {caption}
      </div>
    </div>"""


def _build_html(doc_name: str, figures: list[dict]) -> str:
    """Build the full HTML page."""
    all_labels = sorted({f["category"] for f in figures})
    filter_buttons = "".join(
        f'<button class="filter-btn" data-label="{lbl}" '
        f'style="border-color:{_LABEL_COLORS.get(lbl,"#888")}">{lbl} '
        f'<span class="count">{sum(1 for f in figures if f["category"]==lbl)}</span></button>'
        for lbl in all_labels
    )
    cards = "".join(_figure_card(f) for f in figures)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Figure Labels — {doc_name}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}

  header {{ padding: 2rem; border-bottom: 1px solid #1e293b; }}
  header h1 {{ font-size: 1.4rem; font-weight: 600; color: #f1f5f9; }}
  header p  {{ color: #64748b; margin-top: .3rem; font-size: .875rem; }}

  .filters {{ display: flex; flex-wrap: wrap; gap: .5rem; padding: 1.25rem 2rem;
               border-bottom: 1px solid #1e293b; }}
  .filter-btn {{ background: transparent; color: #cbd5e1; border: 1.5px solid #475569;
                  border-radius: 999px; padding: .3rem .85rem; font-size: .8rem;
                  cursor: pointer; transition: all .15s; }}
  .filter-btn:hover, .filter-btn.active {{ background: #1e293b; color: #f1f5f9; }}
  .filter-btn .count {{ font-weight: 700; opacity: .7; }}

  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 1.25rem; padding: 1.75rem 2rem; }}

  .card {{ background: #1e293b; border-radius: .75rem; overflow: hidden;
            border: 1px solid #334155; transition: transform .15s, box-shadow .15s; }}
  .card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.4); }}
  .card.hidden {{ display: none; }}

  .img-wrap {{ background: #0f172a; display: flex; align-items: center;
                justify-content: center; min-height: 180px; overflow: hidden; }}
  .img-wrap img {{ max-width: 100%; max-height: 260px; object-fit: contain; display: block; }}
  .no-img {{ color: #475569; font-size: .8rem; padding: 2rem; text-align: center; }}

  .meta {{ padding: .9rem 1rem 1rem; }}
  .badge {{ display: inline-block; border-radius: 999px; padding: .2rem .7rem;
             font-size: .72rem; font-weight: 600; color: #fff; letter-spacing: .02em; }}

  .conf-bar-wrap {{ display: flex; align-items: center; gap: .5rem; margin: .55rem 0 .3rem; }}
  .conf-bar {{ height: 5px; border-radius: 3px; transition: width .3s; }}
  .conf-text {{ font-size: .72rem; color: #94a3b8; white-space: nowrap; }}

  .page {{ font-size: .72rem; color: #64748b; margin-top: .3rem; }}
  .caption {{ font-size: .73rem; color: #94a3b8; margin-top: .4rem; font-style: italic;
               line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3;
               -webkit-box-orient: vertical; overflow: hidden; }}
</style>
</head>
<body>
<header>
  <h1>Figure Classification — {doc_name}</h1>
  <p>{len(figures)} figures &nbsp;·&nbsp; click a label to filter</p>
</header>
<div class="filters">
  <button class="filter-btn active" data-label="all">all <span class="count">{len(figures)}</span></button>
  {filter_buttons}
</div>
<div class="grid">{cards}</div>
<script>
  const btns  = document.querySelectorAll('.filter-btn');
  const cards = document.querySelectorAll('.card');
  btns.forEach(btn => btn.addEventListener('click', () => {{
    btns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const lbl = btn.dataset.label;
    cards.forEach(c => c.classList.toggle('hidden', lbl !== 'all' && c.dataset.label !== lbl));
  }}));
</script>
</body>
</html>"""


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an HTML viewer for figure classification results."
    )
    parser.add_argument("--xml", type=Path, required=True,
                        help="Path to the classified XML file.")
    parser.add_argument("--img-root", type=Path, default=None,
                        help="Directory containing the images/ folder. Auto-detected if omitted.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path. Defaults to drafts/<doc_name>_labels.html.")
    parser.add_argument("--no-open", action="store_true",
                        help="Write HTML but do not open it in the browser.")
    args = parser.parse_args()

    xml_path = args.xml.resolve()
    img_root = args.img_root.resolve() if args.img_root else _find_img_root(xml_path)
    doc_name = xml_path.stem

    print(f"XML:      {xml_path}")
    print(f"img-root: {img_root}")

    figures = _load_figures(xml_path, img_root)
    print(f"Figures:  {len(figures)} found")

    html = _build_html(doc_name, figures)

    out_path = args.out or Path(__file__).parent / f"{doc_name}_labels.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved:    {out_path}")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
