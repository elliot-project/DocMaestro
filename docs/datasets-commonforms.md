# CommonForms Dataset

## Overview

[CommonForms](https://huggingface.co/datasets/jbarrow/CommonForms) is processed as
a **second dataset** alongside the conference-papers dataset. It is a form-field
detection dataset: roughly 55,000 documents / 450,000+ pages, with each HuggingFace
row holding **one page image** plus ground-truth form-field annotations.

DocMaestro ingests it without touching the extraction core. A preparation step
converts the per-page rows into **per-document multi-page PDFs**, and the existing
multi-GPU pipeline (`scripts/run_sharded.py`) then processes those PDFs exactly
like any other PDF source. The ground-truth annotations are written to an additive
sidecar that can later be merged into the synthesized XML.

```
HF rows (1 page each)  ──prepare──▶  per-doc PDFs + GT sidecars  ──run_sharded──▶  XML + raw JSON
```

---

## Dataset shape

| Column | Type | Notes |
|--------|------|-------|
| `id` | int64 | Row identifier |
| `image_id` | int64 | Image reference |
| `image` | image | The page JPEG |
| `file_name` | string | e.g. `0005676-1.jpeg` |
| `width`, `height` | int64 | Page pixel dimensions (authoritative) |
| `objects` | dict | COCO-style GT: parallel lists `bbox`, `category_id`, `area`, `id`, ... |

**Splits:** `train`, `test`, **`valid`** (note the name — it is `valid`, not
`validation`). The current setup targets the `valid` split (~18,200 pages).

**Page-to-document mapping:** pages of one document share a root file name and
differ only by a trailing `-<page>`:

```
0005676-1.jpeg, 0005676-2.jpeg, 0005676-3.jpeg   →  document 0005676, pages 1–3
```

The document root is everything **before the last hyphen**, so roots that contain
hyphens are preserved (`00056-76-2.jpeg` → root `00056-76`, page `2`).

**Ground-truth classes** (the dataset's authoritative `ClassLabel`, normalized to
lowercase):

| `category_id` | Name |
|---------------|------|
| 0 | `text` |
| 1 | `checkbox` |
| 2 | `signature` |

---

## On-disk layout

Everything for the dataset lives under one root (default
`/data/151-1/datasets/CommonForms/`):

```
/data/151-1/datasets/CommonForms/
  raw/                       # HuggingFace download cache (re-runnable, offline after first pull)
  pdfs/<doc>.pdf             # materialized multi-page PDFs — the pipeline input
  fields/<doc>.fields.json   # ground-truth form-field sidecars — additive layer
```

Extraction output goes to the location set in the config
(`/data/151-1/users/mserrao/commonforms/`), separate from the resume manifest
(`/data/151-1/users/mserrao/commonforms-state/`).

---

## Step 1 — Prepare (materialize PDFs + sidecars)

```bash
# Full valid split
uv run scripts/prepare_commonforms.py

# Smoke test: first 20 documents only
uv run scripts/prepare_commonforms.py --split valid --limit 20
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--root` | `/data/151-1/datasets/CommonForms` | Output root (`raw/`, `pdfs/`, `fields/`) |
| `--split` | `valid` | Dataset split to materialize |
| `--limit` | none | Process only the first N documents (smoke testing) |

What it does:

1. Loads the split **non-streaming** (memory-mapped Arrow — not held in RAM) with
   `Image(decode=False)` so the original JPEG bytes are read losslessly.
2. Groups page rows by document root, sorted ascending by page number.
3. Builds one PDF per document with PyMuPDF using an explicit page rectangle equal
   to the image's pixel size, embedding the JPEG bytes directly. This makes
   **1 pixel = 1 PDF point**, so GT bounding boxes align with the page the model
   parses.
4. Writes a GT sidecar per document (see below).

The first run downloads the split into `raw/` (multi-GB); later runs reuse the
cache. Writes are **atomic** (temp file → rename) and **idempotent** — a document
whose `.pdf` and `.fields.json` both exist is skipped, so the prep is resumable
after an interruption.

---

## Ground-truth sidecar format

`fields/<doc>.fields.json`, one per document:

```json
{
  "doc_id": "0005676",
  "source": "commonforms",
  "bbox_format": "xyxy_pixels",
  "categories": {"0": "text", "1": "checkbox", "2": "signature"},
  "pages": [
    {
      "page": 1,
      "width": 1700,
      "height": 2200,
      "fields": [
        {"category_id": 0, "category": "text",
         "bbox": [x1, y1, x2, y2], "area": 1234.0, "id": 99}
      ]
    }
  ]
}
```

COCO `[x, y, w, h]` boxes are converted to `[x1, y1, x2, y2]` (matching the
project's bbox convention) in **pixel** coordinates. Because the materialized PDF
is built at 1 pixel = 1 point, these coordinates share the PDF's space and can be
merged into the synthesized XML later (e.g. as `<form-field>` elements). That merge
is intentionally out of scope here — the sidecar is an additive layer kept beside
the output.

---

## Step 2 — Extract

```bash
uv run scripts/run_sharded.py \
    --gpus 6,7 \
    --pdf /data/151-1/datasets/CommonForms/pdfs/ \
    --config configs/extractor/commonforms.yaml
```

This reuses the standard multi-GPU dispatcher, append-only JSONL manifest
(resume/retry), and XML synthesis. Document IDs are flat, so output lands at
`<output_dir>/<doc>/<doc>_hybrid-auto-engine.xml` plus the raw `middle.json` /
`content_list.json`. Interrupt with Ctrl-C and re-run to skip completed documents;
add `--retry-failed` to reprocess failures.

The config sets `gpu_memory_utilization: 0.70` — on 46 GiB cards the hybrid
backend's torch-side work runs outside vLLM's reserved budget, and 0.85 leaves too
little headroom (see [extraction.md](extraction.md) and the GPU-budget notes).

---

## Code map

| Path | Role |
|------|------|
| `src/datasets/commonforms.py` | Pure logic: `parse_file_name`, `group_pages`, `coco_to_xyxy`, `build_fields_sidecar`, `build_pdf`, `CATEGORY_NAMES` |
| `scripts/prepare_commonforms.py` | CLI: download + materialize PDFs and sidecars |
| `configs/extractor/commonforms.yaml` | Extraction config (backend, paths, GPU budget) |
| `tests/test_commonforms.py` | Unit tests for the pure logic |

No new dependencies — `datasets` and `pymupdf` are already part of the project.
