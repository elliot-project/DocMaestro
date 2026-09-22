# Task Composer

The Task Composer is a WebDataset-compatible SSL (Self-Supervised Learning) task generation module. It consumes the parsed XML output of the MinerU pipeline and produces labeled training samples — packaged into `.tar` shards — ready for downstream model training without requiring any manual annotation.

---

## Overview

The module answers one question: **given a structured document XML, how do we derive rich training signal for free?**

Each composer looks at the document's inherent structure — captions, reading order, figure-page relationships — and turns that structure into a supervised learning task. The "labels" are already embedded in the document; the composer just extracts them.

```
Parsed XML
    │
    ├── CaptionPredictionComposer  →  (image + context) → caption text
    ├── FigureSelectionComposer    →  (page text)        → which figure belongs here?
    └── ReadingOrderComposer       →  (shuffled elements) → restore document order
                                              │
                                         list[Sample]
                                              │
                                         ShardWriter
                                              │
                                     caption_prediction-00000.tar
                                     figure_selection-00000.tar
                                     reading_order-00000.tar
```

---

## Architecture

### Core Abstractions (`src/doc_maestro/task_composer/base.py`)

#### `Sample`

A single WebDataset-compatible record. Every sample has a `key` and a `payload` dict whose keys are file extensions and whose values are the file contents.

```python
@dataclass
class Sample:
    key: str               # e.g. "adidas_p002_caption_pred_0"
    payload: dict[str, Any]
```

**Key naming convention:**

```
{doc_name}_p{page_idx:03d}_{task_abbreviation}_{sample_idx}
```

| Task | Abbreviation | Example key |
|------|-------------|-------------|
| Caption prediction | `caption_pred` | `adidas_p002_caption_pred_0` |
| Figure selection | `fig_sel` | `adidas_p002_fig_sel_1` |
| Reading order | `read_ord` | `adidas_p002_read_ord_0` |

**Payload extensions** are the tar entry suffixes. Every sample always has `__key__` (metadata only — never written to disk). Task-specific extensions vary per task (see individual task sections below).

`Sample.__post_init__` enforces two invariants at construction time:
- `payload["__key__"]` must be present
- `payload["__key__"]` must equal `sample.key`

Both checks raise `ValueError` immediately, so malformed samples are caught before writing.

---

#### `DocumentContext`

A read-only snapshot of the full document, passed to every `compose_page` call. This allows composers to perform cross-page lookups (e.g., finding distractors from other pages).

```python
@dataclass
class DocumentContext:
    doc_name: str           # matches XML root `name` attribute
    pages: list[ET.Element] # all <page> elements in document order
    img_root: Path          # base directory for resolving <img src> paths
```

**Do not mutate `pages` after construction.** Composers that need cross-page data should iterate `context.pages` in `compose_page`.

---

#### `BaseTaskComposer`

Abstract base class for all task composers.

```python
class BaseTaskComposer(ABC):
    task_name: ClassVar[str]  # must be declared by every subclass

    @abstractmethod
    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]: ...

    def compose_document(self, root: ET.Element, img_root: Path) -> list[Sample]: ...
```

**Contract:**

- Every subclass **must** declare a non-empty `task_name` class variable. Instantiation raises `TypeError` if missing.
- `compose_page` receives a single `<page>` element and the full document context. It returns zero or more samples.
- `compose_document` is the entry point for processing a full document. The default implementation builds a `DocumentContext` from the root element and calls `compose_page` for every page in order, collecting all results into a flat list.

Override `compose_document` only if your task requires coordination across pages **during** composition (building a shared index before sampling, etc.).

---

### Shard Writer (`src/doc_maestro/task_composer/shard_writer.py`)

Writes samples into WebDataset-compatible `.tar` archives. Fully decoupled from task logic — accepts any `list[Sample]` regardless of origin.

```python
class ShardWriter:
    def __init__(
        self,
        out_dir: Path,
        task_name: str,
        shard_size: int = 1000,
        compress: bool = False,
    ) -> None: ...

    def write(self, samples: list[Sample]) -> list[Path]: ...
```

**Shard naming:**

```
{task_name}-{shard_idx:05d}.tar        # uncompressed
{task_name}-{shard_idx:05d}.tar.gz     # compressed
```

Example with `shard_size=1000` and 2500 samples:
```
caption_prediction-00000.tar   (samples 0–999)
caption_prediction-00001.tar   (samples 1000–1999)
caption_prediction-00002.tar   (samples 2000–2499)
```

**Encoding rules** (how payload values become tar entry bytes):

| Python type | Encoding |
|-------------|----------|
| `bytes` | verbatim |
| `dict` / `list` | UTF-8 JSON |
| anything else | `str(value).encode()` |

The `__key__` entry is **never** written to disk — it's metadata only.

---

## Task Reference

### `CaptionPredictionComposer`

**Task:** Given a figure/chart/table image and its surrounding page context, predict the caption.

**Source:** `src/doc_maestro/task_composer/tasks/caption_prediction.py`

**Eligible elements:** `<figure>`, `<chart>`, `<table>` — only those with a non-empty `<caption>` child.

**Excluded from context window:** `page-header`, `page-footer`, `page-number`, `abandon`

**Sample payload:**

| Extension | Type | Content |
|-----------|------|---------|
| `__key__` | str | sample key (not written to disk) |
| `json` | dict | task metadata (task_type, doc_name, page_idx, element_tag, bbox) |
| `input.json` | dict | `{"preceding": [...], "following": [...]}` — sibling context |
| `target.txt` | str | raw caption text |
| `image.jpg` | bytes | image bytes (omitted if file missing; warning emitted) |

**Context window logic:**

The composer builds a filtered sibling list by removing `_SKIP_CONTEXT_TAGS` from the page's child elements, then for each target element it takes:
- Up to 3 elements **before** the target, listed closest-first (i.e., the immediate predecessor is index 0)
- Up to 3 elements **after** the target, also closest-first

Each sibling is serialized as:
```json
{"tag": "text-block", "text": "first 120 chars...", "bbox": "[x1, y1, x2, y2]"}
```

**Image resolution:**
```
img_path = context.img_root / elem.find("img").attrib["src"]
```
If the path doesn't exist, `image.jpg` is omitted from the payload and `warnings.warn` is called. The sample is still written.

If the `<img>` element has an empty `src` attribute, the image is silently skipped (no warning, no `image.jpg`).

**One sample per eligible captioned element per page.** A page with 2 figures + 1 table (all with captions) produces 3 samples.

---

### `FigureSelectionComposer`

**Task:** Given the text of a page, identify which figure belongs to it from a set of candidates.

**Source:** `src/doc_maestro/task_composer/tasks/figure_selection.py`

**Constructor args:**
- `n_distractors: int = 2` — number of distractor figures per sample
- `seed: int | None = None` — RNG seed for reproducibility

**Eligible figures:** `<figure>` elements with an `<img>` child (on any page in the document).

**Sample payload:**

| Extension | Type | Content |
|-----------|------|---------|
| `__key__` | str | sample key (not written to disk) |
| `input.txt` | str | concatenated `text-block` and `title` text from the page |
| `target.json` | dict | `{"label": int}` — index of the correct candidate in the shuffled list |
| `json` | dict | task metadata including the `candidates` list |

**Candidate structure (inside `json.candidates`):**
```json
[
  {"src": "images/p0.jpg", "page_origin": 0},
  {"src": "images/p3.jpg", "page_origin": 3},
  {"src": "images/p7.jpg", "page_origin": 7}
]
```

**Algorithm:**
1. Collect all figures on the current page (the candidates pool for the correct answer).
2. Collect all figures from **other** pages (the distractor pool).
3. If `len(other_figures) < n_distractors`, skip this page entirely (no samples).
4. For each figure on the current page, draw `n_distractors` distractors via `rng.sample`, build the candidate list `[correct] + distractors`, shuffle it, and record which index holds the correct figure as `label`.

**One sample per eligible figure on the current page.** A page with 2 figures produces 2 samples (each with a different correct candidate).

---

### `ReadingOrderComposer`

**Task:** Given a shuffled subset of page elements, recover the original document reading order.

**Source:** `src/doc_maestro/task_composer/tasks/reading_order.py`

**Constructor args:**
- `n_elements: int = 5` — elements to sample per page
- `seed: int | None = None` — RNG seed for reproducibility

**Excluded from eligibility:** `page-header`, `page-footer`, `page-number`, `abandon`

**Sample payload:**

| Extension | Type | Content |
|-----------|------|---------|
| `__key__` | str | sample key (not written to disk) |
| `input.json` | list | shuffled element descriptions |
| `target.json` | list | permutation indices that restore correct order |
| `json` | dict | task metadata (task_type, doc_name, page_idx, n_elements) |

**Input item format:**
```json
{"idx": 2, "tag": "text-block", "text": "first 120 chars..."}
```
`idx` reflects the item's position in the **shuffled** list (0-indexed).

**Target semantics:**

`target` is a list of indices into the shuffled input. Applying them restores the original document order:

```python
restored = [shuffled[i] for i in target]
# restored is now in the original document reading order
```

**Example** (5 elements, shuffled_perm = [2, 0, 4, 1, 3]):

```
correct order:  [A, B, C, D, E]   (local indices 0–4)
shuffled order: [C, A, E, B, D]   (shuffled_perm applied)
target:         [1, 3, 0, 4, 2]   (where each correct element landed in the shuffle)

restored = [shuffled[1], shuffled[3], shuffled[0], shuffled[4], shuffled[2]]
         = [A,           B,           C,           D,           E]  ✓
```

**One sample per page** (if the page has at least `n_elements` eligible elements).

---

## Usage

### Basic: single document

```python
import xml.etree.ElementTree as ET
from pathlib import Path
from doc_maestro.task_composer import CaptionPredictionComposer, ShardWriter

root = ET.parse("output/adidas/adidas.xml").getroot()
img_root = Path("output/adidas/images")

composer = CaptionPredictionComposer()
samples = composer.compose_document(root, img_root=img_root)

writer = ShardWriter(out_dir=Path("output/shards"), task_name="caption_prediction")
shard_paths = writer.write(samples)

print(f"Wrote {len(samples)} samples to {len(shard_paths)} shards")
```

### All three tasks

```python
from doc_maestro.task_composer import (
    CaptionPredictionComposer,
    FigureSelectionComposer,
    ReadingOrderComposer,
    ShardWriter,
)

composers = [
    CaptionPredictionComposer(),
    FigureSelectionComposer(n_distractors=2, seed=42),
    ReadingOrderComposer(n_elements=5, seed=42),
]

for composer in composers:
    samples = composer.compose_document(root, img_root=img_root)
    writer = ShardWriter(
        out_dir=Path("output/shards"),
        task_name=composer.task_name,
        shard_size=1000,
    )
    writer.write(samples)
```

### Custom task

Subclass `BaseTaskComposer`, declare `task_name`, implement `compose_page`:

```python
from doc_maestro.task_composer.base import BaseTaskComposer, DocumentContext, Sample
import xml.etree.ElementTree as ET

class TitleExtractionComposer(BaseTaskComposer):
    task_name = "title_extraction"

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        samples = []
        for elem in page:
            if elem.tag != "title":
                continue
            key = f"{context.doc_name}_p{page.attrib['idx']}_title_{len(samples)}"
            samples.append(Sample(
                key=key,
                payload={
                    "__key__": key,
                    "target.txt": (elem.text or "").strip(),
                    "json": {"task_type": self.task_name, "level": elem.attrib.get("level", "")},
                }
            ))
        return samples
```

Override `compose_document` only if you need a cross-page index built before page iteration. The default is correct for the vast majority of tasks.

---

## File Layout

```
src/doc_maestro/task_composer/
├── __init__.py                      # public exports
├── base.py                          # Sample, DocumentContext, BaseTaskComposer
├── shard_writer.py                  # ShardWriter
└── tasks/
    ├── __init__.py
    ├── caption_prediction.py        # CaptionPredictionComposer
    ├── figure_selection.py          # FigureSelectionComposer
    └── reading_order.py             # ReadingOrderComposer

tests/task_composer/
├── test_base.py
├── test_shard_writer.py
└── tasks/
    ├── test_caption_prediction.py
    ├── test_figure_selection.py
    └── test_reading_order.py
```

---

## Running the Tests

```bash
uv run pytest tests/task_composer/ -v
```

---

## Design Notes

**Why WebDataset?** `.tar` shards can be streamed directly by PyTorch DataLoader via the `webdataset` library without extracting to disk. Each tar entry (`{key}.{ext}`) maps to a field in the sample dict, making deserialization trivial.

**Why one sample per eligible element (not one per page)?** Caption prediction and figure selection benefit from granularity — a densely illustrated page yields multiple training signals rather than being collapsed to one. Reading order is the exception: it produces one sample per page because the task is inherently page-scoped.

**Why closest-first context in `CaptionPredictionComposer`?** Immediate neighbors carry more signal than distant ones. A model that processes them in order (index 0 = closest) can learn to weight proximity without positional encoding tricks.

**Why `random.Random` (not `numpy.random`)?** The RNG is isolated per-composer instance, which means two composers with the same seed produce the same sequence independently of each other and of any global state. This makes test fixtures and multi-task pipelines deterministic without global seed management.
