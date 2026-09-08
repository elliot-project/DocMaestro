"""Path derivation for the extraction pipeline.

Single source of truth for mapping a source PDF to its place in the output
tree, and for finding the extracted images that belong to a document.
Centralising this avoids the copy-paste drift that previously let one call
site flatten the hierarchy while another preserved it.
"""

from __future__ import annotations

from pathlib import Path


def doc_rel_dir(pdf_path: Path, root: Path) -> Path:
    """Return the per-document output sub-path for ``pdf_path``.

    The returned path is relative and always ends with the document name
    (the PDF stem), so callers can join it under any output root:
    ``output_dir / doc_rel_dir(pdf, root) / f"{stem}_{backend}.xml"``.

    When ``root`` is a directory, the PDF's location *within* ``root`` is
    preserved (e.g. ``icml/2024/paper.pdf`` under ``pdfs/`` →
    ``icml/2024/paper``). Both ``pdf_path`` and ``root`` are resolved to
    absolute paths before comparison, so a relative input directory no longer
    collapses the hierarchy. When ``root`` is a single file, or ``pdf_path``
    lies outside ``root``, the document is placed flat at ``Path(stem)``.

    Args:
        pdf_path: Source PDF path (relative or absolute).
        root: The input root passed on the command line — a directory of PDFs
            or a single PDF file.

    Returns:
        A relative ``Path`` ending in the document stem.
    """
    stem = pdf_path.stem
    if root.is_file():
        return Path(stem)
    try:
        rel_parent = pdf_path.resolve().relative_to(root.resolve()).parent
    except ValueError:
        return Path(stem)
    return rel_parent / stem


def output_xml_path(output_dir: Path, rel_dir: Path, doc_name: str, backend: str) -> Path:
    """Return the synthesized-XML path for a document.

    Args:
        output_dir: Configured output root.
        rel_dir: Per-document sub-path from :func:`doc_rel_dir` (ends in stem).
        doc_name: Document stem.
        backend: Extraction backend name, embedded in the filename.

    Returns:
        ``output_dir / rel_dir / f"{doc_name}_{backend}.xml"``.
    """
    return output_dir / rel_dir / f"{doc_name}_{backend}.xml"


def find_img_root(data_dir: Path, rel_folder: str) -> Path | None:
    """Locate the directory a document's ``<img src>`` values resolve against.

    Synthesized XML stores image references as ``images/<hash>.jpg``, but the
    files themselves live under ``data/``, not next to the XML in ``output/``.
    The two trees do not mirror each other: MinerU nests its own run folders
    below the document (``data/ASML/hybrid-auto-engine/ASML/hybrid_auto/images``),
    and re-runs leave several ``images/`` directories behind. So the directory
    is searched for rather than computed, and the most populated candidate wins
    — a partial benchmark run leaves a thin ``images/`` that should lose to the
    full extraction.

    The document folder is tried first; only if that misses does the search
    widen to the whole of ``data_dir`` matching on the folder's leaf name,
    which covers documents whose output path does not match their data path.

    Args:
        data_dir: Root of the raw-output tree (typically ``data/``).
        rel_folder: The document's folder path relative to the output root,
            e.g. ``"ASML"`` or ``"conf/2024/paper"``.

    Returns:
        The parent of the chosen ``images/`` directory, so that
        ``root / "images" / name`` resolves — or ``None`` when the document has
        no extracted images.
    """
    if not data_dir.exists():
        return None

    direct = _most_populated_images_dir(data_dir / rel_folder)
    if direct is not None:
        return direct.parent

    leaf = rel_folder.rstrip("/").split("/")[-1]
    widened = _most_populated_images_dir(data_dir, name_filter=leaf)
    return widened.parent if widened is not None else None


def _most_populated_images_dir(base: Path, name_filter: str | None = None) -> Path | None:
    """Return the ``images/`` directory under *base* holding the most files.

    Args:
        base: Directory to search recursively. A missing path yields ``None``.
        name_filter: When given, only consider directories whose path contains
            this substring.

    Returns:
        The winning ``images/`` directory, or ``None`` if there is no candidate.
    """
    if not base.exists():
        return None

    best: Path | None = None
    best_count = -1
    for candidate in base.rglob("images"):
        if not candidate.is_dir():
            continue
        if name_filter is not None and name_filter not in str(candidate):
            continue
        count = sum(1 for _ in candidate.iterdir())
        if count > best_count:
            best_count = count
            best = candidate
    return best
