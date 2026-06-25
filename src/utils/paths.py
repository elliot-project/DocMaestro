"""Output path derivation for the extraction pipeline.

Single source of truth for mapping a source PDF to its place in the output
tree. Centralising this avoids the copy-paste drift that previously let one
call site flatten the hierarchy while another preserved it.
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
