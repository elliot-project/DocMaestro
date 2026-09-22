"""Shared ground-truth-generation helpers over parsed document XML.

Pure functions operating on ``xml.etree.ElementTree`` page elements. Reused by
the SSL task composers so that text extraction, layout enumeration, markdown
conversion, bbox stripping, and structural mutation live in exactly one place.
"""

from __future__ import annotations

import copy
import random
import xml.etree.ElementTree as ET

# Structural noise excluded from reading/layout/markdown ground truth.
NOISE_TAGS: set[str] = {
    "page-header",
    "page-footer",
    "page-number",
    "abandon",
    "page-footnote",
    "seal",
}

# Tags whose text content counts as readable body content.
READING_TEXT_TAGS: set[str] = {
    "text-block",
    "title",
    "list",
    "math-block",
    "code-block",
    "table-body",
    "caption",
    "footnote",
    "aside",
    "index",
}


def page_idx(page: ET.Element) -> int:
    """Return the integer ``idx`` attribute of a ``<page>`` element."""
    return int(page.attrib["idx"])


def page_reading_text(page: ET.Element) -> str:
    """Concatenate all text-bearing content on the page in reading order.

    Walks the page depth-first (document order), collecting stripped text from
    every element whose tag is in ``READING_TEXT_TAGS`` and not in
    ``NOISE_TAGS``. Nested text (figure captions, table bodies) is included.

    Args:
        page: The ``<page>`` element.

    Returns:
        Newline-joined text, or ``""`` if the page carries no readable text.
    """
    parts: list[str] = []
    for el in page.iter():
        if el.tag in READING_TEXT_TAGS and el.tag not in NOISE_TAGS:
            text = (el.text or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def text_blocks_with_bbox(page: ET.Element) -> list[dict]:
    """Return top-level text blocks paired with their bounding boxes.

    Args:
        page: The ``<page>`` element.

    Returns:
        List of ``{"bbox": str, "tag": str, "text": str}`` for each direct child
        that has both a ``bbox`` attribute and non-empty direct text, excluding
        noise tags.
    """
    out: list[dict] = []
    for el in page:
        if el.tag in NOISE_TAGS:
            continue
        bbox = el.attrib.get("bbox")
        text = (el.text or "").strip()
        if bbox and text:
            out.append({"bbox": bbox, "tag": el.tag, "text": text})
    return out


def layout_elements(page: ET.Element) -> list[dict]:
    """Return every direct child that has a bbox, as ``{tag, bbox}``.

    Noise tags are excluded. Figures, tables, and charts (which carry a bbox but
    no direct text) are included — this is layout, not reading.

    Args:
        page: The ``<page>`` element.

    Returns:
        List of ``{"tag": str, "bbox": str}``.
    """
    out: list[dict] = []
    for el in page:
        if el.tag in NOISE_TAGS:
            continue
        bbox = el.attrib.get("bbox")
        if bbox:
            out.append({"tag": el.tag, "bbox": bbox})
    return out


def xml_to_markdown(page: ET.Element) -> str:
    """Render a page as semantic markdown.

    Titles map to ``#`` headings by ``level``; figures to ``![caption](src)``;
    tables/charts to their raw ``<table-body>`` HTML (or an image fallback);
    math to ``$$`` blocks; code to fenced blocks; everything else to raw text.

    Args:
        page: The ``<page>`` element.

    Returns:
        Markdown string (blocks separated by blank lines).
    """
    blocks: list[str] = []
    for el in page:
        if el.tag in NOISE_TAGS:
            continue
        if el.tag == "title":
            level = max(1, min(6, int(el.attrib.get("level", "1"))))
            blocks.append("#" * level + " " + (el.text or "").strip())
        elif el.tag == "figure":
            img = el.find("img")
            cap = el.find("caption")
            src = img.attrib.get("src", "") if img is not None else ""
            caption = (cap.text or "").strip() if cap is not None else ""
            blocks.append(f"![{caption}]({src})")
        elif el.tag in ("table", "chart"):
            body = el.find("table-body")
            if body is not None and (body.text or "").strip():
                blocks.append((body.text or "").strip())
            else:
                img = el.find("img")
                if img is not None:
                    blocks.append(f"![]({img.attrib.get('src', '')})")
        elif el.tag == "math-block":
            blocks.append(f"$$\n{(el.text or '').strip()}\n$$")
        elif el.tag == "code-block":
            blocks.append(f"```\n{(el.text or '').strip()}\n```")
        else:
            text = (el.text or "").strip()
            if text:
                blocks.append(text)
    return "\n\n".join(b for b in blocks if b)


def strip_bbox(page: ET.Element) -> str:
    """Return the page XML with every ``bbox`` attribute removed.

    Deep-copies the element so the source tree is untouched. Tags, text, and
    other attributes (``level``, ``src``) are preserved.

    Args:
        page: The ``<page>`` element.

    Returns:
        Serialized bbox-free XML string.
    """
    clone = copy.deepcopy(page)
    for el in clone.iter():
        el.attrib.pop("bbox", None)
    return ET.tostring(clone, encoding="unicode")


def mutate_tree(xml_str: str, rng: random.Random) -> str:
    """Inject one structural mutation into a serialized page tree.

    Picks one of ``delete`` (drop a child), ``retag`` (change a child's tag),
    or ``shuffle`` (reorder children) using ``rng`` for reproducibility.

    Args:
        xml_str: Serialized (typically bbox-free) page XML.
        rng: Seeded RNG controlling the mutation choice and target.

    Returns:
        Serialized mutated XML string. Returns the input unchanged if the page
        has no children.
    """
    root = ET.fromstring(xml_str)
    children = list(root)
    if not children:
        return xml_str

    mutation = rng.choice(["delete", "retag", "shuffle"])
    if mutation == "delete" and len(children) > 1:
        root.remove(rng.choice(children))
    elif mutation == "retag":
        victim = rng.choice(children)
        candidates = [t for t in ["text-block", "title", "figure", "table"] if t != victim.tag]
        victim.tag = rng.choice(candidates) if candidates else "title"
    else:  # shuffle
        rng.shuffle(children)
        for child in list(root):
            root.remove(child)
        for child in children:
            root.append(child)
    return ET.tostring(root, encoding="unicode")
