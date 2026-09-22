"""Per-task input/output modality contract for VLM training samples.

A task composer decides *what* a sample contains; this module decides what the
model is allowed to *see* when answering it. The two are separate concerns
because several tasks are only learnable when an asset is deliberately withheld
— showing the page image to ``doc_reconstruction`` turns layout inference into
reading coordinates off a picture.

The contract is consumed in three places:

- The packer, to know which asset keys a sample must carry.
- The dataloader, to know how many images to resolve and in what role.
- The augmentation policy, via :func:`geometric_augmentation_allowed`, because
  a task whose target is a bounding box cannot survive an un-mirrored geometric
  transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssetPolicy(Enum):
    """How many images of a given kind a task receives.

    Attributes:
        NONE: The task receives none, often deliberately — see the leak-avoidance
            notes on the individual contracts below.
        SINGLE: Exactly one image.
        MULTI: A variable-length set, sized by the composer's own config.
    """

    NONE = "none"
    SINGLE = "single"
    MULTI = "multi"


class TargetFormat(Enum):
    """Serialization the model is expected to emit for a task."""

    TEXT = "text"
    MARKDOWN = "markdown"
    XML = "xml"
    JSON = "json"


@dataclass(frozen=True)
class TaskContract:
    """What one task feeds the model and what it expects back.

    Args:
        task_name: Stable identifier, matching the composer's ``task_name``.
        page_images: How many full-page renders the task receives.
        figure_images: How many figure/table crops the task receives.
        text_input: Whether the prompt carries a structured text payload
            (a shuffled element list, a corrupted tree) beyond the instruction.
        target_format: Serialization the model must produce.
        bbox_bearing: Whether the target contains ``[0, 1000]`` normalized
            coordinates. Drives the geometric-augmentation ban.
        redact_page_numbers: Whether page-number regions must be blacked out of
            the rendered pages before the model sees them.
        notes: Why this contract departs from the obvious choice, where it does.
    """

    task_name: str
    page_images: AssetPolicy
    figure_images: AssetPolicy
    text_input: bool
    target_format: TargetFormat
    bbox_bearing: bool = False
    redact_page_numbers: bool = False
    notes: str = ""


_CONTRACTS: tuple[TaskContract, ...] = (
    # ── Page-level tasks: one render in, one serialization out ────────────────
    TaskContract(
        "basic_reading",
        AssetPolicy.SINGLE,
        AssetPolicy.NONE,
        text_input=False,
        target_format=TargetFormat.TEXT,
    ),
    TaskContract(
        "ocr",
        AssetPolicy.SINGLE,
        AssetPolicy.NONE,
        text_input=False,
        target_format=TargetFormat.JSON,
        bbox_bearing=True,
    ),
    TaskContract(
        "layout",
        AssetPolicy.SINGLE,
        AssetPolicy.NONE,
        text_input=False,
        target_format=TargetFormat.JSON,
        bbox_bearing=True,
    ),
    TaskContract(
        "markdown",
        AssetPolicy.SINGLE,
        AssetPolicy.NONE,
        text_input=False,
        target_format=TargetFormat.MARKDOWN,
    ),
    TaskContract(
        "page_tree",
        AssetPolicy.SINGLE,
        AssetPolicy.NONE,
        text_input=False,
        target_format=TargetFormat.XML,
    ),
    TaskContract(
        "form_filling",
        AssetPolicy.SINGLE,
        AssetPolicy.NONE,
        text_input=False,
        target_format=TargetFormat.JSON,
    ),
    # ── Page render plus a structured text payload ────────────────────────────
    TaskContract(
        "dom_edition",
        AssetPolicy.SINGLE,
        AssetPolicy.NONE,
        text_input=True,
        target_format=TargetFormat.XML,
    ),
    TaskContract(
        "reading_order",
        AssetPolicy.SINGLE,
        AssetPolicy.NONE,
        text_input=True,
        target_format=TargetFormat.JSON,
    ),
    # ── Leak avoidance: the naive input would contain the answer ──────────────
    TaskContract(
        "doc_reconstruction",
        AssetPolicy.NONE,
        AssetPolicy.NONE,
        text_input=True,
        target_format=TargetFormat.JSON,
        bbox_bearing=True,
        notes=(
            "No page image by design. Showing the page turns inferring the "
            "layout into reading the coordinates off the picture; the absence "
            "of the image is the task."
        ),
    ),
    TaskContract(
        "caption_prediction",
        AssetPolicy.NONE,
        AssetPolicy.SINGLE,
        text_input=True,
        target_format=TargetFormat.TEXT,
        bbox_bearing=True,
        notes=(
            "Crop only, never the page: the caption is printed on the page inches from the figure."
        ),
    ),
    TaskContract(
        "figure_selection",
        AssetPolicy.NONE,
        AssetPolicy.MULTI,
        text_input=True,
        target_format=TargetFormat.JSON,
        notes=(
            "Crops plus page text, never the page render: the correct figure is "
            "visibly on that page."
        ),
    ),
    TaskContract(
        "page_sorting",
        AssetPolicy.MULTI,
        AssetPolicy.NONE,
        text_input=False,
        target_format=TargetFormat.JSON,
        redact_page_numbers=True,
        notes=(
            "Page-number regions are blacked out before the model sees them; "
            "otherwise the task degenerates to OCR of a page number."
        ),
    ),
    # ── Cross-page grounding: both asset kinds ────────────────────────────────
    TaskContract(
        "figure_ordering",
        AssetPolicy.MULTI,
        AssetPolicy.MULTI,
        text_input=True,
        target_format=TargetFormat.JSON,
    ),
)

CONTRACTS: dict[str, TaskContract] = {c.task_name: c for c in _CONTRACTS}


def get_contract(task_name: str) -> TaskContract:
    """Return the contract for ``task_name``.

    Args:
        task_name: Key into :data:`CONTRACTS`.

    Returns:
        The task's :class:`TaskContract`.

    Raises:
        KeyError: If the task has no declared contract.
    """
    try:
        return CONTRACTS[task_name]
    except KeyError:
        raise KeyError(
            f"No modality contract for task {task_name!r}. Known: {', '.join(sorted(CONTRACTS))}"
        ) from None


def geometric_augmentation_allowed(task_name: str) -> bool:
    """Return whether geometric augmentation is safe for ``task_name``.

    Crop, rotate, shear, perspective and pad-to-square all move pixels relative
    to the page frame. A task whose target holds ``[0, 1000]`` normalized
    coordinates has those coordinates silently invalidated by such a transform
    unless the identical transform is applied to the target too — a corruption
    that produces no error, only a worse model. Photometric augmentation
    (jitter, blur, JPEG noise, grayscale) is safe everywhere and is not gated
    by this function.

    Args:
        task_name: Key into :data:`CONTRACTS`.

    Returns:
        ``False`` for bbox-bearing tasks, ``True`` otherwise.

    Raises:
        KeyError: If the task has no declared contract.
    """
    return not get_contract(task_name).bbox_bearing
