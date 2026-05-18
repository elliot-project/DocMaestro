"""Figure category label registry for SigLIP2 zero-shot classification.

Each label maps to a list of natural-language descriptions. Multiple descriptions
per label are encoded individually and mean-pooled into a single representative
embedding, which improves zero-shot accuracy by covering different visual phrasings
of the same category (prompt ensembling).
"""

from __future__ import annotations

LABEL_REGISTRY: dict[str, list[str]] = {
    "natural-image": [
        "A cropped image from a document showing a regular photograph.",
        "A camera-captured photo of a real-world scene, object, or person.",
        "A natural color photograph embedded in a page.",
        "A picture of an outdoor landscape or indoor environment.",
        "A photographic image taken with a real camera lens.",
    ],
    "diagram": [
        "A cropped image from a document showing a technical diagram.",
        "A flowchart, process map, or schematic with interconnected lines.",
        "A structural diagram with geometric shapes, arrows, and minimal text labels.",
        "An engineering or system architecture schematic diagram.",
        "A visual representation using connected nodes, blocks, and pathways.",
    ],
    "scientific-medical-image": [
        "A cropped image from a document showing a scientific scan.",
        "A medical imaging scan such as an X-ray, MRI, or ultrasound.",
        "A specialized technical image from a microscope or scientific instrument.",
        "An astronomical photograph or satellite remote sensing image.",
        "A biomedical scan, histology slide, or scientific visual data.",
    ],
    "illustration-artwork": [
        "A cropped image from a document showing a digital illustration.",
        "A piece of hand-drawn artwork, cartoon, or vector graphic.",
        "A purely synthetic, artistic drawing or graphic design element.",
        "A painted, sketched, or digitally rendered artistic image.",
        "A decorative illustration, concept art, or stylized graphic.",
    ],
    "screenshot": [
        "A cropped image from a document showing a software screenshot.",
        "A picture of a computer screen, web browser, or digital user interface.",
        "A screen capture of an operating system desktop or application window.",
        "An image showing a digital app interface or mobile device screen.",
        "A rasterized capture of a software interface or terminal output.",
    ],
    "infographic": [
        "A cropped image from a document showing a stylized infographic.",
        "A composed visual graphic combining colorful charts, icons, and text blocks.",
        "An informational design layout aimed at explaining data visually.",
        "A mixed graphic containing statistical visual elements and explanatory text.",
        "A designed presentation slide or graphic summarizing information.",
    ],
    "logo-icon": [
        "A cropped image from a document showing a company logo.",
        "A small brand mark, institutional seal, or graphical badge.",
        "A standalone corporate symbol or application icon on a plain background.",
        "A vectorized or high-contrast graphic symbol representing a brand.",
        "A small emblem, trademark, or decorative icon graphic.",
    ],
    "map": [
        "A cropped image from a document showing a map.",
        "A geographic layout, street map, or topographic region.",
        "An architectural floor plan or schematic showing spatial locations.",
        "A visual representation of a geographical area or city layout.",
        "A spatial map showing borders, roads, or geographical data.",
    ],
    "signature-handwriting": [
        "A cropped image from a document showing a handwritten signature.",
        "A snippet of human cursive handwriting.",
        "A signature signed in pen on a document.",
        "A close-up of ink strokes forming handwritten words or initials.",
        "A manual handwritten annotation or sign-off.",
    ],
    "rasterized-table": [
        "A cropped image from a document showing a data table.",
        "A structured table with visible grid lines, rows, and columns.",
        "A tabular layout containing aligned numbers and text categories.",
        "An image of a spreadsheet grid with organized data cells.",
        "A visual matrix of rows and columns formatting tabular data.",
    ],
}
