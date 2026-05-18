"""Zero-shot figure classification using SigLIP2 text-image embeddings.

This module provides a registry of figure categories with verbose, natural-language
descriptions designed for zero-shot classification. Each label maps to a detailed
description fed directly to SigLIP2's text encoder to maximize cosine similarity
separation between categories.
"""

from __future__ import annotations

# Maps label slugs (kebab-case) to verbose natural-language descriptions fed directly
# to the SigLIP2 text encoder. Descriptions are intentionally long and specific to
# improve zero-shot cosine similarity separation between categories.
LABEL_REGISTRY: dict[str, str] = {
    "natural-image": (
        "a photograph or camera-captured image of a real-world scene, including outdoor"
        " landscapes, indoor environments, portraits of people, animals, everyday objects,"
        " food, architecture, or any scene recorded by a physical lens without synthetic"
        " construction"
    ),
    "diagram": (
        "a technical or conceptual diagram such as a flowchart, block diagram, system"
        " architecture schematic, UML diagram, neural network topology, pipeline"
        " illustration, or process map — composed primarily of boxes, arrows, nodes, and"
        " labeled connectors on a plain or minimal background"
    ),
    "scientific-medical-image": (
        "a scientific or biomedical imaging artifact such as a microscopy scan (light,"
        " electron, fluorescence), MRI or CT slice, X-ray, histology slide, satellite or"
        " aerial remote sensing image, astronomical photograph, or any image produced by a"
        " specialized scientific instrument rather than a conventional camera"
    ),
    "illustration-artwork": (
        "a hand-drawn or digitally illustrated image including artwork, cartoons, concept"
        " art, graphic novel panels, decorative borders, icons drawn in illustrative style,"
        " clipart, or any image that is clearly synthetic and artistic rather than"
        " photographic or diagrammatic"
    ),
    "screenshot": (
        "a rasterized capture of a software interface, operating system desktop, web"
        " browser, mobile app screen, terminal or command-line output, IDE or code editor"
        " window, or any image that shows a digital user interface as it appeared on a screen"
    ),
    "infographic": (
        "a composed visual communication graphic that integrates text labels, icons,"
        " statistics, timelines, or illustrations to explain a concept or dataset —"
        " typically with a designed layout, brand colors, and mixed text-and-image elements"
        " aimed at a general audience"
    ),
    "logo-icon": (
        "a company logo, product brand mark, certification badge, institutional seal, app"
        " icon, or small symbolic graphic typically displayed at low resolution on a white"
        " or transparent background with minimal surrounding context"
    ),
    "map": (
        "a geographic, topographic, or spatial layout map including world or regional maps,"
        " city street maps, choropleth or heat maps, floor plans, site plans, or any image"
        " whose primary purpose is to encode spatial relationships between locations"
    ),
    "signature-handwriting": (
        "a handwritten signature, cursive handwriting sample, handwritten annotation,"
        " marginalia, or any image dominated by ink strokes produced by a human hand rather"
        " than printed or typeset text"
    ),
    "rasterized-table": (
        "a table or structured data grid that has been rendered as an image rather than"
        " encoded as HTML or text, showing rows, columns, cell borders, and tabular numeric"
        " or categorical content — visually distinct from charts and from text paragraphs"
    ),
}
