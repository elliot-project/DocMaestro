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
        "A cropped image from a document showing a radiological medical scan, such as a grayscale X-ray, CT, or MRI.",
        "A highly magnified microscopy image or colorful histology slide showing cellular structures and tissues.",
        "A diagnostic biomedical scan, like an ultrasound, displaying organic internal anatomy in black and white.",
        "A specialized scientific scan from laboratory equipment, such as a textured electron microscope image.",
        "A false-color satellite rendering or deep space astronomical photograph captured by non-optical sensors."
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
        "An isolated company logo centered on a plain white background with no other elements.",
        "A standalone graphic symbol or brand mark entirely surrounded by empty space.",
        "A cropped document image consisting solely of a corporate logo on a solid, blank background.",
        "An isolated icon, institutional seal, or emblem with no surrounding scenery or context.",
        "A single vectorized corporate trademark isolated on a blank canvas."
    ],
    "map": [
        "A geographic map displaying landmasses, oceans, and country borders.",
        "A street map for navigation showing roads, highways, and city blocks.",
        "A cartographic topographic map illustrating geographical terrain and elevation.",
        "A political map from an atlas highlighting countries, states, and geographic locations.",
        "An architectural floor plan or building blueprint showing rooms, walls, and building layouts."
    ],
    "signature-handwriting": [
        "A cropped image from a document showing a handwritten signature.",
        "A snippet of human cursive handwriting.",
        "A signature signed in pen on a document.",
        "A close-up of ink strokes forming handwritten words or initials.",
        "A manual handwritten annotation or sign-off.",
    ]
}
