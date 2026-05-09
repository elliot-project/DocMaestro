# XML Schema

## Overview

`src/xml_formatter.py` synthesizes MinerU's JSON outputs into a structured XML document that
embeds every content element with its type, text, bounding box, and page position. The format
is designed to be consumed by LLMs, RAG pipelines, and spatial-reasoning systems.

---

## Document Structure

```xml
<?xml version="1.0" ?>
<document name="macdonals" page_count="47" backend="hybrid-auto-engine">

  <page idx="0" width="612.0" height="792.0">
    <title level="1" bbox="[62, 20, 940, 68]">McDonald's Corporation</title>
    <text-block bbox="[62, 80, 940, 120]">Annual Report 2023</text-block>
    <figure bbox="[62, 130, 940, 400]">
      <img src="images/figure_p0_0.jpg"/>
      <caption>McDonald's global footprint as of December 2023.</caption>
    </figure>
    <table bbox="[62, 420, 940, 600]">
      <img src="images/table_p0_0.jpg"/>
      <caption>Key financial highlights</caption>
      <table-body>&lt;html&gt;&lt;body&gt;&lt;table&gt;...&lt;/table&gt;&lt;/body&gt;&lt;/html&gt;</table-body>
    </table>
  </page>

  <page idx="1" width="612.0" height="792.0">
    ...
  </page>

</document>
```

---

## Root Element: `<document>`

The root element wraps all pages.

| Attribute | Value | Description |
|-----------|-------|-------------|
| `name` | string | Stem of the source PDF filename |
| `page_count` | integer | Total number of pages detected by MinerU |
| `backend` | string | Backend used (from `middle.json._backend`) |

---

## Page Element: `<page>`

One `<page>` element per page containing content blocks. Pages are ordered by index.

| Attribute | Value | Description |
|-----------|-------|-------------|
| `idx` | integer | Zero-based page index |
| `width` | float | Page width in PDF points (from `middle.json page_size`) |
| `height` | float | Page height in PDF points |

---

## Content Elements

### Bounding Box Format

Every content element carries a `bbox` attribute when MinerU provides geometry:

```
bbox="[x1, y1, x2, y2]"
```

Coordinates are **integers** (floats truncated) in the **0–1000 normalized scale** used by
`content_list.json`, where `(0, 0)` is the top-left corner of the page and `(1000, 1000)` is
the bottom-right. This scale is independent of actual page dimensions.

---

### Full Tag Reference

| MinerU `type` | `text_level` | XML Tag | Extra Attributes | Children |
|---------------|-------------|---------|-----------------|---------|
| `text` | `0` | `<text-block>` | `bbox` | text content |
| `text` | `1` | `<title level="1">` | `bbox`, `level` | text content |
| `text` | `2` | `<title level="2">` | `bbox`, `level` | text content |
| `text` | `3` | `<title level="3">` | `bbox`, `level` | text content |
| `text` | `4` | `<title level="4">` | `bbox`, `level` | text content |
| `image` | — | `<figure>` | `bbox` | `<img>`, `<caption>*`, `<footnote>*` |
| `table` | — | `<table>` | `bbox` | `<img>`, `<caption>*`, `<table-body>`, `<footnote>*` |
| `chart` | — | `<chart>` | `bbox` | `<img>`, `<caption>*` |
| `equation` | — | `<math-block>` | `bbox` | text (LaTeX), `<img>` |
| `code` | — | `<code-block>` | `bbox`, `sub-type` | text content |
| `list` | — | `<list>` | `bbox` | text content |
| `header` | — | `<page-header>` | `bbox` | text content |
| `footer` | — | `<page-footer>` | `bbox` | text content |
| `page_number` | — | `<page-number>` | `bbox` | text content |
| `aside_text` | — | `<aside>` | `bbox` | text content |
| `page_footnote` | — | `<page-footnote>` | `bbox` | text content |
| `seal` | — | `<seal>` | `bbox` | text content |
| `index` | — | `<index>` | `bbox` | text content |
| *(unknown)* | — | `<block>` | `bbox` | text content |

---

### `<title>` — Section Headings

```xml
<title level="1" bbox="[62, 20, 940, 68]">Introduction</title>
<title level="2" bbox="[62, 90, 500, 115]">Background</title>
```

`level` ranges from `1` (top-level chapter/section) to `4` (paragraph-level heading).
Level assignment requires title-aided extraction; without it, all detected titles receive
`level="1"`.

---

### `<text-block>` — Body Text

```xml
<text-block bbox="[62, 130, 940, 170]">Revenue grew 8% year-over-year...</text-block>
```

Plain body paragraphs. `text_level=0` in MinerU.

---

### `<figure>` — Images

```xml
<figure bbox="[62, 200, 940, 500]">
  <img src="images/figure_p2_0.jpg"/>
  <caption>Figure 3. Revenue by region.</caption>
  <footnote>Source: Company filings.</footnote>
</figure>
```

`<img src>` is the relative path written by MinerU (relative to the `output_dir`).
Multiple `<caption>` and `<footnote>` children are possible.

---

### `<table>` — Tables

```xml
<table bbox="[62, 300, 940, 580]">
  <img src="images/table_p3_0.jpg"/>
  <caption>Table 2. Operating margins by segment</caption>
  <table-body>&lt;html&gt;&lt;body&gt;&lt;table&gt;...&lt;/table&gt;&lt;/body&gt;&lt;/html&gt;</table-body>
  <footnote>All figures in USD millions.</footnote>
</table>
```

`<table-body>` contains the raw HTML string from MinerU's table parser, HTML-escaped. Parse
with an HTML parser (e.g., BeautifulSoup, lxml) to access cell data.

---

### `<math-block>` — Equations

```xml
<math-block bbox="[62, 400, 600, 440]">$$Q_{\%} = \frac{P_{out}}{P_{in}} \times 100$$</math-block>
```

Text content is LaTeX. An `<img>` child may also be present with the rendered equation image.

---

### `<chart>` — Charts / Plots

```xml
<chart bbox="[62, 200, 940, 500]">
  <img src="images/chart_p5_0.jpg"/>
  <caption>Q3 performance versus benchmark.</caption>
</chart>
```

MinerU classifies charts separately from figures. No structured data is extracted; only the
image path and caption are preserved.

---

### `<code-block>` — Code

```xml
<code-block sub-type="algorithm" bbox="[62, 300, 940, 400]">
  Algorithm 1: Gradient descent with momentum
  ...
</code-block>
```

`sub-type` is the MinerU `sub_type` field (e.g., `"code"`, `"algorithm"`).

---

## Text Sanitization

MinerU's OCR output occasionally contains characters that are illegal in XML 1.0. The
formatter strips them before building the element tree:

```python
_INVALID_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
```

This regex covers:
- Null bytes (`\x00`)
- C0 control characters (except tab `\x09`, LF `\x0A`, CR `\x0D` which are valid in XML)
- DEL character (`\x7F`)

The sanitization is applied to all text content in every element type.

---

## Implementation: `src/xml_formatter.py`

### `build_document_xml(content_list, middle_json, doc_name) → str`

Main entry point. Groups content items by `page_idx`, creates `<page>` elements in index
order, dispatches each item to the appropriate element builder, and serializes with
`xml.dom.minidom` for pretty-printing.

### Element Builders

| Function | Handles |
|----------|---------|
| `_build_figure_element(item)` | `type=image` |
| `_build_chart_element(item)` | `type=chart` |
| `_build_table_element(item)` | `type=table` |
| `_build_equation_element(item)` | `type=equation` |
| `_build_code_element(item)` | `type=code` |
| `_build_generic_text_element(item)` | all text-bearing types (delegates to `xml_schema.resolve_tag`) |

### `save_document_xml(xml_str, output_path)`

Writes the XML string to `output/<doc_name>/<doc_name>_<backend>.xml`, creating parent
directories as needed.

---

## Implementation: `src/xml_schema.py`

### `resolve_tag(content_type, text_level) → XMLTagConfig`

Returns an `XMLTagConfig(tag, extra_attrs)` for any content type. The `text_level` parameter
distinguishes headings from body text: `text_level >= 1` maps to `<title level=N>`, while
`text_level = 0` maps to `<text-block>`.

### `format_bbox(bbox) → str | None`

Converts `[x0, y0, x1, y1]` (floats or ints) to the canonical `"[x1, y1, x2, y2]"` string.
Returns `None` if the input is absent or not exactly four values.
