"""Unit tests for xml_schema.py — tag resolution and bbox formatting."""

from doc_maestro.utils.xml_schema import format_bbox, resolve_tag


class TestResolveTag:
    def test_plain_text_returns_text_block(self):
        config = resolve_tag("text", 0)
        assert config.tag == "text-block"
        assert config.extra_attrs == {}

    def test_title_level_1(self):
        config = resolve_tag("text", 1)
        assert config.tag == "title"
        assert config.extra_attrs == {"level": "1"}

    def test_title_level_3(self):
        config = resolve_tag("text", 3)
        assert config.tag == "title"
        assert config.extra_attrs == {"level": "3"}

    def test_image_type(self):
        config = resolve_tag("image")
        assert config.tag == "figure"

    def test_table_type(self):
        config = resolve_tag("table")
        assert config.tag == "table"

    def test_equation_type(self):
        config = resolve_tag("equation")
        assert config.tag == "math-block"

    def test_code_type(self):
        config = resolve_tag("code")
        assert config.tag == "code-block"

    def test_list_type(self):
        config = resolve_tag("list")
        assert config.tag == "list"

    def test_chart_type(self):
        config = resolve_tag("chart")
        assert config.tag == "chart"

    def test_header_type(self):
        config = resolve_tag("header")
        assert config.tag == "page-header"

    def test_footer_type(self):
        config = resolve_tag("footer")
        assert config.tag == "page-footer"

    def test_unknown_type_returns_fallback(self):
        config = resolve_tag("unknown_future_type")
        assert config.tag == "block"

    def test_text_level_zero_is_body_text(self):
        config = resolve_tag("text", 0)
        assert config.tag == "text-block"
        assert "level" not in config.extra_attrs


class TestFormatBbox:
    def test_integer_coords(self):
        assert format_bbox([100, 150, 400, 350]) == "[100, 150, 400, 350]"

    def test_float_coords_truncated_to_int(self):
        assert format_bbox([100.7, 150.2, 400.9, 350.1]) == "[100, 150, 400, 350]"

    def test_none_returns_none(self):
        assert format_bbox(None) is None

    def test_empty_list_returns_none(self):
        assert format_bbox([]) is None

    def test_wrong_length_returns_none(self):
        assert format_bbox([1, 2, 3]) is None

    def test_zero_coords(self):
        assert format_bbox([0, 0, 0, 0]) == "[0, 0, 0, 0]"

    def test_large_coords(self):
        result = format_bbox([1000, 1000, 5000, 5000])
        assert result == "[1000, 1000, 5000, 5000]"
