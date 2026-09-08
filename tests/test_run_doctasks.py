"""Tests for scripts/run_doctasks.py — sample plumbing and shard emission.

``process_xml`` originally returned only JSON-ready payloads, which meant the
``Sample`` objects the composers produced were discarded and no WebDataset
shards could ever be written. These tests pin the contract that it hands back
both representations.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.task_composer.base import Sample

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_doctasks.py"
_spec = importlib.util.spec_from_file_location("run_doctasks", _MODULE_PATH)
run_doctasks = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("run_doctasks", run_doctasks)
_spec.loader.exec_module(run_doctasks)


def _write_doc_xml(path: Path, doc_name: str = "doc", n_pages: int = 2) -> Path:
    """Write a minimal document XML with text blocks on every page."""
    root = ET.Element("document", {"name": doc_name, "page_count": str(n_pages)})
    for idx in range(n_pages):
        page = ET.SubElement(root, "page", {"idx": str(idx), "width": "1000", "height": "1000"})
        for j in range(3):
            el = ET.SubElement(page, "text-block", {"bbox": f"[0, {j * 100}, 500, {j * 100 + 50}]"})
            el.text = f"page {idx} block {j}"
    path.write_bytes(ET.tostring(root))
    return path


@pytest.fixture
def doc_xml(tmp_path: Path) -> Path:
    return _write_doc_xml(tmp_path / "doc_hybrid-auto-engine.xml")


class TestProcessXml:
    def test_returns_result_and_samples(self, doc_xml):
        result, samples_by_task = run_doctasks.process_xml(doc_xml, ["basic_reading"])
        assert result["doc_name"] == "doc"
        assert set(samples_by_task) == {"basic_reading"}

    def test_samples_are_sample_objects(self, doc_xml):
        _, samples_by_task = run_doctasks.process_xml(doc_xml, ["basic_reading"])
        assert samples_by_task["basic_reading"]
        assert all(isinstance(s, Sample) for s in samples_by_task["basic_reading"])

    def test_payloads_match_samples(self, doc_xml):
        result, samples_by_task = run_doctasks.process_xml(doc_xml, ["basic_reading"])
        payloads = result["tasks"]["basic_reading"]
        samples = samples_by_task["basic_reading"]
        assert len(payloads) == len(samples)
        assert [p["__key__"] for p in payloads] == [s.key for s in samples]

    def test_failing_task_yields_no_samples(self, doc_xml, monkeypatch):
        def _boom(**kwargs):
            raise RuntimeError("composer exploded")

        spec = run_doctasks.registry.REGISTRY["basic_reading"]
        patched = {
            **run_doctasks.registry.REGISTRY,
            "basic_reading": dataclasses.replace(spec, composer_factory=_boom),
        }
        monkeypatch.setattr(run_doctasks.registry, "REGISTRY", patched)
        result, samples_by_task = run_doctasks.process_xml(doc_xml, ["basic_reading"])
        assert "composer exploded" in result["tasks"]["basic_reading"][0]["__error__"]
        assert samples_by_task["basic_reading"] == []


class TestShardEmission:
    def _run(self, tmp_path: Path, extra: list[str]) -> None:
        xml = _write_doc_xml(tmp_path / "doc_hybrid-auto-engine.xml")
        argv = ["run_doctasks.py", "--xml", str(xml), "--output-dir", str(tmp_path)]
        sys.argv = argv + extra
        run_doctasks.main()

    def test_no_shard_dir_writes_no_tars(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._run(tmp_path, ["--tasks", "basic_reading"])
        assert list(tmp_path.rglob("*.tar")) == []

    def test_shard_dir_writes_tars(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        self._run(tmp_path, ["--tasks", "basic_reading", "--shard-dir", str(shards)])
        assert (shards / "basic_reading-00000.tar").exists()

    def test_shard_contents_are_sample_payloads(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        self._run(tmp_path, ["--tasks", "basic_reading", "--shard-dir", str(shards)])
        with tarfile.open(shards / "basic_reading-00000.tar") as tar:
            names = tar.getnames()
        assert any(n.endswith(".target.txt") for n in names)
        assert not any("__key__" in n for n in names)

    def test_shard_size_is_respected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        self._run(
            tmp_path,
            ["--tasks", "basic_reading", "--shard-dir", str(shards), "--shard-size", "1"],
        )
        # 2 pages → 2 basic_reading samples → 2 single-sample shards
        assert sorted(p.name for p in shards.glob("*.tar")) == [
            "basic_reading-00000.tar",
            "basic_reading-00001.tar",
        ]

    def test_compress_writes_tar_gz(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        self._run(
            tmp_path,
            ["--tasks", "basic_reading", "--shard-dir", str(shards), "--compress"],
        )
        assert (shards / "basic_reading-00000.tar.gz").exists()

    def test_json_output_still_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        self._run(tmp_path, ["--tasks", "basic_reading", "--shard-dir", str(shards)])
        assert (tmp_path / "doc_doctasks.json").exists()
        assert (tmp_path / "doctasks_index.json").exists()


class TestImageResolution:
    """Images live under data/, not beside the XML in output/. The runner must
    resolve them or every image-bearing sample silently loses its bytes."""

    def _doc_with_figure(self, tmp_path: Path) -> tuple[Path, Path]:
        out_dir = tmp_path / "output" / "doc"
        out_dir.mkdir(parents=True)
        root = ET.Element("document", {"name": "doc", "page_count": "1"})
        page = ET.SubElement(root, "page", {"idx": "0", "width": "1000", "height": "1000"})
        fig = ET.SubElement(page, "figure", {"bbox": "[0, 0, 500, 500]"})
        ET.SubElement(fig, "img", {"src": "images/fig.jpg"})
        cap = ET.SubElement(fig, "caption")
        cap.text = "A figure caption"
        xml = out_dir / "doc_hybrid-auto-engine.xml"
        xml.write_bytes(ET.tostring(root))

        data_dir = tmp_path / "data"
        images = data_dir / "doc" / "backend" / "images"
        images.mkdir(parents=True)
        (images / "fig.jpg").write_bytes(b"\xff\xd8\xff-jpeg-bytes")
        return xml, data_dir

    def test_image_bytes_reach_the_sample(self, tmp_path):
        xml, data_dir = self._doc_with_figure(tmp_path)
        _, samples_by_task = run_doctasks.process_xml(
            xml, ["caption_prediction"], data_dir=data_dir, output_dir=tmp_path / "output"
        )
        payload = samples_by_task["caption_prediction"][0].payload
        assert payload["image.jpg"] == b"\xff\xd8\xff-jpeg-bytes"

    def test_missing_data_dir_still_produces_samples(self, tmp_path):
        xml, _ = self._doc_with_figure(tmp_path)
        _, samples_by_task = run_doctasks.process_xml(
            xml,
            ["caption_prediction"],
            data_dir=tmp_path / "absent",
            output_dir=tmp_path / "output",
        )
        payload = samples_by_task["caption_prediction"][0].payload
        assert "image.jpg" not in payload
        assert payload["target.txt"] == "A figure caption"

    def test_shards_carry_image_entries(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        xml, data_dir = self._doc_with_figure(tmp_path)
        shards = tmp_path / "shards"
        sys.argv = [
            "run_doctasks.py",
            "--xml",
            str(xml),
            "--output-dir",
            str(tmp_path / "output"),
            "--data-dir",
            str(data_dir),
            "--tasks",
            "caption_prediction",
            "--shard-dir",
            str(shards),
        ]
        run_doctasks.main()
        with tarfile.open(shards / "caption_prediction-00000.tar") as tar:
            names = tar.getnames()
        assert any(n.endswith(".image.jpg") for n in names), names

    def test_json_output_summarises_image_bytes(self, tmp_path, monkeypatch):
        """Binary belongs in the shards; the JSON index records it as a stub."""
        monkeypatch.chdir(tmp_path)
        xml, data_dir = self._doc_with_figure(tmp_path)
        sys.argv = [
            "run_doctasks.py",
            "--xml",
            str(xml),
            "--output-dir",
            str(tmp_path / "output"),
            "--data-dir",
            str(data_dir),
            "--tasks",
            "caption_prediction",
        ]
        run_doctasks.main()
        written = json.loads((tmp_path / "output" / "doc" / "doc_doctasks.json").read_text())
        payload = written["tasks"]["caption_prediction"][0]
        assert payload["image.jpg"] == {"__bytes__": len(b"\xff\xd8\xff-jpeg-bytes")}
