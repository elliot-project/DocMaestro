"""Tests for scripts/benchmark_throughput.py — GPU index resolution.

The script previously set ``CUDA_VISIBLE_DEVICES`` *after* the warm-up
``extract()`` call, which is too late: vLLM initialises CUDA during warm-up,
so the env var has no effect on which physical GPU(s) actually run compute.
``_resolve_gpu_indices`` isolates the pure parsing logic so it can be called
before any extraction/model-init happens in ``main()``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_throughput.py"
_spec = importlib.util.spec_from_file_location("benchmark_throughput", _MODULE_PATH)
benchmark_throughput = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("benchmark_throughput", benchmark_throughput)
_spec.loader.exec_module(benchmark_throughput)

_resolve_gpu_indices = benchmark_throughput._resolve_gpu_indices


class TestResolveGpuIndices:
    def test_explicit_single_index(self) -> None:
        assert _resolve_gpu_indices("4", None) == [4]

    def test_explicit_comma_separated_list(self) -> None:
        assert _resolve_gpu_indices("4,5", None) == [4, 5]

    def test_explicit_list_tolerates_whitespace(self) -> None:
        assert _resolve_gpu_indices(" 4, 5 ", None) == [4, 5]

    def test_falls_back_to_env_when_arg_unset(self) -> None:
        assert _resolve_gpu_indices(None, "2,3") == [2, 3]

    def test_falls_back_to_zero_when_arg_and_env_unset(self) -> None:
        assert _resolve_gpu_indices(None, None) == [0]

    def test_falls_back_to_zero_when_env_is_empty_string(self) -> None:
        assert _resolve_gpu_indices(None, "") == [0]

    @pytest.mark.parametrize("env_value", ["2,3", " 2 , 3 "])
    def test_env_fallback_tolerates_whitespace(self, env_value: str) -> None:
        assert _resolve_gpu_indices(None, env_value) == [2, 3]
