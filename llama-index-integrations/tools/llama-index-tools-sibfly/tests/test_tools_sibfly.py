"""Unit tests for llama-index-tools-sibfly (no network)."""

import pytest

from llama_index.tools.sibfly import SibflyToolSpec


def test_requires_key(monkeypatch):
    monkeypatch.delenv("SIBFLY_API_KEY", raising=False)
    with pytest.raises(ValueError):
        SibflyToolSpec()


def test_spec_functions():
    spec = SibflyToolSpec(api_key="sf_test")
    assert "check_ground_motion" in spec.spec_functions
    assert "check_coverage" in spec.spec_functions
    tools = spec.to_tool_list()
    assert len(tools) == 2


def test_params_by_address():
    spec = SibflyToolSpec(api_key="sf_test")
    assert spec._params("1100 Congress Ave", None, None) == {"address": "1100 Congress Ave"}


def test_params_by_coords():
    spec = SibflyToolSpec(api_key="sf_test")
    assert spec._params(None, 30.3, -97.8) == {"lat": 30.3, "lon": -97.8}


def test_params_requires_location():
    spec = SibflyToolSpec(api_key="sf_test")
    with pytest.raises(ValueError):
        spec._params(None, None, None)


def test_shape_keeps_and_drops():
    shaped = SibflyToolSpec._shape(
        {
            "status": "ok",
            "velocity_vertical_mm_yr": -6.0,
            "assessment_code": "notable_subsidence",
            "cost_usd": 0.4,
            "seasonal_amplitude_mm": None,
            "internal_debug": "x",
            "query": {"address": "a", "lat": 1.0, "lon": 2.0, "geocoded_address": None},
        }
    )
    assert shaped["velocity_vertical_mm_yr"] == -6.0
    assert "seasonal_amplitude_mm" not in shaped
    assert "internal_debug" not in shaped
    assert shaped["query"] == {"address": "a", "lat": 1.0, "lon": 2.0}
