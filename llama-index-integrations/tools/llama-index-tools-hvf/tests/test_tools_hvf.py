"""Tests for HVFToolSpec."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from llama_index.tools.hvf.base import HVFToolSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tool() -> HVFToolSpec:
    return HVFToolSpec()


def _mock_resp(json_data, status_code=200):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        http_err = requests.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
    return resp


# ---------------------------------------------------------------------------
# spec_functions
# ---------------------------------------------------------------------------


def test_spec_functions(tool):
    assert set(tool.spec_functions) == {
        "hvf_get_services",
        "hvf_assess_property",
        "hvf_submit_quote",
        "hvg_get_services",
        "hvg_assess_property",
        "hvg_submit_quote",
        "og_get_services",
        "og_assess_project",
        "og_submit_quote",
    }
    assert len(tool.to_tool_list()) == 9


# ---------------------------------------------------------------------------
# HVF Residential
# ---------------------------------------------------------------------------


def test_hvf_get_services_success(tool):
    services = [{"service_type": "forestry_mulching", "base_price_per_acre": 800}]
    with patch.object(tool._session, "get", return_value=_mock_resp(services)):
        result = tool.hvf_get_services()
    assert result == services


def test_hvf_get_services_error(tool):
    with patch.object(
        tool._session, "get", side_effect=requests.ConnectionError("refused")
    ):
        result = tool.hvf_get_services()
    assert result["error"] == "request_failed"


def test_hvf_assess_property_eligible(tool):
    resp_data = {
        "eligible": True,
        "price_estimate_usd": {"low": 4000, "high": 6000},
        "lead_time_days": 14,
    }
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)) as mock_post:
        result = tool.hvf_assess_property(
            lat=41.8, lng=-73.9, acreage=5.0, service_type="forestry_mulching"
        )
        call_json = mock_post.call_args.kwargs["json"]
    assert result["eligible"] is True
    assert call_json["lat"] == 41.8
    assert call_json["vegetation_density"] == "unknown"


def test_hvf_assess_property_with_density(tool):
    resp_data = {"eligible": True, "price_estimate_usd": {"low": 5000, "high": 7000}}
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)) as mock_post:
        tool.hvf_assess_property(
            lat=41.8, lng=-73.9, acreage=5.0,
            service_type="forestry_mulching", vegetation_density="heavy"
        )
        call_json = mock_post.call_args.kwargs["json"]
    assert call_json["vegetation_density"] == "heavy"


def test_hvf_submit_quote_success(tool):
    resp_data = {"quote_id": "Q-12345", "message": "Quote received"}
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)) as mock_post:
        result = tool.hvf_submit_quote(
            email="test@example.com",
            name="Jane Smith",
            acreage=3.0,
            service_type="selective_thinning",
            property_description="Wooded lot with overgrown brush",
        )
        call_json = mock_post.call_args.kwargs["json"]
    assert result["quote_id"] == "Q-12345"
    assert "phone" not in call_json  # optional fields omitted when empty


def test_hvf_submit_quote_with_optionals(tool):
    resp_data = {"quote_id": "Q-99", "message": "ok"}
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)) as mock_post:
        tool.hvf_submit_quote(
            email="a@b.com", name="X", acreage=1.0,
            service_type="forestry_mulching",
            property_description="test",
            phone="845-555-1234",
            address="123 Main St",
            lat=41.8, lng=-73.9,
        )
        call_json = mock_post.call_args.kwargs["json"]
    assert call_json["phone"] == "845-555-1234"
    assert call_json["lat"] == 41.8


# ---------------------------------------------------------------------------
# HVG Goat Grazing
# ---------------------------------------------------------------------------


def test_hvg_get_services_success(tool):
    services = [{"service_type": "goat_grazing", "base_price_per_acre": 500}]
    with patch.object(tool._session, "get", return_value=_mock_resp(services)):
        result = tool.hvg_get_services()
    assert result == services


def test_hvg_assess_property_eligible(tool):
    resp_data = {"eligible": True, "price_estimate_usd": {"low": 1500, "high": 2500}}
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)) as mock_post:
        result = tool.hvg_assess_property(lat=41.8, lng=-73.9, acreage=2.0)
        call_json = mock_post.call_args.kwargs["json"]
    assert result["eligible"] is True
    assert call_json["vegetation_type"] == "unknown"


def test_hvg_assess_property_ineligible(tool):
    resp_data = {"eligible": False, "message": "Outside service area"}
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)):
        result = tool.hvg_assess_property(lat=34.0, lng=-118.0, acreage=2.0)
    assert result["eligible"] is False


def test_hvg_submit_quote_success(tool):
    resp_data = {"quote_id": "GQ-001", "message": "Goat quote received"}
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)):
        result = tool.hvg_submit_quote(
            email="goat@example.com", name="Bob",
            acreage=1.5, service_type="goat_grazing",
            property_description="Invasive knotweed on hillside",
        )
    assert result["quote_id"] == "GQ-001"


# ---------------------------------------------------------------------------
# Commercial O&G
# ---------------------------------------------------------------------------


def test_og_get_services_success(tool):
    services = [{"service_type": "row_clearing", "description": "Pipeline ROW clearing"}]
    with patch.object(tool._session, "get", return_value=_mock_resp(services)):
        result = tool.og_get_services()
    assert result == services


def test_og_assess_project_eligible(tool):
    resp_data = {"eligible": True, "message": "Location in Northeast service area"}
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)) as mock_post:
        result = tool.og_assess_project(
            lat=41.5, lng=-74.0, service_type="row_clearing",
            corridor_miles=12.5
        )
        call_json = mock_post.call_args.kwargs["json"]
    assert result["eligible"] is True
    assert call_json["corridor_miles"] == 12.5
    assert "project_description" not in call_json  # empty string omitted


def test_og_assess_project_minimal(tool):
    resp_data = {"eligible": True}
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)) as mock_post:
        tool.og_assess_project(lat=40.7, lng=-74.0)
        call_json = mock_post.call_args.kwargs["json"]
    assert "service_type" not in call_json
    assert "acreage" not in call_json


def test_og_submit_quote_success(tool):
    resp_data = {"quote_id": "CQ-555", "message": "Commercial quote received"}
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)) as mock_post:
        result = tool.og_submit_quote(
            email="pm@pipeline.com", name="Alice Johnson",
            service_type="row_clearing",
            project_description="12-mile natural gas pipeline corridor through Catskills",
            company="Northeast Pipeline LLC",
            corridor_miles=12.0,
            timeline="Q3 2026",
        )
        call_json = mock_post.call_args.kwargs["json"]
    assert result["quote_id"] == "CQ-555"
    assert call_json["company"] == "Northeast Pipeline LLC"
    assert call_json["timeline"] == "Q3 2026"


def test_og_submit_quote_omits_empty_optionals(tool):
    resp_data = {"quote_id": "CQ-1"}
    with patch.object(tool._session, "post", return_value=_mock_resp(resp_data)) as mock_post:
        tool.og_submit_quote(
            email="x@y.com", name="X",
            service_type="site_prep",
            project_description="Compressor pad clearing",
        )
        call_json = mock_post.call_args.kwargs["json"]
    assert "company" not in call_json
    assert "phone" not in call_json
    assert "timeline" not in call_json


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


def test_http_error_returns_error_dict(tool):
    with patch.object(tool._session, "post", return_value=_mock_resp({}, 422)):
        result = tool.hvf_assess_property(lat=0, lng=0, acreage=1.0, service_type="x")
    assert "error" in result


def test_connection_error_returns_error_dict(tool):
    with patch.object(
        tool._session, "get", side_effect=requests.ConnectionError("refused")
    ):
        result = tool.og_get_services()
    assert result["error"] == "request_failed"
