"""Tests for HVFToolSpec."""

from unittest.mock import MagicMock, patch

import pytest

from llama_index.tools.hvf.base import HVFToolSpec


@pytest.fixture()
def hvf_tool() -> HVFToolSpec:
    return HVFToolSpec()


class TestHealthCheck:
    def test_health_check_success(self, hvf_tool: HVFToolSpec) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None

        with patch.object(hvf_tool._session, "get", return_value=mock_resp):
            result = hvf_tool.health_check()

        assert result["status"] == "ok"
        assert "healthy" in result["message"].lower()

    def test_health_check_failure(self, hvf_tool: HVFToolSpec) -> None:
        import requests

        with patch.object(
            hvf_tool._session,
            "get",
            side_effect=requests.ConnectionError("connection refused"),
        ):
            result = hvf_tool.health_check()

        assert result["status"] == "error"
        assert "connection refused" in result["message"]


class TestGetServices:
    def test_get_services_all(self, hvf_tool: HVFToolSpec) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = [
            {
                "name": "Timber Harvesting",
                "description": "Selective timber harvesting services.",
                "category": "residential",
                "url": "https://www.hudsonvalleyforestry.com/services.html",
            }
        ]

        with patch.object(hvf_tool._session, "get", return_value=mock_resp):
            result = hvf_tool.get_services()

        assert isinstance(result, list)
        assert result[0]["name"] == "Timber Harvesting"

    def test_get_services_with_category(self, hvf_tool: HVFToolSpec) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = []

        with patch.object(hvf_tool._session, "get", return_value=mock_resp) as mock_get:
            hvf_tool.get_services(category="commercial")
            call_kwargs = mock_get.call_args
            assert call_kwargs.kwargs["params"]["category"] == "commercial"

    def test_get_services_error(self, hvf_tool: HVFToolSpec) -> None:
        import requests

        with patch.object(
            hvf_tool._session,
            "get",
            side_effect=requests.Timeout("timed out"),
        ):
            result = hvf_tool.get_services()

        assert len(result) == 1
        assert "error" in result[0]


class TestSubmitResidentialInquiry:
    def test_submit_success(self, hvf_tool: HVFToolSpec) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "message": "Inquiry received.",
            "inquiry_id": "RES-001",
        }

        with patch.object(hvf_tool._session, "post", return_value=mock_resp):
            result = hvf_tool.submit_residential_inquiry(
                name="Jane Doe",
                email="jane@example.com",
                phone="845-555-0001",
                address="123 Forest Rd, Woodstock, NY",
                service_type="Tree Removal",
                message="I need three large oaks removed.",
                acreage=2.5,
            )

        assert result["success"] is True
        assert result["inquiry_id"] == "RES-001"

    def test_submit_http_error(self, hvf_tool: HVFToolSpec) -> None:
        import requests

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "Invalid email address"}
        http_exc = requests.HTTPError(response=mock_resp)

        with patch.object(
            hvf_tool._session, "post", side_effect=http_exc
        ):
            result = hvf_tool.submit_residential_inquiry(
                name="Jane Doe",
                email="not-an-email",
                phone="845-555-0001",
                address="123 Forest Rd",
                service_type="Tree Removal",
                message="Help needed.",
            )

        assert result["success"] is False
        assert "Invalid email" in result["message"]


class TestSubmitCommercialInquiry:
    def test_submit_commercial_success(self, hvf_tool: HVFToolSpec) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "message": "Commercial inquiry received.",
            "inquiry_id": "COM-042",
        }

        with patch.object(hvf_tool._session, "post", return_value=mock_resp) as mock_post:
            result = hvf_tool.submit_commercial_inquiry(
                company_name="Acme Land Co.",
                contact_name="Bob Smith",
                email="bob@acme.com",
                phone="845-555-0099",
                address="500 Industrial Blvd, Kingston, NY",
                service_type="Land Development Clearing",
                message="We need 40 acres cleared by spring.",
                acreage=40.0,
                project_timeline="Spring 2026",
            )

            payload = mock_post.call_args.kwargs["json"]
            assert payload["client_type"] == "commercial"
            assert payload["project_timeline"] == "Spring 2026"

        assert result["success"] is True
        assert result["inquiry_id"] == "COM-042"


class TestToolList:
    def test_tool_list_has_expected_tools(self, hvf_tool: HVFToolSpec) -> None:
        tools = hvf_tool.to_tool_list()
        tool_names = [t.metadata.name for t in tools]
        assert "health_check" in tool_names
        assert "get_services" in tool_names
        assert "submit_residential_inquiry" in tool_names
        assert "submit_commercial_inquiry" in tool_names
