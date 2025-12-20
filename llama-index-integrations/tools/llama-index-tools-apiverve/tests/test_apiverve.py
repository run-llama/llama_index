"""Tests for APIVerve tool specification."""

import pytest
from unittest.mock import patch, MagicMock

from llama_index.tools.apiverve import APIVerveToolSpec


# Mock schema data for testing
MOCK_SCHEMAS = {
    "emailvalidator": {
        "apiId": "emailvalidator",
        "title": "Email Validator",
        "description": "Validate email addresses",
        "category": "Validation",
        "methods": ["GET"],
        "parameters": [
            {"name": "email", "type": "string", "required": True}
        ]
    },
    "dnslookup": {
        "apiId": "dnslookup",
        "title": "DNS Lookup",
        "description": "Lookup DNS records for a domain",
        "category": "Lookup",
        "methods": ["GET"],
        "parameters": [
            {"name": "domain", "type": "string", "required": True}
        ]
    }
}


@pytest.fixture
def mock_schemas():
    """Mock the schema loading."""
    with patch('llama_index.tools.apiverve.base._load_schemas', return_value=MOCK_SCHEMAS):
        yield


def test_requires_api_key(mock_schemas):
    """Test that API key is required."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="API key is required"):
            APIVerveToolSpec()


def test_initialization(mock_schemas):
    """Test successful initialization."""
    spec = APIVerveToolSpec(api_key="test-key")
    assert spec.api_key == "test-key"
    assert len(spec._schemas) == 2


def test_list_categories(mock_schemas):
    """Test listing categories."""
    spec = APIVerveToolSpec(api_key="test-key")
    categories = spec.list_categories()
    assert "Validation" in categories
    assert "Lookup" in categories


def test_list_available_apis(mock_schemas):
    """Test listing available APIs."""
    spec = APIVerveToolSpec(api_key="test-key")
    apis = spec.list_available_apis()
    assert len(apis) == 2

    # Test category filter
    validation_apis = spec.list_available_apis(category="Validation")
    assert len(validation_apis) == 1
    assert validation_apis[0]["id"] == "emailvalidator"


def test_list_available_apis_search(mock_schemas):
    """Test searching APIs."""
    spec = APIVerveToolSpec(api_key="test-key")

    # Search by keyword
    results = spec.list_available_apis(search="email")
    assert len(results) == 1
    assert results[0]["id"] == "emailvalidator"


def test_get_api_details(mock_schemas):
    """Test getting API details."""
    spec = APIVerveToolSpec(api_key="test-key")

    details = spec.get_api_details("emailvalidator")
    assert details is not None
    assert details["id"] == "emailvalidator"
    assert details["category"] == "Validation"
    assert len(details["parameters"]) == 1


def test_get_api_details_not_found(mock_schemas):
    """Test getting details for unknown API."""
    spec = APIVerveToolSpec(api_key="test-key")
    details = spec.get_api_details("unknown_api")
    assert details is None


def test_call_api_unknown(mock_schemas):
    """Test calling unknown API raises error."""
    spec = APIVerveToolSpec(api_key="test-key")

    with pytest.raises(ValueError, match="Unknown API"):
        spec.call_api("unknown_api", {})


def test_call_api_success(mock_schemas):
    """Test successful API call."""
    spec = APIVerveToolSpec(api_key="test-key")

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "ok", "data": {"valid": True}}
    mock_response.raise_for_status = MagicMock()

    with patch.object(spec._session, 'get', return_value=mock_response):
        result = spec.call_api("emailvalidator", {"email": "test@example.com"})
        assert result["status"] == "ok"
        assert result["data"]["valid"] is True


def test_spec_functions():
    """Test that spec_functions is properly defined."""
    assert "call_api" in APIVerveToolSpec.spec_functions
    assert "list_available_apis" in APIVerveToolSpec.spec_functions
    assert "list_categories" in APIVerveToolSpec.spec_functions
