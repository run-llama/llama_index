from unittest.mock import Mock, patch
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.google import (
    GmailToolSpec,
    GoogleCalendarToolSpec,
    GoogleSearchToolSpec,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in GoogleCalendarToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in GmailToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in GoogleSearchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_google_calendar_tool_spec_init_without_creds():
    """Test GoogleCalendarToolSpec initialization without credentials."""
    tool = GoogleCalendarToolSpec()
    assert tool.creds is None


def test_google_calendar_tool_spec_init_with_creds():
    """Test GoogleCalendarToolSpec initialization with credentials."""
    mock_creds = Mock()
    tool = GoogleCalendarToolSpec(creds=mock_creds)
    assert tool.creds is mock_creds


def test_google_calendar_tool_spec_get_credentials_with_provided_creds():
    """Test _get_credentials returns provided credentials when available."""
    mock_creds = Mock()
    tool = GoogleCalendarToolSpec(creds=mock_creds)

    credentials = tool._get_credentials()
    assert credentials is mock_creds


@patch("os.path.exists")
@patch("google.oauth2.credentials.Credentials.from_authorized_user_file")
def test_google_calendar_tool_spec_get_credentials_oauth_flow(
    mock_from_file, mock_exists
):
    """Test _get_credentials falls back to OAuth flow when no creds provided."""
    mock_exists.return_value = True
    mock_creds = Mock()
    mock_creds.valid = True
    mock_from_file.return_value = mock_creds

    tool = GoogleCalendarToolSpec()  # No creds provided

    credentials = tool._get_credentials()
    assert credentials is mock_creds
    mock_from_file.assert_called_once_with(
        "token.json", ["https://www.googleapis.com/auth/calendar"]
    )


def test_google_calendar_tool_spec_init_with_allowed_calendar_ids():
    """Test GoogleCalendarToolSpec initialization with allowed calendar IDs."""
    allowed_ids = ["test@example.com", "secondary@example.com"]
    tool = GoogleCalendarToolSpec(allowed_calendar_ids=allowed_ids)
    assert tool.allowed_calendar_ids == allowed_ids


def test_calendar_id_validation_error():
    """Test calendar ID validation returns structured error."""
    allowed_ids = ["test@example.com"]
    tool = GoogleCalendarToolSpec(allowed_calendar_ids=allowed_ids)

    result = tool._validate_calendar_id("invalid@example.com")
    expected = {"error": "Invalid calendar_id", "allowed_values": ["test@example.com"]}
    assert result == expected


def test_calendar_id_validation_success():
    """Test calendar ID validation returns None for valid ID."""
    allowed_ids = ["test@example.com", "secondary@example.com"]
    tool = GoogleCalendarToolSpec(allowed_calendar_ids=allowed_ids)

    result = tool._validate_calendar_id("test@example.com")
    assert result is None
