from unittest.mock import Mock, patch, mock_open
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


@patch("os.path.isfile")
@patch("google.oauth2.credentials.Credentials.from_authorized_user_file")
def test_google_calendar_tool_spec_get_credentials_oauth_flow(
    mock_from_file, mock_isfile
):
    """Test _get_credentials falls back to OAuth flow when no creds provided."""
    # isfile returns False for service_account_key_path but True for token_path.
    mock_isfile.side_effect = lambda path: path == "token.json"

    mock_creds = Mock()
    mock_creds.valid = True
    mock_from_file.return_value = mock_creds

    tool = GoogleCalendarToolSpec()  # No creds provided

    credentials = tool._get_credentials()
    assert credentials is mock_creds
    mock_from_file.assert_called_once_with(
        "token.json", ["https://www.googleapis.com/auth/calendar"]
    )


@patch("os.path.isfile", return_value=False)
@patch("google.oauth2.service_account.Credentials.from_service_account_info")
def test_google_calendar_service_account_key_dict(mock_sa, mock_isfile):
    """Test service account key dict is used when provided."""
    mock_creds = Mock()
    mock_sa.return_value = mock_creds
    key_dict = {"type": "service_account", "project_id": "test"}

    tool = GoogleCalendarToolSpec(service_account_key=key_dict)
    credentials = tool._get_credentials()

    assert credentials is mock_creds
    mock_sa.assert_called_once_with(
        key_dict, scopes=["https://www.googleapis.com/auth/calendar"]
    )


def test_google_calendar_creds_takes_precedence():
    """Test that pre-built creds take precedence over service account."""
    mock_creds = Mock()
    key_dict = {"type": "service_account", "project_id": "test"}

    tool = GoogleCalendarToolSpec(creds=mock_creds, service_account_key=key_dict)
    credentials = tool._get_credentials()

    assert credentials is mock_creds


def test_google_calendar_custom_paths():
    """Test that custom paths are stored as attributes."""
    tool = GoogleCalendarToolSpec(
        credentials_path="/custom/creds.json",
        token_path="/custom/token.json",
        service_account_key_path="/custom/sa.json",
    )
    assert tool.credentials_path == "/custom/creds.json"
    assert tool.token_path == "/custom/token.json"
    assert tool.service_account_key_path == "/custom/sa.json"


@patch("os.path.isfile", return_value=False)
@patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file")
def test_google_calendar_is_cloud_no_file_write(mock_flow_cls, mock_isfile):
    """Test that is_cloud=True skips writing token file."""
    mock_creds = Mock()
    mock_creds.to_json.return_value = "{}"
    mock_flow = Mock()
    mock_flow.run_local_server.return_value = mock_creds
    mock_flow_cls.return_value = mock_flow

    tool = GoogleCalendarToolSpec(
        credentials_path="creds.json",
        is_cloud=True,
    )

    with patch("builtins.open", mock_open()) as mocked_file:
        credentials = tool._get_credentials()
        mocked_file.assert_not_called()

    assert credentials is mock_creds


# --- Gmail tests ---


def test_gmail_tool_spec_init_without_creds():
    """Test GmailToolSpec initialization without credentials."""
    tool = GmailToolSpec()
    assert tool.creds is None
    assert tool.credentials_path == "credentials.json"
    assert tool.token_path == "token.json"
    assert tool.is_cloud is False


def test_gmail_tool_spec_init_with_creds():
    """Test GmailToolSpec initialization with credentials."""
    mock_creds = Mock()
    tool = GmailToolSpec(creds=mock_creds)
    assert tool.creds is mock_creds


def test_gmail_get_credentials_with_provided_creds():
    """Test _get_credentials returns provided credentials when available."""
    mock_creds = Mock()
    tool = GmailToolSpec(creds=mock_creds)

    credentials = tool._get_credentials()
    assert credentials is mock_creds


@patch("os.path.isfile", return_value=False)
@patch("google.oauth2.service_account.Credentials.from_service_account_info")
def test_gmail_service_account_key_dict(mock_sa, mock_isfile):
    """Test service account key dict works for Gmail."""
    mock_creds = Mock()
    mock_sa.return_value = mock_creds
    key_dict = {"type": "service_account", "project_id": "test"}

    tool = GmailToolSpec(service_account_key=key_dict)
    credentials = tool._get_credentials()

    assert credentials is mock_creds
    mock_sa.assert_called_once_with(
        key_dict,
        scopes=[
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/gmail.readonly",
        ],
    )
