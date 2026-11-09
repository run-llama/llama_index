"""The OAuth token file written after an interactive login must be owner-only."""

import os
import stat
from unittest.mock import Mock, patch

import pytest
from llama_index.tools.google import GmailToolSpec, GoogleCalendarToolSpec

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX permission bits are not enforced on Windows"
)

TOKEN_JSON = '{"refresh_token": "secret"}'


@pytest.fixture
def umask_022():
    previous = os.umask(0o022)
    yield
    os.umask(previous)


@pytest.mark.parametrize("spec_cls", [GmailToolSpec, GoogleCalendarToolSpec])
@patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file")
def test_token_file_is_owner_only(mock_flow, spec_cls, tmp_path, umask_022):
    creds = Mock()
    creds.to_json.return_value = TOKEN_JSON
    mock_flow.return_value.run_local_server.return_value = creds
    token_path = tmp_path / "token.json"

    spec = spec_cls(credentials_path="creds.json", token_path=str(token_path))
    assert spec._get_credentials() is creds

    assert token_path.read_text(encoding="utf-8") == TOKEN_JSON
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
