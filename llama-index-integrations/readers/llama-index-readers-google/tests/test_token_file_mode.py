"""The OAuth token file written after an interactive login must be owner-only."""

import json
import os
import stat
from unittest.mock import Mock, patch

import pytest
from llama_index.readers.google import (
    GmailReader,
    GoogleCalendarReader,
    GoogleChatReader,
    GoogleDocsReader,
    GoogleDriveReader,
    GoogleSheetsReader,
)

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX permission bits are not enforced on Windows"
)

TOKEN_JSON = '{"refresh_token": "secret"}'

# Captured at import time, before any test runs: test_readers_google_drive.py
# replaces GoogleDriveReader._get_credentials with a MagicMock and never
# restores it.
REAL_DRIVE_GET_CREDENTIALS = GoogleDriveReader._get_credentials


@pytest.fixture
def umask_022():
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def _creds():
    creds = Mock()
    creds.to_json.return_value = TOKEN_JSON
    return creds


@pytest.mark.parametrize(
    "make_reader",
    [
        pytest.param(
            lambda: GmailReader(service=Mock(), results_per_page=10), id="gmail"
        ),
        pytest.param(GoogleCalendarReader, id="calendar"),
        pytest.param(GoogleChatReader, id="chat"),
        pytest.param(GoogleDocsReader, id="docs"),
        pytest.param(GoogleSheetsReader, id="sheets"),
    ],
)
@patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file")
def test_token_json_in_cwd_is_owner_only(
    mock_flow, make_reader, tmp_path, monkeypatch, umask_022
):
    creds = _creds()
    mock_flow.return_value.run_local_server.return_value = creds
    monkeypatch.chdir(tmp_path)
    # The docs reader reads the redirect port from credentials.json.
    (tmp_path / "credentials.json").write_text(
        json.dumps({"web": {"redirect_uris": ["http://localhost:8080/"]}}),
        encoding="utf-8",
    )

    assert make_reader()._get_credentials() is creds

    token_path = tmp_path / "token.json"
    assert token_path.read_text(encoding="utf-8") == TOKEN_JSON
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


@patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_config")
def test_drive_token_file_is_owner_only(mock_flow, tmp_path, umask_022):
    creds = _creds()
    mock_flow.return_value.run_local_server.return_value = creds
    token_path = tmp_path / "token.json"
    reader = GoogleDriveReader(
        client_config={"client_id": "id", "client_secret": "secret"},
        token_path=str(token_path),
    )

    assert REAL_DRIVE_GET_CREDENTIALS(reader) is creds

    assert token_path.read_text(encoding="utf-8") == TOKEN_JSON
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
