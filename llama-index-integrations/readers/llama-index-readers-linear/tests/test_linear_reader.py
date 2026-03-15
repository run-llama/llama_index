"""Unit tests for LinearReader."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from llama_index_readers_linear.base import LinearReader


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MOCK_TEAMS = {
    "teams": {
        "nodes": [
            {"id": "team-123", "name": "Engineering"},
            {"id": "team-456", "name": "Design"},
        ]
    }
}

MOCK_ISSUES_PAGE_1 = {
    "team": {
        "id": "team-123",
        "name": "Engineering",
        "issues": {
            "nodes": [
                {
                    "id": "issue-001",
                    "identifier": "ENG-1",
                    "title": "Fix login bug",
                    "description": "Users cannot log in with SSO.",
                    "priority": 1,
                    "createdAt": "2024-01-01T00:00:00Z",
                    "updatedAt": "2024-01-02T00:00:00Z",
                    "state": {"name": "In Progress", "type": "started"},
                    "assignee": {"id": "user-1", "name": "Alice", "email": "alice@example.com"},
                    "labels": {"nodes": [{"name": "bug"}, {"name": "auth"}]},
                }
            ],
            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-abc"},
        },
    }
}

MOCK_ISSUES_PAGE_2 = {
    "team": {
        "id": "team-123",
        "name": "Engineering",
        "issues": {
            "nodes": [
                {
                    "id": "issue-002",
                    "identifier": "ENG-2",
                    "title": "Add dark mode",
                    "description": None,
                    "priority": 3,
                    "createdAt": "2024-01-03T00:00:00Z",
                    "updatedAt": "2024-01-04T00:00:00Z",
                    "state": {"name": "Todo", "type": "unstarted"},
                    "assignee": None,
                    "labels": {"nodes": []},
                }
            ],
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        },
    }
}

MOCK_COMMENTS = {
    "issue": {
        "comments": {
            "nodes": [
                {
                    "id": "comment-1",
                    "body": "Looking into this now.",
                    "createdAt": "2024-01-01T12:00:00Z",
                    "user": {"name": "Alice", "email": "alice@example.com"},
                },
                {
                    "id": "comment-2",
                    "body": "Fixed in PR #42.",
                    "createdAt": "2024-01-02T09:00:00Z",
                    "user": {"name": "Bob", "email": "bob@example.com"},
                },
            ]
        }
    }
}


def make_reader() -> LinearReader:
    return LinearReader(api_key="lin_api_test_key")


def mock_response(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": data}
    return resp


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


def test_init_with_explicit_key():
    reader = LinearReader(api_key="lin_api_xxx")
    assert reader.api_key == "lin_api_xxx"


def test_init_from_env(monkeypatch):
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_env")
    reader = LinearReader()
    assert reader.api_key == "lin_api_env"


def test_init_no_key_raises(monkeypatch):
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Linear API key"):
        LinearReader()


# ---------------------------------------------------------------------------
# list_teams
# ---------------------------------------------------------------------------


@patch("llama_index_readers_linear.base.requests.post")
def test_list_teams(mock_post):
    mock_post.return_value = mock_response(MOCK_TEAMS)
    reader = make_reader()
    teams = reader.list_teams()
    assert len(teams) == 2
    assert teams[0]["name"] == "Engineering"


# ---------------------------------------------------------------------------
# load_data — pagination
# ---------------------------------------------------------------------------


@patch("llama_index_readers_linear.base.requests.post")
def test_load_data_paginates(mock_post):
    mock_post.side_effect = [
        mock_response(MOCK_ISSUES_PAGE_1),
        mock_response(MOCK_ISSUES_PAGE_2),
    ]
    reader = make_reader()
    docs = reader.load_data(team_id="team-123")
    assert len(docs) == 2
    assert mock_post.call_count == 2  # one call per page


@patch("llama_index_readers_linear.base.requests.post")
def test_load_data_document_text(mock_post):
    mock_post.side_effect = [
        mock_response(MOCK_ISSUES_PAGE_1),
        mock_response(MOCK_ISSUES_PAGE_2),
    ]
    reader = make_reader()
    docs = reader.load_data(team_id="team-123")

    first_doc = docs[0]
    assert "ENG-1" in first_doc.text
    assert "Fix login bug" in first_doc.text
    assert "In Progress" in first_doc.text
    assert "Users cannot log in with SSO." in first_doc.text


@patch("llama_index_readers_linear.base.requests.post")
def test_load_data_metadata(mock_post):
    mock_post.side_effect = [
        mock_response(MOCK_ISSUES_PAGE_1),
        mock_response(MOCK_ISSUES_PAGE_2),
    ]
    reader = make_reader()
    docs = reader.load_data(team_id="team-123")

    meta = docs[0].metadata
    assert meta["identifier"] == "ENG-1"
    assert meta["status"] == "In Progress"
    assert meta["assignee_name"] == "Alice"
    assert "bug" in meta["labels"]


# ---------------------------------------------------------------------------
# load_data — no description
# ---------------------------------------------------------------------------


@patch("llama_index_readers_linear.base.requests.post")
def test_load_data_no_description(mock_post):
    """Issues without a description should not raise."""
    mock_post.side_effect = [
        mock_response(MOCK_ISSUES_PAGE_1),
        mock_response(MOCK_ISSUES_PAGE_2),
    ]
    reader = make_reader()
    docs = reader.load_data(team_id="team-123")
    # Second issue has description=None — should produce a valid doc
    assert docs[1].text is not None


# ---------------------------------------------------------------------------
# load_data — with comments
# ---------------------------------------------------------------------------


@patch("llama_index_readers_linear.base.requests.post")
def test_load_data_with_comments(mock_post):
    # Page 1 issues → page 2 issues → comments for issue-001 → comments for issue-002
    mock_post.side_effect = [
        mock_response(MOCK_ISSUES_PAGE_1),
        mock_response(MOCK_ISSUES_PAGE_2),
        mock_response(MOCK_COMMENTS),
        mock_response({"issue": {"comments": {"nodes": []}}}),
    ]
    reader = make_reader()
    docs = reader.load_data(team_id="team-123", include_comments=True)

    assert "Looking into this now." in docs[0].text
    assert "Fixed in PR #42." in docs[0].text
    # Second issue has no comments — should not contain comment section
    assert "## Comments" not in docs[1].text


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@patch("llama_index_readers_linear.base.requests.post")
def test_graphql_error_raises(mock_post):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"errors": [{"message": "Unauthorized"}]}
    mock_post.return_value = resp

    reader = make_reader()
    with pytest.raises(RuntimeError, match="Linear GraphQL error"):
        reader.list_teams()