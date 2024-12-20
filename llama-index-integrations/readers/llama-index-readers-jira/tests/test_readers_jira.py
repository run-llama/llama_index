from llama_index.core.readers.base import BaseReader
from llama_index.readers.jira import JiraReader
import pytest
from unittest.mock import patch, MagicMock


def test_class():
    names_of_base_classes = [b.__name__ for b in JiraReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


@pytest.fixture(autouse=True)
def mock_jira():
    with patch("jira.JIRA") as mock_jira:
        mock_jira.return_value = MagicMock()
        yield mock_jira


@pytest.fixture()
def mock_issue():
    issue = MagicMock()
    # Setup basic issue fields
    issue.id = "TEST-123"
    issue.fields.summary = "Test Summary"
    issue.fields.description = "Test Description"
    issue.fields.issuetype.name = "Story"
    issue.fields.created = "2024-01-01"
    issue.fields.updated = "2024-01-02"
    issue.fields.labels = ["test-label"]
    issue.fields.status.name = "In Progress"
    issue.fields.project.name = "Test Project"
    issue.fields.priority.name = "High"

    # Setup assignee and reporter
    issue.fields.assignee = MagicMock()
    issue.fields.assignee.displayName = "Test Assignee"
    issue.fields.assignee.emailAddress = "assignee@test.com"
    issue.fields.reporter = MagicMock()
    issue.fields.reporter.displayName = "Test Reporter"
    issue.fields.reporter.emailAddress = "reporter@test.com"

    # Setup raw fields for parent/epic info
    issue.raw = {
        "fields": {
            "parent": {
                "key": "EPIC-1",
                "fields": {
                    "summary": "Epic Summary",
                    "status": {"description": "Epic Description"},
                },
            }
        }
    }

    issue.permalink.return_value = "https://test.atlassian.net/browse/TEST-123"
    return issue


def test_basic_auth(mock_jira):
    reader = JiraReader(
        email="test@example.com",
        api_token="test-token",
        server_url="example.atlassian.net",
    )

    mock_jira.assert_called_once_with(
        basic_auth=("test@example.com", "test-token"),
        server="https://example.atlassian.net",
    )


def test_oauth2(mock_jira):
    reader = JiraReader(Oauth2={"cloud_id": "test-cloud", "api_token": "test-token"})

    mock_jira.assert_called_once_with(
        options={
            "server": "https://api.atlassian.com/ex/jira/test-cloud",
            "headers": {"Authorization": "Bearer test-token"},
        }
    )


def test_pat_auth(mock_jira):
    reader = JiraReader(
        PATauth={
            "server_url": "https://example.atlassian.net",
            "api_token": "test-token",
        }
    )

    mock_jira.assert_called_once_with(
        options={
            "server": "https://example.atlassian.net",
            "headers": {"Authorization": "Bearer test-token"},
        }
    )


def test_load_data_basic(mock_jira, mock_issue):
    # Setup mock JIRA instance
    jira_instance = mock_jira.return_value
    jira_instance.search_issues.return_value = [mock_issue]

    reader = JiraReader(
        email="test@example.com",
        api_token="test-token",
        server_url="example.atlassian.net",
    )

    documents = reader.load_data("project = TEST")

    # Verify search_issues was called correctly
    jira_instance.search_issues.assert_called_once_with(
        "project = TEST", startAt=0, maxResults=50
    )

    # Verify returned document
    assert len(documents) == 1
    doc = documents[0]
    assert doc.doc_id == "TEST-123"
    assert doc.text == "Test Summary \n Test Description"

    # Verify extra_info
    assert doc.extra_info["assignee"] == "Test Assignee"
    assert doc.extra_info["reporter"] == "Test Reporter"
    assert doc.extra_info["epic_key"] == "EPIC-1"
    assert doc.extra_info["epic_summary"] == "Epic Summary"
    assert doc.extra_info["epic_description"] == "Epic Description"


def test_load_data_exclude_epics(mock_jira, mock_issue):
    # Modify mock issue to be an epic
    mock_issue.fields.issuetype.name = "Epic"
    jira_instance = mock_jira.return_value
    jira_instance.search_issues.return_value = [mock_issue]

    reader = JiraReader(
        email="test@example.com",
        api_token="test-token",
        server_url="example.atlassian.net",
        include_epics=False,
    )

    documents = reader.load_data("project = TEST")

    # Verify no documents returned since epics are excluded
    assert len(documents) == 0


def test_load_data_no_assignee_reporter(mock_jira, mock_issue):
    # Remove assignee and reporter
    mock_issue.fields.assignee = None
    mock_issue.fields.reporter = None
    jira_instance = mock_jira.return_value
    jira_instance.search_issues.return_value = [mock_issue]

    reader = JiraReader(
        email="test@example.com",
        api_token="test-token",
        server_url="example.atlassian.net",
    )

    documents = reader.load_data("project = TEST")

    # Verify document has empty assignee/reporter fields
    assert len(documents) == 1
    doc = documents[0]
    assert doc.extra_info["assignee"] == ""
    assert doc.extra_info["reporter"] == ""
