import pytest
from unittest.mock import MagicMock
from llama_index.readers.gitlab import GitLabIssuesReader
import gitlab

# import so that pants sees it as a dependency
import pytest_mock  # noqa


@pytest.fixture()
def gitlab_client_mock():
    return MagicMock(spec=gitlab.Gitlab)


@pytest.fixture()
def gitlab_issues_reader(gitlab_client_mock):
    return GitLabIssuesReader(
        gitlab_client=gitlab_client_mock, project_id=123, verbose=True
    )


def test_load_data_project_issues(gitlab_issues_reader, mocker):
    mock_issue = MagicMock()
    mock_issue.asdict.return_value = {
        "iid": 1,
        "title": "Issue title",
        "description": "Issue description",
        "state": "opened",
        "labels": ["bug"],
        "created_at": "2023-01-01T00:00:00",
        "closed_at": None,
        "_links": {"self": "http://api.url"},
        "web_url": "http://web.url",
        "assignee": {"username": "assignee_user"},
        "author": {"username": "author_user"},
    }

    mocker.patch.object(
        gitlab_issues_reader, "_get_project_issues", return_value=[mock_issue]
    )

    documents = gitlab_issues_reader.load_data()

    assert len(documents) == 1
    assert documents[0].doc_id == "1"
    assert documents[0].text == "Issue title\nIssue description"
    assert documents[0].extra_info["state"] == "opened"
    assert documents[0].extra_info["labels"] == ["bug"]
    assert documents[0].extra_info["created_at"] == "2023-01-01T00:00:00"
    assert documents[0].extra_info["url"] == "http://api.url"
    assert documents[0].extra_info["source"] == "http://web.url"
    assert documents[0].extra_info["assignee"] == "assignee_user"
    assert documents[0].extra_info["author"] == "author_user"


def test_load_data_group_issues(gitlab_issues_reader, mocker):
    gitlab_issues_reader._project_id = None
    gitlab_issues_reader._group_id = 456

    mock_issue = MagicMock()
    mock_issue.asdict.return_value = {
        "iid": 2,
        "title": "Group issue title",
        "description": "Group issue description",
        "state": "closed",
        "labels": ["enhancement"],
        "created_at": "2023-02-01T00:00:00",
        "closed_at": "2023-02-02T00:00:00",
        "_links": {"self": "http://api.group.url"},
        "web_url": "http://web.group.url",
        "assignee": {"username": "group_assignee_user"},
        "author": {"username": "group_author_user"},
    }

    mocker.patch.object(
        gitlab_issues_reader, "_get_group_issues", return_value=[mock_issue]
    )

    documents = gitlab_issues_reader.load_data()

    assert len(documents) == 1
    assert documents[0].doc_id == "2"
    assert documents[0].text == "Group issue title\nGroup issue description"
    assert documents[0].extra_info["state"] == "closed"
    assert documents[0].extra_info["labels"] == ["enhancement"]
    assert documents[0].extra_info["created_at"] == "2023-02-01T00:00:00"
    assert documents[0].extra_info["closed_at"] == "2023-02-02T00:00:00"
    assert documents[0].extra_info["url"] == "http://api.group.url"
    assert documents[0].extra_info["source"] == "http://web.group.url"
    assert documents[0].extra_info["assignee"] == "group_assignee_user"
    assert documents[0].extra_info["author"] == "group_author_user"
