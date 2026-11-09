from unittest.mock import AsyncMock, Mock

import pytest

from llama_index.readers.github import GitHubRepositoryIssuesReader


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (None, ["1", "2", "3", "4"]),
        ([], ["1", "2", "3", "4"]),
        ([("bug", "INCLUDE")], ["1", "2", "3"]),
        ([("wontfix", "EXCLUDE")], ["1", "4"]),
        ([("bug", "INCLUDE"), ("wontfix", "EXCLUDE")], ["1"]),
        ([("wontfix", "EXCLUDE"), ("bug", "INCLUDE")], ["1"]),
        ([("bug", "INCLUDE"), ("urgent", "INCLUDE")], ["2"]),
        ([("wontfix", "EXCLUDE"), ("feature", "EXCLUDE")], ["1"]),
    ],
)
def test_load_data_applies_all_label_filters(filters, expected):
    issues = [
        {
            "number": number,
            "title": f"Issue {number}",
            "body": "Issue body",
            "state": "open",
            "created_at": "2024-01-01T00:00:00Z",
            "url": f"https://api.github.com/repos/owner/repo/issues/{number}",
            "html_url": f"https://github.com/owner/repo/issues/{number}",
            "closed_at": None,
            "assignee": None,
            "labels": [{"name": name} for name in labels],
        }
        for number, labels in enumerate(
            [["bug"], ["bug", "urgent", "wontfix"], ["bug", "wontfix"], ["feature"]],
            start=1,
        )
    ]
    client = Mock(get_issues=AsyncMock(side_effect=[issues, []]))
    reader = GitHubRepositoryIssuesReader(client, owner="owner", repo="repo")
    label_filters = (
        [(name, getattr(reader.FilterType, mode)) for name, mode in filters]
        if filters is not None
        else None
    )

    documents = reader.load_data(labelFilters=label_filters)

    assert [doc.id_ for doc in documents] == expected
    assert client.get_issues.await_count == 2
