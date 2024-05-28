import unittest
from unittest.mock import MagicMock, AsyncMock
from llama_index.readers.github.issues.base import GitHubRepositoryIssuesReader


COMMENTS = {
    1: [{"body": "Comment 1.1"}],
    2: [],
    3: [{"body": "Comment 3.1"}, {"body": "Comment 3.2"}],
}

ISSUES = [
    {
        "number": 1,
        "title": "Issue 1",
        "body": "Body 1",
        "state": "open",
        "created_at": "2022-01-01T00:00:00Z",
        "closed_at": None,
        "url": "",
        "html_url": "",
        "assignee": {"login": "assignee"},
        "labels": [{"name": "label1"}, {"name": "label2"}],
        "comments": len(COMMENTS[1]),
    },
    {
        "number": 2,
        "title": "Issue 2",
        "body": "Body 2",
        "state": "closed",
        "created_at": "2022-01-01T00:00:00Z",
        "closed_at": "2022-01-02T00:00:00Z",
        "url": "",
        "html_url": "",
        "assignee": None,
        "labels": None,
        "comments": len(COMMENTS[2]),
    },
    {
        "number": 3,
        "title": "Issue 3",
        "body": "Body 3",
        "state": "open",
        "created_at": "2022-01-01T00:00:00Z",
        "closed_at": None,
        "url": "",
        "html_url": "",
        "assignee": None,
        "labels": None,
        "comments": len(COMMENTS[3]),
    },
]


async def mock_get_issues(
    owner: str,
    repo: str,
    state: str = "open",
    page: int = 1,
):
    if page > 1:
        return {}
    return ISSUES


async def mock_get_comments(
    owner: str,
    repo: str,
    issue_number: int,
    page: int = 1,
):
    if page > 1:
        return []
    return COMMENTS[issue_number]


class TestGitHubRepositoryIssuesReader(unittest.TestCase):
    def setUp(self):
        # Mock the GitHub client
        self.github_client = AsyncMock()
        self.github_client.get_issues.side_effect = mock_get_issues
        self.github_client.get_comments.side_effect = mock_get_comments

        # Create an instance of GitHubRepositoryIssuesReader
        self.reader = GitHubRepositoryIssuesReader(
            github_client=self.github_client,
            owner="foo",
            repo="bar",
            include_comments=True,
            verbose=False,
        )

    def test_load_data(self):
        documents = self.reader.load_data()
        got = [doc.text for doc in documents]

        self.assertEqual(
            got,
            [
                "Issue 1\nBody 1\nComment 1.1",
                "Issue 2\nBody 2",
                "Issue 3\nBody 3\nComment 3.1\nComment 3.2",
            ],
        )


if __name__ == "__main__":
    unittest.main()
