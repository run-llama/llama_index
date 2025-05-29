import pytest
from llama_index.readers.github import GithubRepositoryReader


class MockGithubClient:
    pass


@pytest.fixture()
def github_reader():
    return GithubRepositoryReader(
        github_client=MockGithubClient(), owner="owner", repo="repo"
    )


@pytest.mark.parametrize(
    ("blob_url", "expected_base_url"),
    [
        ("https://github.com/owner/repo/blob/main/file.py", "https://github.com/"),
        (
            "https://github-enterprise.com/owner/repo/blob/main/file.py",
            "https://github-enterprise.com/",
        ),
        (
            "https://custom-domain.com/owner/repo/blob/main/file.py",
            "https://custom-domain.com/",
        ),
        (
            "https://subdomain.github.com/owner/repo/blob/main/file.py",
            "https://subdomain.github.com/",
        ),
        (
            "https://something.org/owner/repo/blob/main/file.py",
            "https://github.com/",
        ),
        ("", "https://github.com/"),
    ],
)
def test_get_base_url(github_reader, blob_url, expected_base_url):
    base_url = github_reader._get_base_url(blob_url)
    assert (
        base_url == expected_base_url
    ), f"Expected {expected_base_url}, but got {base_url}"
