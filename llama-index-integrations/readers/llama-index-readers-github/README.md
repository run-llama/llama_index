# LlamaIndex Readers Integration: Github

`pip install llama-index-readers-github`

The github readers package consists of three separate readers:

1. Repository Reader
2. Issues Reader
3. Collaborators Reader

All three readers will require a personal access token (which you can generate under your account settings).

## Repository Reader

This reader will read through a repo, with options to specifically filter directories and file extensions.

```python
from llama_index.readers.github import GithubRepositoryReader, GithubClient

client = github_client = GithubClient(github_token=github_token, verbose=False)

reader = GithubRepositoryReader(
    github_client=github_client,
    owner="run-llama",
    repo="llama_index",
    use_parser=False,
    verbose=True,
    filter_directories=(
        ["docs"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_file_extensions=(
        [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            "json",
            ".ipynb",
        ],
        GithubRepositoryReader.FilterType.EXCLUDE,
    ),
)

documents = reader.load_data(branch="main")
```

## Issues Reader

```python
from llama_index.readers.github import (
    GitHubRepositoryIssuesReader,
    GitHubIssuesClient,
)

github_client = GitHubIssuesClient(github_token=github_token, verbose=True)

reader = GitHubRepositoryIssuesReader(
    github_client=github_client,
    owner="moncho",
    repo="dry",
    verbose=True,
)

documents = reader.load_data(
    state=GitHubRepositoryIssuesReader.IssueState.ALL,
    labelFilters=[("bug", GitHubRepositoryIssuesReader.FilterType.INCLUDE)],
)
```

## Collaborators Reader

```python
from llama_index.readers.github import (
    GitHubRepositoryCollaboratorsReader,
    GitHubCollaboratorsClient,
)

github_client = GitHubCollaboratorsClient(
    github_token=github_token, verbose=True
)

reader = GitHubRepositoryCollaboratorsReader(
    github_client=github_client,
    owner="moncho",
    repo="dry",
    verbose=True,
)

documents = reader.load_data()
```
