# LlamaIndex Readers Integration: Gitlab

`pip install llama-index-readers-gitlab`

The gitlab readers package leverages [python-gitlab](https://python-gitlab.readthedocs.io/en/stable/index.html) to fetch contents from gitlab projects. It consists of two separate readers:

1. Repository Reader
2. Issues Reader

Access token or oauth token is required for private groups and projects.

## Repository Reader

This reader will read through files in a repo based on branch or commit.

```python
import gitlab
from llama_index.readers.gitlab import GitLabRepositoryReader

gitlab_client = gitlab.Gitlab("https://gitlab.com")

project_repo_reader = GitLabRepositoryReader(
    gitlab_client=gitlab_client,
    project_id=project_id,
    verbose=True,
)

docs = project_repo_reader.load_data(file_path="README.rst", ref="develop")
```

## Issues Reader

```python
import gitlab
from llama_index.readers.gitlab import GitLabIssuesReader

gitlab_client = gitlab.Gitlab("https://gitlab.com")

# load issues by project id
project_issues_reader = GitLabIssuesReader(
    gitlab_client=gitlab_client,
    project_id=project_id,
    verbose=True,
)

docs = project_issues_reader.load_data(
    state=GitLabIssuesReader.IssueState.OPEN
)

# load issues by group id
group_issues_reader = GitLabIssuesReader(
    gitlab_client=gitlab_client,
    group_id=group_id,
    verbose=True,
)

docs = group_issues_reader.load_data(state=GitLabIssuesReader.IssueState.OPEN)
```
