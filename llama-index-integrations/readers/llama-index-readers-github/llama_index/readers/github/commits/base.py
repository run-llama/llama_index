"""
GitHub repository commits reader.

Retrieves the commit history of a GitHub repository and converts them to documents.

Each commit is converted to a document by doing the following:

    - The text of the document is the commit message and optionally diff information.
    - The doc_id of the document is the commit SHA.
    - The extra_info of the document is a dictionary with the following keys:
        - sha: Full SHA hash of the commit.
        - author_name: Name of the commit author.
        - author_email: Email of the commit author.
        - author_date: Date when the commit was authored.
        - committer_name: Name of the committer.
        - committer_email: Email of the committer.
        - committer_date: Date when the commit was committed.
        - url: URL of the commit.
        - html_url: Human-readable URL of the commit.
        - files_changed: List of files changed in the commit.
        - additions: Number of lines added.
        - deletions: Number of lines deleted.
        - total_changes: Total number of changes.

"""

import asyncio
import enum
import logging
from datetime import datetime
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.github.commits.github_client import (
    BaseGitHubCommitsClient,
    GitHubCommitsClient,
)

logger = logging.getLogger(__name__)


def print_if_verbose(verbose: bool, message: str) -> None:
    """Log message if verbose is True."""
    if verbose:
        print(message)


class GitHubRepositoryCommitsReader(BaseReader):
    """
    GitHub repository commits reader.

    Retrieves the commit history of a GitHub repository and returns a list of documents.

    Examples:
        >>> client = GitHubCommitsClient("your_token")
        >>> reader = GitHubRepositoryCommitsReader(client, "owner", "repo")
        >>> commits = reader.load_data()
        >>> print(commits)

    """

    class CommitDateFilter(enum.Enum):
        """
        Date filter options for commits.

        Used to filter commits by date range.
        """

        LAST_WEEK = "last_week"
        LAST_MONTH = "last_month"
        LAST_YEAR = "last_year"
        CUSTOM = "custom"

    def __init__(
        self,
        github_client: BaseGitHubCommitsClient,
        owner: str,
        repo: str,
        verbose: bool = False,
    ):
        """
        Initialize params.

        Args:
            - github_client (BaseGitHubCommitsClient): GitHub client.
            - owner (str): Owner of the repository.
            - repo (str): Name of the repository.
            - verbose (bool): Whether to print verbose messages.

        Raises:
            - `ValueError`: If the github_token is not provided and
                the GITHUB_TOKEN environment variable is not set.

        """
        super().__init__()

        self._owner = owner
        self._repo = repo
        self._verbose = verbose

        # Set up the event loop
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # If there is no running loop, create a new one
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self._github_client = github_client

    def load_data(
        self,
        branch_or_sha: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        path: Optional[str] = None,
        max_commits: Optional[int] = None,
        include_diff_info: bool = False,
    ) -> List[Document]:
        """
        Load commits from a repository and converts them to documents.

        Each commit is converted to a document with the commit message as text
        and comprehensive metadata in extra_info.

        Args:
            - branch_or_sha (str, optional): Branch name or commit SHA to start from.
                If not provided, uses the default branch.
            - since (str, optional): Only commits after this date (ISO 8601 format).
            - until (str, optional): Only commits before this date (ISO 8601 format).
            - path (str, optional): Only commits containing changes to this file path.
            - max_commits (int, optional): Maximum number of commits to retrieve.
                If not provided, retrieves all commits.
            - include_diff_info (bool): Whether to include detailed diff information
                for each commit (slower but more comprehensive).

        Returns:
            - List[Document]: List of documents representing commits.

        """
        documents = []
        page = 1
        commits_processed = 0

        print_if_verbose(
            self._verbose,
            f"Loading commits from {self._owner}/{self._repo}"
            + (f" (branch/sha: {branch_or_sha})" if branch_or_sha else ""),
        )

        # Loop until there are no more commits or we hit the max limit
        while True:
            if max_commits and commits_processed >= max_commits:
                print_if_verbose(
                    self._verbose, f"Reached maximum commit limit: {max_commits}"
                )
                break

            commits: List[Dict] = self._loop.run_until_complete(
                self._github_client.get_commits(
                    owner=self._owner,
                    repo=self._repo,
                    sha=branch_or_sha,
                    path=path,
                    since=since,
                    until=until,
                    page=page,
                    per_page=min(100, max_commits - commits_processed if max_commits else 100),
                )
            )

            if len(commits) == 0:
                print_if_verbose(self._verbose, "No more commits found, stopping")
                break

            print_if_verbose(
                self._verbose, f"Found {len(commits)} commits on page {page}"
            )
            page += 1

            for commit in commits:
                if max_commits and commits_processed >= max_commits:
                    break

                document = self._create_commit_document(commit, include_diff_info)
                documents.append(document)
                commits_processed += 1

            print_if_verbose(
                self._verbose, f"Processed {commits_processed} commits so far"
            )

        print_if_verbose(
            self._verbose, f"Total commits processed: {len(documents)}"
        )
        return documents

    def _create_commit_document(
        self, commit: Dict, include_diff_info: bool = False
    ) -> Document:
        """
        Create a Document object from a commit.

        Args:
            - commit (Dict): Commit data from GitHub API.
            - include_diff_info (bool): Whether to fetch detailed diff information.

        Returns:
            - Document: Document representing the commit.
        """
        commit_data = commit["commit"]
        sha = commit["sha"]
        message = commit_data["message"]

        # Extract author and committer information
        author = commit_data["author"]
        committer = commit_data["committer"]

        # Build the document text (commit message + optional diff info)
        text_parts = [f"Commit: {sha[:8]}", f"Message: {message}"]

        # Get detailed commit info if requested
        if include_diff_info:
            try:
                detailed_commit = self._loop.run_until_complete(
                    self._github_client.get_commit(self._owner, self._repo, sha)
                )
                files_info = self._extract_files_info(detailed_commit)
                if files_info:
                    text_parts.append(f"Files changed: {files_info}")
            except Exception as e:
                print_if_verbose(
                    self._verbose, f"Failed to get detailed info for {sha}: {e}"
                )

        document = Document(
            doc_id=f"commit_{sha}",
            text="\n\n".join(text_parts),
        )

        # Build comprehensive metadata
        extra_info = {
            "type": "commit",
            "sha": sha,
            "short_sha": sha[:8],
            "message": message,
            "author_name": author["name"],
            "author_email": author["email"],
            "author_date": author["date"],
            "committer_name": committer["name"],
            "committer_email": committer["email"],
            "committer_date": committer["date"],
            "url": commit["url"],
            "html_url": commit["html_url"],
        }

        # Add GitHub user information if available
        if commit.get("author"):
            extra_info["author_github_login"] = commit["author"]["login"]
        if commit.get("committer"):
            extra_info["committer_github_login"] = commit["committer"]["login"]

        # Add diff statistics if available
        if "stats" in commit:
            stats = commit["stats"]
            extra_info.update({
                "additions": stats.get("additions", 0),
                "deletions": stats.get("deletions", 0),
                "total_changes": stats.get("total", 0),
            })

        # Add files information if available
        if "files" in commit:
            extra_info["files_changed"] = [f["filename"] for f in commit["files"]]
            extra_info["files_count"] = len(commit["files"])

        document.extra_info = extra_info
        return document

    def _extract_files_info(self, detailed_commit: Dict) -> str:
        """
        Extract a summary of files changed from detailed commit info.

        Args:
            - detailed_commit (Dict): Detailed commit information from GitHub API.

        Returns:
            - str: Summary of files changed.
        """
        if "files" not in detailed_commit:
            return ""

        files = detailed_commit["files"]
        if not files:
            return ""

        file_summaries = []
        for file_info in files[:10]:  # Limit to first 10 files
            filename = file_info["filename"]
            status = file_info["status"]  # added, removed, modified, renamed
            additions = file_info.get("additions", 0)
            deletions = file_info.get("deletions", 0)

            summary = f"{filename} ({status}"
            if additions or deletions:
                summary += f", +{additions}/-{deletions}"
            summary += ")"
            file_summaries.append(summary)

        result = ", ".join(file_summaries)
        if len(files) > 10:
            result += f" ... and {len(files) - 10} more files"

        return result

    def get_commit_by_sha(self, commit_sha: str, include_diff_info: bool = True) -> Document:
        """
        Get a specific commit by its SHA.

        Args:
            - commit_sha (str): SHA of the commit to retrieve.
            - include_diff_info (bool): Whether to include detailed diff information.

        Returns:
            - Document: Document representing the specific commit.
        """
        print_if_verbose(
            self._verbose, f"Loading specific commit: {commit_sha}"
        )

        commit = self._loop.run_until_complete(
            self._github_client.get_commit(self._owner, self._repo, commit_sha)
        )

        return self._create_commit_document(commit, include_diff_info)


if __name__ == "__main__":
    """Load recent commits from a repository."""
    github_client = GitHubCommitsClient(verbose=True)

    reader = GitHubRepositoryCommitsReader(
        github_client=github_client,
        owner="octocat",
        repo="Hello-World",
        verbose=True,
    )

    # Load last 10 commits
    documents = reader.load_data(max_commits=10, include_diff_info=True)
    print(f"Got {len(documents)} commit documents")

    # Show first commit
    if documents:
        first_commit = documents[0]
        print(f"\nFirst commit:")
        print(f"SHA: {first_commit.extra_info['sha']}")
        print(f"Author: {first_commit.extra_info['author_name']}")
        print(f"Date: {first_commit.extra_info['author_date']}")
        print(f"Message: {first_commit.extra_info['message']}")
