"""
GitHub repository pull requests reader.

Retrieves pull requests from a GitHub repository and converts them to documents.

Each pull request is converted to a document by doing the following:

    - The text of the document is the title, body, and optionally reviews/comments.
    - The doc_id of the document is the pull request number.
    - The extra_info of the document is a dictionary with the following keys:
        - number: Pull request number.
        - title: Title of the pull request.
        - state: State of the pull request (open, closed, merged).
        - author: GitHub login of the pull request author.
        - created_at: Date when the pull request was created.
        - updated_at: Date when the pull request was last updated.
        - merged_at: Date when the pull request was merged (if applicable).
        - closed_at: Date when the pull request was closed (if applicable).
        - url: API URL of the pull request.
        - html_url: Human-readable URL of the pull request.
        - merge_commit_sha: SHA of the merge commit (if merged).
        - head_sha: SHA of the head commit.
        - base_branch: Name of the base branch.
        - head_branch: Name of the head branch.
        - files_changed: Number of files changed.
        - additions: Number of lines added.
        - deletions: Number of lines deleted.
        - comments_count: Number of review comments.
        - review_comments_count: Number of review comments.
        - commits_count: Number of commits in the PR.

"""

import asyncio
import enum
import logging
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.github.pull_requests.github_client import (
    BaseGitHubPullRequestsClient,
    GitHubPullRequestsClient,
)

logger = logging.getLogger(__name__)


def print_if_verbose(verbose: bool, message: str) -> None:
    """Log message if verbose is True."""
    if verbose:
        print(message)


class GitHubRepositoryPullRequestsReader(BaseReader):
    """
    GitHub repository pull requests reader.

    Retrieves pull requests from a GitHub repository and returns a list of documents.

    Examples:
        >>> client = GitHubPullRequestsClient("your_token")
        >>> reader = GitHubRepositoryPullRequestsReader(client, "owner", "repo")
        >>> prs = reader.load_data()
        >>> print(prs)

    """

    class PRState(enum.Enum):
        """
        Pull request state.

        Used to decide what pull requests to retrieve.

        Attributes:
            - OPEN: Just open pull requests.
            - CLOSED: Just closed pull requests.
            - ALL: All pull requests, open and closed.

        """

        OPEN = "open"
        CLOSED = "closed"
        ALL = "all"

    class SortBy(enum.Enum):
        """
        Sort options for pull requests.

        Attributes:
            - CREATED: Sort by creation date.
            - UPDATED: Sort by last update date.
            - POPULARITY: Sort by number of comments.
            - LONG_RUNNING: Sort by age.

        """

        CREATED = "created"
        UPDATED = "updated"
        POPULARITY = "popularity"
        LONG_RUNNING = "long-running"

    def __init__(
        self,
        github_client: BaseGitHubPullRequestsClient,
        owner: str,
        repo: str,
        verbose: bool = False,
    ):
        """
        Initialize params.

        Args:
            - github_client (BaseGitHubPullRequestsClient): GitHub client.
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
        state: Optional[PRState] = PRState.ALL,
        sort: Optional[SortBy] = SortBy.CREATED,
        max_prs: Optional[int] = None,
        include_reviews: bool = False,
        include_comments: bool = False,
    ) -> List[Document]:
        """
        Load pull requests from a repository and converts them to documents.

        Each pull request is converted to a document with title and body as text
        and comprehensive metadata in extra_info.

        Args:
            - state (PRState): State of the pull requests to retrieve.
                Default is PRState.ALL.
            - sort (SortBy): How to sort the pull requests.
                Default is SortBy.CREATED.
            - max_prs (int, optional): Maximum number of pull requests to retrieve.
                If not provided, retrieves all pull requests.
            - include_reviews (bool): Whether to include review information
                in the document text (slower but more comprehensive).
            - include_comments (bool): Whether to include review comments
                in the document text (slower but more comprehensive).

        Returns:
            - List[Document]: List of documents representing pull requests.

        """
        documents = []
        page = 1
        prs_processed = 0

        print_if_verbose(
            self._verbose,
            f"Loading pull requests from {self._owner}/{self._repo} "
            + f"(state: {state.value}, sort: {sort.value})",
        )

        # Loop until there are no more PRs or we hit the max limit
        while True:
            if max_prs and prs_processed >= max_prs:
                print_if_verbose(
                    self._verbose, f"Reached maximum PR limit: {max_prs}"
                )
                break

            prs: List[Dict] = self._loop.run_until_complete(
                self._github_client.get_pull_requests(
                    owner=self._owner,
                    repo=self._repo,
                    state=state.value,
                    sort=sort.value,
                    page=page,
                    per_page=min(100, max_prs - prs_processed if max_prs else 100),
                )
            )

            if len(prs) == 0:
                print_if_verbose(self._verbose, "No more pull requests found, stopping")
                break

            print_if_verbose(
                self._verbose, f"Found {len(prs)} pull requests on page {page}"
            )
            page += 1

            for pr in prs:
                if max_prs and prs_processed >= max_prs:
                    break

                document = self._create_pr_document(pr, include_reviews, include_comments)
                documents.append(document)
                prs_processed += 1

            print_if_verbose(
                self._verbose, f"Processed {prs_processed} pull requests so far"
            )

        print_if_verbose(
            self._verbose, f"Total pull requests processed: {len(documents)}"
        )
        return documents

    def _create_pr_document(
        self, pr: Dict, include_reviews: bool = False, include_comments: bool = False
    ) -> Document:
        """
        Create a Document object from a pull request.

        Args:
            - pr (Dict): Pull request data from GitHub API.
            - include_reviews (bool): Whether to fetch and include review information.
            - include_comments (bool): Whether to fetch and include comment information.

        Returns:
            - Document: Document representing the pull request.
        """
        number = pr["number"]
        title = pr["title"]
        body = pr["body"] or ""

        # Build the document text
        text_parts = [f"Pull Request #{number}: {title}"]
        if body:
            text_parts.append(f"Description:\n{body}")

        # Add reviews if requested
        if include_reviews:
            reviews_text = self._get_reviews_text(number)
            if reviews_text:
                text_parts.append(f"Reviews:\n{reviews_text}")

        # Add comments if requested
        if include_comments:
            comments_text = self._get_comments_text(number)
            if comments_text:
                text_parts.append(f"Comments:\n{comments_text}")

        document = Document(
            doc_id=f"pr_{number}",
            text="\n\n".join(text_parts),
        )

        # Build comprehensive metadata
        extra_info = {
            "type": "pull_request",
            "number": number,
            "title": title,
            "state": pr["state"],
            "author": pr["user"]["login"],
            "created_at": pr["created_at"],
            "updated_at": pr["updated_at"],
            "url": pr["url"],
            "html_url": pr["html_url"],
            "draft": pr.get("draft", False),
        }

        # Add closure/merge information
        if pr["closed_at"]:
            extra_info["closed_at"] = pr["closed_at"]
        if pr["merged_at"]:
            extra_info["merged_at"] = pr["merged_at"]
            extra_info["merged"] = True
        else:
            extra_info["merged"] = False

        # Add commit information
        if pr["merge_commit_sha"]:
            extra_info["merge_commit_sha"] = pr["merge_commit_sha"]
        
        extra_info["head_sha"] = pr["head"]["sha"]
        extra_info["base_branch"] = pr["base"]["ref"]
        extra_info["head_branch"] = pr["head"]["ref"]

        # Add statistics if available
        if "changed_files" in pr:
            extra_info["files_changed"] = pr["changed_files"]
        if "additions" in pr:
            extra_info["additions"] = pr["additions"]
        if "deletions" in pr:
            extra_info["deletions"] = pr["deletions"]
        if "comments" in pr:
            extra_info["comments_count"] = pr["comments"]
        if "review_comments" in pr:
            extra_info["review_comments_count"] = pr["review_comments"]
        if "commits" in pr:
            extra_info["commits_count"] = pr["commits"]

        # Add assignees and reviewers if available
        if pr.get("assignees"):
            extra_info["assignees"] = [user["login"] for user in pr["assignees"]]
        if pr.get("requested_reviewers"):
            extra_info["requested_reviewers"] = [
                user["login"] for user in pr["requested_reviewers"]
            ]

        # Add labels if available
        if pr.get("labels"):
            extra_info["labels"] = [label["name"] for label in pr["labels"]]

        document.extra_info = extra_info
        return document

    def _get_reviews_text(self, pr_number: int) -> str:
        """
        Get reviews text for a pull request.

        Args:
            - pr_number (int): Pull request number.

        Returns:
            - str: Formatted reviews text.
        """
        try:
            reviews = self._loop.run_until_complete(
                self._github_client.get_pull_request_reviews(
                    self._owner, self._repo, pr_number
                )
            )

            if not reviews:
                return ""

            reviews_parts = []
            for review in reviews:
                reviewer = review["user"]["login"]
                state = review["state"]  # APPROVED, CHANGES_REQUESTED, COMMENTED
                body = review.get("body", "")
                
                review_text = f"Review by {reviewer} ({state})"
                if body:
                    review_text += f": {body}"
                reviews_parts.append(review_text)

            return "\n".join(reviews_parts)

        except Exception as e:
            print_if_verbose(
                self._verbose, f"Failed to get reviews for PR #{pr_number}: {e}"
            )
            return ""

    def _get_comments_text(self, pr_number: int) -> str:
        """
        Get review comments text for a pull request.

        Args:
            - pr_number (int): Pull request number.

        Returns:
            - str: Formatted comments text.
        """
        try:
            comments = self._loop.run_until_complete(
                self._github_client.get_pull_request_comments(
                    self._owner, self._repo, pr_number
                )
            )

            if not comments:
                return ""

            comments_parts = []
            for comment in comments:
                author = comment["user"]["login"]
                body = comment.get("body", "")
                
                if body:
                    comment_text = f"Comment by {author}: {body}"
                    comments_parts.append(comment_text)

            return "\n".join(comments_parts)

        except Exception as e:
            print_if_verbose(
                self._verbose, f"Failed to get comments for PR #{pr_number}: {e}"
            )
            return ""

    def get_pull_request_by_number(
        self, pr_number: int, include_reviews: bool = True, include_comments: bool = True
    ) -> Document:
        """
        Get a specific pull request by its number.

        Args:
            - pr_number (int): Number of the pull request to retrieve.
            - include_reviews (bool): Whether to include review information.
            - include_comments (bool): Whether to include comment information.

        Returns:
            - Document: Document representing the specific pull request.
        """
        print_if_verbose(
            self._verbose, f"Loading specific pull request: #{pr_number}"
        )

        pr = self._loop.run_until_complete(
            self._github_client.get_pull_request(self._owner, self._repo, pr_number)
        )

        return self._create_pr_document(pr, include_reviews, include_comments)


if __name__ == "__main__":
    """Load recent pull requests from a repository."""
    github_client = GitHubPullRequestsClient(verbose=True)

    reader = GitHubRepositoryPullRequestsReader(
        github_client=github_client,
        owner="microsoft",
        repo="vscode",
        verbose=True,
    )

    # Load last 5 PRs with reviews
    documents = reader.load_data(
        state=GitHubRepositoryPullRequestsReader.PRState.ALL,
        max_prs=5,
        include_reviews=True,
    )
    print(f"Got {len(documents)} pull request documents")

    # Show first PR
    if documents:
        first_pr = documents[0]
        print(f"\nFirst PR:")
        print(f"Number: #{first_pr.extra_info['number']}")
        print(f"Title: {first_pr.extra_info['title']}")
        print(f"Author: {first_pr.extra_info['author']}")
        print(f"State: {first_pr.extra_info['state']}")
        print(f"Created: {first_pr.extra_info['created_at']}")
