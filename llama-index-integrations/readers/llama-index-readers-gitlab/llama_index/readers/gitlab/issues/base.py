""" GitLab issues reader. """

from datetime import datetime
import enum
from typing import Any, List, Optional, Union

import gitlab
from gitlab.v4.objects import Issue as GitLabIssue
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class GitLabIssuesReader(BaseReader):
    """
    GitLab issues reader.
    """

    class IssueState(enum.Enum):
        """
        Issue type.

        Used to decide what issues to retrieve.

        Attributes:
            - OPEN: Issues that are open.
            - CLOSED: Issues that are closed.
            - ALL: All issues, open and closed.
        """

        OPEN = "opened"
        CLOSED = "closed"
        ALL = "all"

    class IssueType(enum.Enum):
        """
        Issue type.

        Used to decide what issues to retrieve.

        Attributes:
            - ISSUE: Issues.
            - INCIDENT: Incident.
            - TEST_CASE: Test case.
            - TASK: Task.
        """

        ISSUE = "issue"
        INCIDENT = "incident"
        TEST_CASE = "test_case"
        TASK = "task"

    class Scope(enum.Enum):
        """
        Scope.

        Used to determine the scope of the issue.

        Attributes:
            - CREATED_BY_ME: Issues created by the authenticated user.
            - ASSIGNED_TO_ME: Issues assigned to the authenticated user.
            - ALL: All issues.
        """

        CREATED_BY_ME = "created_by_me"
        ASSIGNED_TO_ME = "assigned_to_me"
        ALL = "all"

    def __init__(
        self,
        gitlab_client: gitlab.Gitlab,
        project_id: Optional[int] = None,
        group_id: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__()

        self._gl = gitlab_client
        self._project_id = project_id
        self._group_id = group_id
        self._verbose = verbose

    def _build_document_from_issue(self, issue: GitLabIssue) -> Document:
        issue_dict = issue.asdict()
        title = issue_dict["title"]
        description = issue_dict["description"]
        document = Document(
            doc_id=issue_dict["iid"],
            text=f"{title}\n{description}",
        )
        extra_info = {
            "state": issue_dict["state"],
            "labels": issue_dict["labels"],
            "created_at": issue_dict["created_at"],
            "closed_at": issue_dict["closed_at"],
            "url": issue_dict["_links"]["self"],  # API URL
            "source": issue_dict["web_url"],  # HTML URL, more convenient for humans
        }
        if issue_dict["assignee"]:
            extra_info["assignee"] = issue_dict["assignee"]["username"]
        if issue_dict["author"]:
            extra_info["author"] = issue_dict["author"]["username"]
        document.extra_info = extra_info
        return document

    def _get_project_issues(self, **kwargs):
        project = self._gl.projects.get(self._project_id)
        return project.issues.list(**kwargs)

    def _get_group_issues(self, **kwargs):
        group = self._gl.groups.get(self._group_id)
        return group.issues.list(**kwargs)

    def _to_gitlab_datetime_format(self, dt: Optional[datetime]) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else None

    def load_data(
        self,
        assignee: Optional[Union[str, int]] = None,
        author: Optional[Union[str, int]] = None,
        confidential: Optional[bool] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        iids: Optional[List[int]] = None,
        issue_type: Optional[IssueType] = None,
        labels: Optional[List[str]] = None,
        milestone: Optional[str] = None,
        non_archived: Optional[bool] = None,
        scope: Optional[Scope] = None,
        search: Optional[str] = None,
        state: Optional[IssueState] = IssueState.OPEN,
        updated_after: Optional[datetime] = None,
        updated_before: Optional[datetime] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Load group or project issues and converts them to documents. Please refer to the GitLab API documentation for the full list of parameters.

        Each issue is converted to a document by doing the following:

            - The doc_id of the document is the issue number.
            - The text of the document is the concatenation of the title and the description of the issue.
            - The extra_info of the document is a dictionary with the following keys:
                - state: State of the issue.
                - labels: List of labels of the issue.
                - created_at: Date when the issue was created.
                - closed_at: Date when the issue was closed. Only present if the issue is closed.
                - url: URL of the issue.
                - source: URL of the issue. More convenient for humans.
                - assignee: username of the user assigned to the issue. Only present if the issue is assigned.

        Args:
            - assignee: Username or ID of the user assigned to the issue.
            - author: Username or ID of the user that created the issue.
            - confidential: Filter confidential issues.
            - created_after: Filter issues created after the specified date.
            - created_before: Filter issues created before the specified date.
            - iids: Return only the issues having the given iid.
            - issue_type: Filter issues by type.
            - labels: List of label names, issues must have all labels to be returned.
            - milestone: The milestone title.
            - non_archived: Return issues from non archived projects.
            - scope: Return issues for the given scope.
            - search: Search issues against their title and description.
            - state: State of the issues to retrieve.
            - updated_after: Filter issues updated after the specified date.
            - updated_before: Filter issues updated before the specified date.


        Returns:
            List[Document]: List of documents.
        """
        to_gitlab_datetime_format = self._to_gitlab_datetime_format
        params = {
            "confidential": confidential,
            "created_after": to_gitlab_datetime_format(created_after),
            "created_before": to_gitlab_datetime_format(created_before),
            "iids": iids,
            "issue_type": issue_type.value if issue_type else None,
            "labels": labels,
            "milestone": milestone,
            "non_archived": non_archived,
            "scope": scope.value if scope else None,
            "search": search,
            "state": state.value if state else None,
            "updated_after": to_gitlab_datetime_format(updated_after),
            "updated_before": to_gitlab_datetime_format(updated_before),
        }

        if isinstance(assignee, str):
            params["assignee_username"] = assignee
        elif isinstance(assignee, int):
            params["assignee_id"] = assignee

        if isinstance(author, str):
            params["author_username"] = author
        elif isinstance(author, int):
            params["author_id"] = author

        filtered_params = {k: v for k, v in params.items() if v is not None}

        filtered_params.update(kwargs)

        issues = []

        if self._project_id:
            issues = self._get_project_issues(**filtered_params)
        if self._group_id:
            issues = self._get_group_issues(**filtered_params)

        return [self._build_document_from_issue(issue) for issue in issues]


if __name__ == "__main__":
    reader = GitLabIssuesReader(
        gitlab_client=gitlab.Gitlab("https://gitlab.com"),
        project_id=48082128,
        group_id=10707808,
        verbose=True,
    )
    docs = reader.load_data(state=GitLabIssuesReader.IssueState.OPEN)
    print(docs)
