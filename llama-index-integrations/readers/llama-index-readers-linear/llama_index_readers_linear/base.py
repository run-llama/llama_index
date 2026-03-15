"""Linear Reader for LlamaIndex.

Fetches issues (and optionally comments) from the Linear GraphQL API
and converts them into LlamaIndex Document objects.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

ISSUES_QUERY = """
query TeamIssues($teamId: String!, $after: String) {
  team(id: $teamId) {
    id
    name
    issues(first: 50, after: $after) {
      nodes {
        id
        identifier
        title
        description
        priority
        createdAt
        updatedAt
        state {
          name
          type
        }
        assignee {
          id
          name
          email
        }
        labels {
          nodes {
            name
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""

COMMENTS_QUERY = """
query IssueComments($issueId: String!) {
  issue(id: $issueId) {
    comments {
      nodes {
        id
        body
        createdAt
        user {
          name
          email
        }
      }
    }
  }
}
"""

TEAMS_QUERY = """
query Teams {
  teams {
    nodes {
      id
      name
    }
  }
}
"""


class LinearReader(BaseReader):
    """Linear Reader.

    Reads issues from a Linear workspace using the Linear GraphQL API
    and returns them as LlamaIndex Document objects.

    Args:
        api_key (str): Linear personal API key. If not provided, reads
            from the ``LINEAR_API_KEY`` environment variable.

    Example:
        .. code-block:: python

            from llama_index_readers_linear import LinearReader

            reader = LinearReader(api_key="lin_api_xxx")

            # Load all issues from a specific team
            docs = reader.load_data(team_id="YOUR_TEAM_ID")

            # Include comments in the documents
            docs = reader.load_data(
                team_id="YOUR_TEAM_ID",
                include_comments=True,
            )
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("LINEAR_API_KEY")
        if not self.api_key:
            raise ValueError(
                "A Linear API key is required. Pass api_key= or set the "
                "LINEAR_API_KEY environment variable."
            )
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key,
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_teams(self) -> list[dict[str, str]]:
        """Return all teams visible to the authenticated user.

        Returns:
            List of dicts with ``id`` and ``name`` keys.
        """
        data = self._run_query(TEAMS_QUERY)
        return data["teams"]["nodes"]

    # ------------------------------------------------------------------
    # BaseReader interface
    # ------------------------------------------------------------------

    def load_data(
        self,
        team_id: str,
        include_comments: bool = False,
    ) -> list[Document]:
        """Load issues from a Linear team as LlamaIndex Documents.

        Args:
            team_id (str): The Linear team UUID. Use :meth:`list_teams`
                to discover available team IDs.
            include_comments (bool): When *True*, fetch all comments for
                each issue and append them to the document text.

        Returns:
            List of :class:`llama_index.core.schema.Document` objects,
            one per issue.
        """
        issues = self._fetch_all_issues(team_id)

        documents: list[Document] = []
        for issue in issues:
            text = self._issue_to_text(issue)

            if include_comments:
                comments = self._fetch_comments(issue["id"])
                if comments:
                    comment_block = "\n\n## Comments\n" + "\n\n".join(
                        f"**{c['user']['name']}** ({c['createdAt']}):\n{c['body']}"
                        for c in comments
                    )
                    text += comment_block

            metadata = self._build_metadata(issue)
            documents.append(Document(text=text, metadata=metadata))

        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_query(
        self,
        query: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL query against the Linear API."""
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(
            LINEAR_GRAPHQL_URL,
            json=payload,
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()
        if "errors" in result:
            raise RuntimeError(f"Linear GraphQL error: {result['errors']}")

        return result["data"]

    def _fetch_all_issues(self, team_id: str) -> list[dict[str, Any]]:
        """Paginate through all issues for the given team."""
        issues: list[dict[str, Any]] = []
        cursor: Optional[str] = None

        while True:
            variables: dict[str, Any] = {"teamId": team_id}
            if cursor:
                variables["after"] = cursor

            data = self._run_query(ISSUES_QUERY, variables)
            nodes = data["team"]["issues"]["nodes"]
            page_info = data["team"]["issues"]["pageInfo"]

            issues.extend(nodes)

            if not page_info["hasNextPage"]:
                break
            cursor = page_info["endCursor"]

        return issues

    def _fetch_comments(self, issue_id: str) -> list[dict[str, Any]]:
        """Fetch all comments for a single issue."""
        data = self._run_query(COMMENTS_QUERY, {"issueId": issue_id})
        return data["issue"]["comments"]["nodes"]

    @staticmethod
    def _issue_to_text(issue: dict[str, Any]) -> str:
        """Render an issue dict as human-readable text."""
        lines = [
            f"# [{issue['identifier']}] {issue['title']}",
            "",
            f"**Status:** {issue['state']['name']} ({issue['state']['type']})",
            f"**Priority:** {issue['priority']}",
            f"**Created:** {issue['createdAt']}",
            f"**Updated:** {issue['updatedAt']}",
        ]

        if issue.get("assignee"):
            lines.append(f"**Assignee:** {issue['assignee']['name']}")

        label_names = [lbl["name"] for lbl in issue.get("labels", {}).get("nodes", [])]
        if label_names:
            lines.append(f"**Labels:** {', '.join(label_names)}")

        lines.append("")

        if issue.get("description"):
            lines.append("## Description")
            lines.append(issue["description"])

        return "\n".join(lines)

    @staticmethod
    def _build_metadata(issue: dict[str, Any]) -> dict[str, Any]:
        """Build metadata dict from an issue, safe for vector store storage."""
        return {
            "issue_id": issue["id"],
            "identifier": issue["identifier"],
            "title": issue["title"],
            "status": issue["state"]["name"],
            "status_type": issue["state"]["type"],
            "priority": issue["priority"],
            "created_at": issue["createdAt"],
            "updated_at": issue["updatedAt"],
            "assignee_name": (issue["assignee"] or {}).get("name"),
            "assignee_email": (issue["assignee"] or {}).get("email"),
            "labels": [
                lbl["name"] for lbl in issue.get("labels", {}).get("nodes", [])
            ],
        }