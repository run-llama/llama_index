"""Jira tool spec."""

from typing import Optional, Union
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class JiraToolSpec(BaseToolSpec):
    """Atlassian Jira Tool Spec."""

    spec_functions = [
        "jira_issues_query",
        "jira_issue_query",
        "jira_comments_query",
        "jira_all_projects",
    ]

    def __init__(
        self,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        server_url: Optional[str] = None,
    ) -> None:
        """Initialize the Atlassian Jira tool spec."""
        from jira import JIRA

        if email and api_token and server_url:
            self.jira = JIRA(
                basic_auth=(email, api_token),
                server=server_url,
            )
        else:
            raise Exception("Error: Please provide Jira api credentials to continue")

    def jira_all_projects(self) -> dict:
        """
        Retrieve all projects from the Atlassian Jira account.

        This method fetches a list of projects from Jira and returns them in a structured
        format, including the project ID, key, and name. If an error occurs during
        retrieval, an error message is returned.

        Returns:
            dict: A dictionary containing:
                - 'error' (bool): Indicates whether the request was successful.
                - 'message' (str): A description of the result.
                - 'projects' (list, optional): A list of projects with their details
                  (ID, key, name) if retrieval is successful.

        """
        try:
            projects = self.jira.projects()

            if projects:
                return {
                    "error": False,
                    "message": "All Projects from the account",
                    "projects": [
                        {"id": project.id, "key": project.key, "name": project.name}
                        for project in projects
                    ],
                }
        except Exception:
            pass

        return {"error": True, "message": "Unable to fetch projects"}

    def jira_comments_query(
        self, issue_key: str, author_email: Optional[str] = None
    ) -> dict:
        """
        Retrieve all comments for a given Jira issue, optionally filtering by the author's email.

        This function fetches comments from a specified Jira issue and returns them as a structured
        JSON response. If an `author_email` is provided, only comments from that specific author
        will be included.

        Args:
            issue_key (str): The Jira issue key for which to retrieve comments.
            author_email (str, Optional): filters comments by the author's email.

        Returns:
            dict: A dictionary containing:
                - 'error' (bool): Indicates whether the request was successful.
                - 'message' (str): A descriptive message about the result.
                - 'comments' (list, optional): A list of comments, where each comment includes:
                    - 'id' (str): The unique identifier of the comment.
                    - 'author' (str): The display name of the comment's author.
                    - 'author_email' (str): The author's email address.
                    - 'body' (str): The content of the comment.
                    - 'created_at' (str): The timestamp when the comment was created.
                    - 'updated_at' (str): The timestamp when the comment was last updated.

        """
        error = False

        try:
            issue = self.jira.issue(issue_key)

            all_comments = list(issue.fields.comment.comments)
            filtered_results = []

            for comment in all_comments:
                if (
                    author_email is not None
                    and author_email not in comment.author.emailAddress
                ):
                    continue

                filtered_results.append(
                    {
                        "id": comment.id,
                        "author": comment.author.displayName,
                        "author_email": comment.author.emailAddress,
                        "body": comment.body,
                        "created_at": comment.created,
                        "updated_at": comment.updated,
                    }
                )

            message = f'All the comments in the issue key "{issue_key}"'
        except Exception:
            error = True
            message = "Unable to fetch comments due to some error"

        response = {"error": error, "message": message}

        if error is False:
            response["comments"] = filtered_results

        return response

    def jira_issue_query(
        self, issue_key: str, just_payload: bool = False
    ) -> Union[None, dict]:
        """
        Retrieves detailed information about a specific Jira issue.

        This method fetches issue details such as summary, description, type, project, priority, status,
        reporter, assignee, labels, and timestamps. The response structure can be adjusted using the
        `just_payload` flag.

        Args:
            issue_key (str): The unique key or ticket number of the Jira issue.
            just_payload (bool, optional): If True, returns only the issue payload without the response
                                           metadata. Defaults to False.

        Returns:
            Union[None, dict]: A dictionary containing issue details if found, or an error response if the issue
                               cannot be retrieved.

        Example:
            > jira_client.load_issue("JIRA-123", just_payload=True)
            {
                'key': 'JIRA-123',
                'summary': 'Fix login bug',
                'description': 'Users unable to log in under certain conditions...',
                'type': 'Bug',
                'project_name': 'Web App',
                'priority': 'High',
                'status': 'In Progress',
                'reporter': 'John Doe',
                'reporter_email': 'john.doe@example.com',
                'labels': ['authentication', 'urgent'],
                'created_at': '2024-02-01T10:15:30.000Z',
                'updated_at': '2024-02-02T12:20:45.000Z',
                'assignee': 'Jane Smith',
                'assignee_email': 'jane.smith@example.com'
            }

        """
        error = False
        try:
            issue = self.jira.issue(issue_key)

            payload = {
                "key": issue.key,
                "summary": issue.fields.summary,
                "description": issue.fields.description,
                "type": issue.fields.issuetype.name,
                "project_name": issue.fields.project.name,
                "priority": issue.fields.priority.name,
                "status": issue.fields.status.name,
                "reporter": issue.fields.reporter.displayName
                if issue.fields.reporter
                else None,
                "reporter_email": issue.fields.reporter.emailAddress
                if issue.fields.reporter
                else None,
                "labels": issue.fields.labels,
                "created_at": issue.fields.created,
                "updated_at": issue.fields.updated,
                "assignee": issue.fields.assignee.displayName
                if issue.fields.assignee
                else None,
                "assignee_email": issue.fields.assignee.emailAddress
                if issue.fields.assignee
                else None,
            }

            message = f"Details of the issue: {issue.key}"

        except Exception:
            error = True
            message = "Unable to fetch issue due to some error"

        if error is False and just_payload:
            return payload

        response = {"error": error, "message": message}

        if error is False:
            response["result"] = payload

        return response

    def jira_issues_query(self, keyword: str, max_results: int = 10) -> dict:
        """
        Search for Jira issues containing a specific keyword.

        This function searches for Jira issues where the specified `keyword` appears in the summary, description, or comments.
        The results are sorted by creation date in descending order.

        Args:
            keyword (str): The keyword to search for within issue summaries, descriptions, or comments.
            max_results (int, optional): The maximum number of issues to return. Defaults to 10. If set higher than 100, it will be limited to 100.

        Returns:
            dict: A dictionary with the following structure:
                - 'error' (bool): Indicates if an error occurred during the fetch operation.
                - 'message' (str): Describes the outcome of the operation.
                - 'results' (list, optional): A list of issues matching the search criteria, present only if no error occurred.

        """
        error = False

        max_results = min(max_results, 100)

        # if custom_query is not None:
        #     jql = custom_query
        # else:
        jql = f'summary ~ "{keyword}" or description ~ "{keyword}" or text ~ "{keyword}" order by created desc'

        try:
            issues = [
                self.jira_issue_query(issue.key, just_payload=True)
                for issue in self.jira.search_issues(jql, maxResults=max_results)
            ]

            message = "All the issues with specific matching conditions"
        except Exception:
            error = True
            message = "Unable to fetch issue due to some error"

        response = {"error": error, "message": message}

        if error is False:
            response["results"] = issues

        return response
