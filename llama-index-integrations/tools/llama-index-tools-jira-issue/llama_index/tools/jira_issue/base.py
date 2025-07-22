"""Jira tool spec."""

import os
from typing import Optional, Dict, Any, Literal
from jira import JIRA
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class JiraIssueToolSpec(BaseToolSpec):
    """Atlassian Jira Issue Tool Spec."""

    spec_functions = [
        "search_issues",
        "create_issue",
        "add_comment_to_issue",
        "update_issue_summary",
        "update_issue_assignee",
        "update_issue_status",
        "update_issue_due_date",
        "delete_issue",
    ]

    def __init__(
        self,
        email: str = os.environ.get("JIRA_ACCOUNT_EMAIL", ""),
        api_key: Optional[str] = os.environ.get("JIRA_API_KEY", ""),
        server_url: Optional[str] = os.environ.get("JIRA_SERVER_URL", ""),
    ) -> None:
        if email and api_key and server_url:
            self.jira = JIRA(
                basic_auth=(email, api_key),
                server=server_url,
            )
        else:
            raise Exception("Please provide Jira credentials to continue.")

    def search_issues(self, jql_str: str) -> Dict[str, Any]:
        """
        Search for JIRA issues using JQL.

        Args:
            jql_str (str): JQL query string to search for issues.

        Returns:
            Dict[str, Any]: A dictionary containing the search results or error message.

        """
        try:
            issues = self.jira.search_issues(jql_str)

            if issues:
                return {
                    "error": False,
                    "message": "Issues found",
                    "issues": [
                        {
                            "key": issue.key,
                            "summary": issue.fields.summary,
                            "status": issue.fields.status.name,
                            "assignee": issue.fields.assignee.displayName
                            if issue.fields.assignee
                            else None,
                        }
                        for issue in issues
                    ],
                }
            else:
                return {
                    "error": True,
                    "message": "No issues found.",
                }
        except Exception as e:
            return {
                "error": True,
                "message": f"Failed to search issues: {e!s}",
            }

    def create_issue(
        self,
        project_key: str = "KAN",
        summary: str = "New Issue",
        description: Optional[str] = None,
        issue_type: Literal["Task", "Bug", "Epic"] = "Task",
    ) -> Dict[str, Any]:
        """
        Create a new JIRA issue.

        Args:
            project_key (str): The key of the project to create the issue in (default is "KAN").
            summary (str): The summary of the new issue (default is "New Issue").
            description (Optional[str]): The description of the new issue.
            issue_type (str): The type of the issue to create, can be "Task", "Bug", or "Epic" (default is "Task").

        Returns:
            Dict[str, Any]: A dictionary indicating success or failure of the operation.

        """
        try:
            new_issue = self.jira.create_issue(
                project=project_key,
                summary=summary,
                description=description,
                issuetype={"name": issue_type},
            )
            return {
                "error": False,
                "message": f"Issue {new_issue.key} created successfully.",
                "issue_key": new_issue.key,
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"Failed to create new issue: {e!s}",
            }

    def add_comment_to_issue(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """
        Add a comment to a JIRA issue.

        Args:
            issue_key (str): The key of the JIRA issue to comment on.
            comment (str): The comment text to add.

        Returns:
            Dict[str, Any]: A dictionary indicating success or failure of the operation.

        """
        try:
            issue = self.jira.issue(issue_key)
            self.jira.add_comment(issue, comment)
            return {"error": False, "message": f"Comment added to issue {issue_key}."}
        except Exception as e:
            return {
                "error": True,
                "message": f"Failed to add comment to issue {issue_key}: {e!s}",
            }

    def update_issue_summary(
        self, issue_key: str, new_summary: str, notify: bool = False
    ) -> Dict[str, Any]:
        """
        Update the summary of a JIRA issue.

        Args:
            issue_key (str): The key of the JIRA issue to update.
            new_summary (str): The new summary text for the issue.
            notify (bool): Whether to email watchers of the issue about the update.

        Returns:
            Dict[str, Any]: A dictionary indicating success or failure of the operation.

        """
        try:
            issue = self.jira.issue(issue_key)
            issue.update(summary=new_summary, notify=notify)
            return {"error": False, "message": f"Issue {issue_key} summary updated."}
        except Exception as e:
            return {
                "error": True,
                "message": f"Failed to update issue {issue_key}: {e!s}",
            }

    def update_issue_assignee(self, issue_key, assignee_full_name):
        """
        Update the assignee of the Jira issue using the assignee's full name.

        Args:
            issue_key (str): The key of the Jira issue to update.
            assignee_full_name (str): The full name of the user to assign the issue to.

        Returns:
            Dict[str, Any]: A dictionary indicating success or failure of the operation.

        """
        try:
            # Search for users by display name
            users = self.jira.search_users(query=assignee_full_name)

            # Find exact match for the full name
            target_user = None
            for user in users:
                if user.displayName.lower() == assignee_full_name.lower():
                    target_user = user
                    break

            if not target_user:
                return {
                    "error": True,
                    "message": f"User with full name '{assignee_full_name}' not found",
                }

            # Get the issue
            issue = self.jira.issue(issue_key)
            issue.update(assignee={"accountId": target_user.accountId})

            return {
                "error": False,
                "message": f"Issue {issue_key} successfully assigned to {assignee_full_name}",
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"An error occurred while updating the assignee: {e!s}",
            }

    def update_issue_status(
        self, issue_key: str, new_status: Literal["To Do", "In Progress", "Done"]
    ) -> Dict[str, Any]:
        """
        Update the status of a JIRA issue.

        Args:
            issue_key (str): The key of the JIRA issue to update.
            new_status (str): The new status to set for the issue.

        Returns:
            Dict[str, Any]: A dictionary indicating success or failure of the operation.

        """
        try:
            issue = self.jira.issue(issue_key)
            transitions = self.jira.transitions(issue)
            transition_id = next(
                (t["id"] for t in transitions if t["name"] == new_status), None
            )

            if transition_id:
                self.jira.transition_issue(issue, transition_id)
                return {
                    "error": False,
                    "message": f"Issue {issue_key} status updated to {new_status}.",
                }
            else:
                available_statuses = [t["name"] for t in transitions]
                return {
                    "error": True,
                    "message": f"Status '{new_status}' not available for issue {issue_key}. Available transitions: {available_statuses}",
                }
        except Exception as e:
            return {
                "error": True,
                "message": f"Failed to update status for issue {issue_key}: {e!s}",
            }

    def update_issue_due_date(
        self, issue_key: str, due_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update the due date of a JIRA issue.

        Args:
            issue_key (str): The key of the JIRA issue to update.
            due_date (Optional[str]): The new due date in 'YYYY-MM-DD' format.

        Returns:
            Dict[str, Any]: A dictionary indicating success or failure of the operation.

        """
        if due_date:
            try:
                from datetime import datetime

                datetime.strptime(due_date, "%Y-%m-%d")
            except ValueError:
                return {
                    "error": True,
                    "message": "Invalid date format. Use YYYY-MM-DD.",
                }
        try:
            issue = self.jira.issue(issue_key)
            issue.update(duedate=due_date)
            return {
                "error": False,
                "message": f"Issue {issue_key} due date {'updated' if due_date else 'cleared'}.",
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"Failed to update due date for issue {issue_key}: {e!s}",
            }

    def delete_issue(self, issue_key: str) -> Dict[str, Any]:
        """
        Delete a JIRA issue.

        Args:
            issue_key (str): The key of the JIRA issue to delete.

        Returns:
            Dict[str, Any]: A dictionary indicating success or failure of the operation.

        """
        try:
            issue = self.jira.issue(issue_key)
            issue.delete()
            return {
                "error": False,
                "message": f"Issue {issue_key} deleted successfully.",
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"Failed to delete issue {issue_key}: {e!s}",
            }
