"""Unit tests for JiraIssueToolSpec."""

from unittest.mock import Mock, patch

import pytest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.jira_issue.base import JiraIssueToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in JiraIssueToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


class TestJiraIssueToolSpec:
    """Test suite for JiraIssueToolSpec."""

    @pytest.fixture
    def mock_jira(self):
        """Create a mock JIRA client."""
        with patch("llama_index.tools.jira_issue.base.JIRA") as mock:
            yield mock

    @pytest.fixture
    def jira_tool_spec(self, mock_jira):
        """Create a JiraIssueToolSpec instance with mocked JIRA client."""
        mock_jira.return_value = Mock()
        return JiraIssueToolSpec(
            email="test@example.com",
            api_key="test-api-key",
            server_url="https://test.atlassian.net",
        )

    def test_init_with_missing_credentials(self):
        """Test that initialization fails with missing credentials."""
        with pytest.raises(Exception, match="Please provide Jira credentials"):
            JiraIssueToolSpec(email="", api_key="", server_url="")

    def test_search_issues_success(self, jira_tool_spec):
        """Test successful issue search."""
        # Mock issue objects
        mock_issue1 = Mock()
        mock_issue1.key = "PROJ-123"
        mock_issue1.fields.summary = "Test Issue 1"
        mock_issue1.fields.status.name = "In Progress"
        mock_issue1.fields.assignee = Mock(displayName="John Doe")

        mock_issue2 = Mock()
        mock_issue2.key = "PROJ-124"
        mock_issue2.fields.summary = "Test Issue 2"
        mock_issue2.fields.status.name = "To Do"
        mock_issue2.fields.assignee = None

        jira_tool_spec.jira.search_issues.return_value = [mock_issue1, mock_issue2]

        result = jira_tool_spec.search_issues("project = PROJ")

        assert result["error"] is False
        assert result["message"] == "Issues found"
        assert len(result["issues"]) == 2
        assert result["issues"][0]["key"] == "PROJ-123"
        assert result["issues"][0]["assignee"] == "John Doe"
        assert result["issues"][1]["assignee"] is None

    def test_search_issues_no_results(self, jira_tool_spec):
        """Test issue search with no results."""
        jira_tool_spec.jira.search_issues.return_value = []

        result = jira_tool_spec.search_issues("project = NONEXISTENT")

        assert result["error"] is True
        assert result["message"] == "No issues found."

    def test_search_issues_failure(self, jira_tool_spec):
        """Test failed issue search."""
        jira_tool_spec.jira.search_issues.side_effect = Exception("Invalid JQL")

        result = jira_tool_spec.search_issues("invalid jql")

        assert result["error"] is True
        assert "Failed to search issues: Invalid JQL" in result["message"]

    def test_create_issue_success(self, jira_tool_spec):
        """Test successful issue creation."""
        mock_issue = Mock(key="KAN-123")
        jira_tool_spec.jira.create_issue.return_value = mock_issue

        result = jira_tool_spec.create_issue(
            project_key="KAN",
            summary="New Test Issue",
            description="Test description",
            issue_type="Task",
        )

        assert result["error"] is False
        assert result["message"] == "Issue KAN-123 created successfully."
        assert result["issue_key"] == "KAN-123"

        # Verify the create_issue was called with correct parameters
        jira_tool_spec.jira.create_issue.assert_called_once_with(
            project="KAN",
            summary="New Test Issue",
            description="Test description",
            issuetype={"name": "Task"},
        )

    def test_create_issue_failure(self, jira_tool_spec):
        """Test failed issue creation."""
        jira_tool_spec.jira.create_issue.side_effect = Exception("Project not found")

        result = jira_tool_spec.create_issue(project_key="INVALID")

        assert result["error"] is True
        assert "Failed to create new issue: Project not found" in result["message"]

    def test_add_comment_to_issue_success(self, jira_tool_spec):
        """Test successful comment addition."""
        mock_issue = Mock()
        jira_tool_spec.jira.issue.return_value = mock_issue

        result = jira_tool_spec.add_comment_to_issue("KAN-123", "Test comment")

        assert result["error"] is False
        assert result["message"] == "Comment added to issue KAN-123."
        jira_tool_spec.jira.add_comment.assert_called_once_with(
            mock_issue, "Test comment"
        )

    def test_add_comment_to_issue_failure(self, jira_tool_spec):
        """Test failed comment addition."""
        jira_tool_spec.jira.issue.side_effect = Exception("Issue not found")

        result = jira_tool_spec.add_comment_to_issue("INVALID-123", "Test comment")

        assert result["error"] is True
        assert "Failed to add comment to issue INVALID-123" in result["message"]

    def test_update_issue_summary_success(self, jira_tool_spec):
        """Test successful summary update."""
        mock_issue = Mock()
        jira_tool_spec.jira.issue.return_value = mock_issue

        result = jira_tool_spec.update_issue_summary(
            "KAN-123", "Updated Summary", notify=True
        )

        assert result["error"] is False
        assert result["message"] == "Issue KAN-123 summary updated."
        mock_issue.update.assert_called_once_with(
            summary="Updated Summary", notify=True
        )

    def test_update_issue_summary_failure(self, jira_tool_spec):
        """Test failed summary update."""
        jira_tool_spec.jira.issue.side_effect = Exception("Permission denied")

        result = jira_tool_spec.update_issue_summary("KAN-123", "Updated Summary")

        assert result["error"] is True
        assert "Failed to update issue KAN-123: Permission denied" in result["message"]

    def test_update_issue_assignee_success(self, jira_tool_spec):
        """Test successful assignee update."""
        mock_user = Mock()
        mock_user.displayName = "John Doe"
        mock_user.accountId = "12345"
        jira_tool_spec.jira.search_users.return_value = [mock_user]

        mock_issue = Mock()
        jira_tool_spec.jira.issue.return_value = mock_issue

        result = jira_tool_spec.update_issue_assignee("KAN-123", "John Doe")

        assert result["error"] is False
        assert result["message"] == "Issue KAN-123 successfully assigned to John Doe"
        mock_issue.update.assert_called_once_with(assignee={"accountId": "12345"})

    def test_update_issue_assignee_user_not_found(self, jira_tool_spec):
        """Test assignee update when user is not found."""
        jira_tool_spec.jira.search_users.return_value = []

        result = jira_tool_spec.update_issue_assignee("KAN-123", "Unknown User")

        assert result["error"] is True
        assert "User with full name 'Unknown User' not found" in result["message"]

    def test_update_issue_assignee_failure(self, jira_tool_spec):
        """Test failed assignee update."""
        jira_tool_spec.jira.search_users.side_effect = Exception("API Error")

        result = jira_tool_spec.update_issue_assignee("KAN-123", "John Doe")

        assert result["error"] is True
        assert (
            "An error occurred while updating the assignee: API Error"
            in result["message"]
        )

    def test_update_issue_status_success(self, jira_tool_spec):
        """Test successful status update."""
        mock_issue = Mock()
        jira_tool_spec.jira.issue.return_value = mock_issue
        jira_tool_spec.jira.transitions.return_value = [
            {"id": "1", "name": "To Do"},
            {"id": "2", "name": "In Progress"},
            {"id": "3", "name": "Done"},
        ]

        result = jira_tool_spec.update_issue_status("KAN-123", "Done")

        assert result["error"] is False
        assert result["message"] == "Issue KAN-123 status updated to Done."
        jira_tool_spec.jira.transition_issue.assert_called_once_with(mock_issue, "3")

    def test_update_issue_status_invalid_transition(self, jira_tool_spec):
        """Test status update with invalid transition."""
        mock_issue = Mock()
        jira_tool_spec.jira.issue.return_value = mock_issue
        jira_tool_spec.jira.transitions.return_value = [
            {"id": "1", "name": "To Do"},
            {"id": "2", "name": "In Progress"},
        ]

        result = jira_tool_spec.update_issue_status("KAN-123", "Done")

        assert result["error"] is True
        assert "Status 'Done' not available for issue KAN-123" in result["message"]
        assert "Available transitions: ['To Do', 'In Progress']" in result["message"]

    def test_update_issue_status_failure(self, jira_tool_spec):
        """Test failed status update."""
        jira_tool_spec.jira.issue.side_effect = Exception("Issue not found")

        result = jira_tool_spec.update_issue_status("INVALID-123", "Done")

        assert result["error"] is True
        assert "Failed to update status for issue INVALID-123" in result["message"]

    def test_update_issue_due_date_success(self, jira_tool_spec):
        """Test successful due date update."""
        mock_issue = Mock()
        jira_tool_spec.jira.issue.return_value = mock_issue

        result = jira_tool_spec.update_issue_due_date("KAN-123", "2024-12-31")

        assert result["error"] is False
        assert result["message"] == "Issue KAN-123 due date updated."
        mock_issue.update.assert_called_once_with(duedate="2024-12-31")

    def test_update_issue_due_date_clear(self, jira_tool_spec):
        """Test clearing due date."""
        mock_issue = Mock()
        jira_tool_spec.jira.issue.return_value = mock_issue

        result = jira_tool_spec.update_issue_due_date("KAN-123", None)

        assert result["error"] is False
        assert result["message"] == "Issue KAN-123 due date cleared."
        mock_issue.update.assert_called_once_with(duedate=None)

    def test_update_issue_due_date_invalid_format(self, jira_tool_spec):
        """Test due date update with invalid date format."""
        result = jira_tool_spec.update_issue_due_date("KAN-123", "31-12-2024")

        assert result["error"] is True
        assert result["message"] == "Invalid date format. Use YYYY-MM-DD."

    def test_update_issue_due_date_failure(self, jira_tool_spec):
        """Test failed due date update."""
        jira_tool_spec.jira.issue.side_effect = Exception("Permission denied")

        result = jira_tool_spec.update_issue_due_date("KAN-123", "2024-12-31")

        assert result["error"] is True
        assert "Failed to update due date for issue KAN-123" in result["message"]

    def test_delete_issue_success(self, jira_tool_spec):
        """Test successful issue deletion."""
        mock_issue = Mock()
        jira_tool_spec.jira.issue.return_value = mock_issue

        result = jira_tool_spec.delete_issue("KAN-123")

        assert result["error"] is False
        assert result["message"] == "Issue KAN-123 deleted successfully."
        mock_issue.delete.assert_called_once()

    def test_delete_issue_failure(self, jira_tool_spec):
        """Test failed issue deletion."""
        jira_tool_spec.jira.issue.side_effect = Exception("Issue not found")

        result = jira_tool_spec.delete_issue("INVALID-123")

        assert result["error"] is True
        assert "Failed to delete issue INVALID-123" in result["message"]
