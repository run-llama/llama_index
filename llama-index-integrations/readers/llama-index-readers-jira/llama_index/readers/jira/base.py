from typing import List, Optional, TypedDict

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class BasicAuth(TypedDict):
    email: str
    api_token: str
    server_url: str


class Oauth2(TypedDict):
    cloud_id: str
    api_token: str


class PATauth(TypedDict):
    server_url: str
    api_token: str


class JiraReader(BaseReader):
    """Jira reader. Reads data from Jira issues from passed query.

    Args:
        Optional basic_auth:{
            "email": "email",
            "api_token": "token",
            "server_url": "server_url"
        }
        Optional oauth:{
            "cloud_id": "cloud_id",
            "api_token": "token"
        }
        Optional patauth:{
            "server_url": "server_url",
            "api_token": "token"
        }
    """

    include_epics: bool = True

    def __init__(
        self,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        server_url: Optional[str] = None,
        BasicAuth: Optional[BasicAuth] = None,
        Oauth2: Optional[Oauth2] = None,
        PATauth: Optional[PATauth] = None,
        include_epics: bool = True,
    ) -> None:
        from jira import JIRA

        if email and api_token and server_url:
            if BasicAuth is None:
                BasicAuth = {}
            BasicAuth["email"] = email
            BasicAuth["api_token"] = api_token
            BasicAuth["server_url"] = server_url

        if Oauth2:
            options = {
                "server": f"https://api.atlassian.com/ex/jira/{Oauth2['cloud_id']}",
                "headers": {"Authorization": f"Bearer {Oauth2['api_token']}"},
            }
            self.jira = JIRA(options=options)
        elif PATauth:
            options = {
                "server": PATauth["server_url"],
                "headers": {"Authorization": f"Bearer {PATauth['api_token']}"},
            }
            self.jira = JIRA(options=options)
        else:
            self.jira = JIRA(
                basic_auth=(BasicAuth["email"], BasicAuth["api_token"]),
                server=f"https://{BasicAuth['server_url']}",
            )

        self.include_epics = include_epics

    def load_data(
        self, query: str, start_at: int = 0, max_results: int = 50
    ) -> List[Document]:
        relevant_issues = self.jira.search_issues(
            query, startAt=start_at, maxResults=max_results
        )

        issues = []

        assignee = ""
        reporter = ""
        epic_key = ""
        epic_summary = ""
        epic_descripton = ""

        for issue in relevant_issues:
            issue_type = issue.fields.issuetype.name
            if issue_type == "Epic" and not self.include_epics:
                continue

            assignee = ""
            reporter = ""
            epic_key = ""
            epic_summary = ""
            epic_descripton = ""

            if issue.fields.assignee:
                assignee = issue.fields.assignee.displayName
            if issue.fields.reporter:
                reporter = issue.fields.reporter.displayName

            if "parent" in issue.raw["fields"]:
                if issue.raw["fields"]["parent"]["key"]:
                    epic_key = issue.raw["fields"]["parent"]["key"]

                if issue.raw["fields"]["parent"]["fields"]["summary"]:
                    epic_summary = issue.raw["fields"]["parent"]["fields"]["summary"]

                if issue.raw["fields"]["parent"]["fields"]["status"]["description"]:
                    epic_descripton = issue.raw["fields"]["parent"]["fields"]["status"][
                        "description"
                    ]

            issues.append(
                Document(
                    text=f"{issue.fields.summary} \n {issue.fields.description}",
                    doc_id=issue.id,
                    extra_info={
                        "id": issue.id,
                        "title": issue.fields.summary,
                        "url": issue.permalink(),
                        "created_at": issue.fields.created,
                        "updated_at": issue.fields.updated,
                        "labels": issue.fields.labels,
                        "status": issue.fields.status.name,
                        "assignee": assignee,
                        "reporter": reporter,
                        "project": issue.fields.project.name,
                        "issue_type": issue_type,
                        "priority": issue.fields.priority.name,
                        "epic_key": epic_key,
                        "epic_summary": epic_summary,
                        "epic_description": epic_descripton,
                    },
                )
            )

        return issues
