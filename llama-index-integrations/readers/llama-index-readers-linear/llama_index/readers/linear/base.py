from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class LinearReader(BaseReader):
    """
    Linear reader. Reads data from Linear issues for the passed query.

    Args:
        api_key (str): Personal API token.

    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def load_data(self, query: str) -> List[Document]:
        # Define the GraphQL query
        graphql_endpoint = "https://api.linear.app/graphql"
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"query": query}

        # Make the GraphQL request
        response = requests.post(graphql_endpoint, json=payload, headers=headers)
        data = response.json()

        # Extract relevant information
        issues = []
        team_data = data.get("data", {}).get("team", {})
        for issue in team_data.get("issues", {}).get("nodes", []):
            assignee = issue.get("assignee", {}).get("name", "")
            labels = [
                label_node["name"]
                for label_node in issue.get("labels", {}).get("nodes", [])
            ]
            project = issue.get("project", {}).get("name", "")
            state = issue.get("state", {}).get("name", "")
            creator = issue.get("creator", {}).get("name", "")

            issues.append(
                Document(
                    text=f"{issue['title']} \n {issue['description']}",
                    extra_info={
                        "id": issue["id"],
                        "title": issue["title"],
                        "created_at": issue["createdAt"],
                        "archived_at": issue["archivedAt"],
                        "auto_archived_at": issue["autoArchivedAt"],
                        "auto_closed_at": issue["autoClosedAt"],
                        "branch_name": issue["branchName"],
                        "canceled_at": issue["canceledAt"],
                        "completed_at": issue["completedAt"],
                        "creator": creator,
                        "due_date": issue["dueDate"],
                        "estimate": issue["estimate"],
                        "labels": labels,
                        "project": project,
                        "state": state,
                        "updated_at": issue["updatedAt"],
                        "assignee": assignee,
                    },
                )
            )

        return issues
