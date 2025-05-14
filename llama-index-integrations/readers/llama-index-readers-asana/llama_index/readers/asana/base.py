from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

import asana


class AsanaReader(BaseReader):
    """
    Asana reader. Reads data from an Asana workspace.

    Args:
        asana_token (str): Asana token.

    """

    def __init__(self, asana_token: str) -> None:
        """Initialize Asana reader."""
        self.client = asana.Client.access_token(asana_token)

    def load_data(
        self, workspace_id: Optional[str] = None, project_id: Optional[str] = None
    ) -> List[Document]:
        """
        Load data from the workspace.

        Args:
            workspace_id (Optional[str], optional): Workspace ID. Defaults to None.
            project_id (Optional[str], optional): Project ID. Defaults to None.


        Returns:
            List[Document]: List of documents.

        """
        if workspace_id is None and project_id is None:
            raise ValueError("Either workspace_id or project_id must be provided")

        if workspace_id is not None and project_id is not None:
            raise ValueError(
                "Only one of workspace_id or project_id should be provided"
            )

        results = []

        if workspace_id is not None:
            workspace_name = self.client.workspaces.find_by_id(workspace_id)["name"]
            projects = self.client.projects.find_all({"workspace": workspace_id})

        # Case: Only project_id is provided
        else:  # since we've handled the other cases, this means project_id is not None
            projects = [self.client.projects.find_by_id(project_id)]
            workspace_name = projects[0]["workspace"]["name"]

        for project in projects:
            tasks = self.client.tasks.find_all(
                {
                    "project": project["gid"],
                    "opt_fields": "name,notes,completed,completed_at,completed_by,assignee,followers,custom_fields",
                }
            )
            for task in tasks:
                stories = self.client.tasks.stories(task["gid"], opt_fields="type,text")
                comments = "\n".join(
                    [
                        story["text"]
                        for story in stories
                        if story.get("type") == "comment" and "text" in story
                    ]
                )

                task_metadata = {
                    "task_id": task.get("gid", ""),
                    "name": task.get("name", ""),
                    "assignee": (task.get("assignee") or {}).get("name", ""),
                    "completed_on": task.get("completed_at", ""),
                    "completed_by": (task.get("completed_by") or {}).get("name", ""),
                    "project_name": project.get("name", ""),
                    "custom_fields": [
                        i["display_value"]
                        for i in task.get("custom_fields")
                        if task.get("custom_fields") is not None
                    ],
                    "workspace_name": workspace_name,
                    "url": f"https://app.asana.com/0/{project['gid']}/{task['gid']}",
                }

                if task.get("followers") is not None:
                    task_metadata["followers"] = [
                        i.get("name") for i in task.get("followers") if "name" in i
                    ]
                else:
                    task_metadata["followers"] = []

                results.append(
                    Document(
                        text=task.get("name", "")
                        + " "
                        + task.get("notes", "")
                        + " "
                        + comments,
                        extra_info=task_metadata,
                    )
                )

        return results
