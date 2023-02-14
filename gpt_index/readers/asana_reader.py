"""Asana reader."""
import logging
from typing import List

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class AsanaReader(BaseReader):
    """Asana reader. Reads data from an Asana workspace.

    Args:
        asana_token (str): Asana token.
        asana_workspace (str): Asana workspace.
    """

    def __init__(self, asana_token: str) -> None:
        """Initialize Asana reader."""
        try:
            import asana
        except ImportError:
            raise ValueError(
                "`asana` package not found, please run `pip install asana`"
            )
        self.client = asana.Client.access_token(asana_token)

    def load_data(self, workspace_id: str) -> List[Document]:
        """Load data from the workspace.

        Args:
            workspace_id (str): Workspace ID.
        Returns:
            List[Document]: List of documents.
        """
        results = []

        projects = self.client.projects.find_all({"workspace": workspace_id})

        for project in projects:
            tasks = self.client.tasks.find_all(
                {
                    "project": project["gid"],
                    "opt_fields": "name,notes,completed,due_on,assignee",
                }
            )
            for task in tasks:
                stories = self.client.tasks.stories(task["gid"], opt_fields="type,text")
                comments = "\n".join(
                    [story["text"] for story in stories if story["type"] == "comment"]
                )
                results.append(
                    Document(
                        task["name"] + " " + task["notes"] + " " + comments,
                        extra_info={
                            "task_id": task["gid"],
                            "name": task["name"],
                            "assignee": task["assignee"],
                            "project": project["name"],
                        },
                    )
                )

        return results
