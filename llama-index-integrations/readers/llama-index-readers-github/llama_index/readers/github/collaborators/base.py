"""
GitHub repository collaborators reader.

Retrieves the list of collaborators in a GitHub repository and converts them to documents.

Each collaborator is converted to a document by doing the following:

    - The text of the document is the login.
    - The title of the document is also the login.
    - The extra_info of the document is a dictionary with the following keys:
        - login: str, the login of the user
        - type: str, the type of user e.g. "User"
        - site_admin: bool, whether the user has admin permissions
        - role_name: str, e.g. "admin"
        - name: str, the name of the user, if available
        - email: str, the email of the user, if available
        - permissions: str, the permissions of the user, if available

"""

import asyncio
import enum
import logging
from typing import Dict, List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.github.collaborators.github_client import (
    BaseGitHubCollaboratorsClient,
    GitHubCollaboratorsClient,
)

logger = logging.getLogger(__name__)


def print_if_verbose(verbose: bool, message: str) -> None:
    """Log message if verbose is True."""
    if verbose:
        print(message)


class GitHubRepositoryCollaboratorsReader(BaseReader):
    """
    GitHub repository collaborators reader.

    Retrieves the list of collaborators of a GitHub repository and returns a list of documents.

    Examples:
        >>> reader = GitHubRepositoryCollaboratorsReader("owner", "repo")
        >>> colabs = reader.load_data()
        >>> print(colabs)

    """

    class FilterType(enum.Enum):
        """
        Filter type.

        Used to determine whether the filter is inclusive or exclusive.
        """

        EXCLUDE = enum.auto()
        INCLUDE = enum.auto()

    def __init__(
        self,
        github_client: BaseGitHubCollaboratorsClient,
        owner: str,
        repo: str,
        verbose: bool = False,
    ):
        """
        Initialize params.

        Args:
            - github_client (BaseGitHubCollaboratorsClient): GitHub client.
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
    ) -> List[Document]:
        """
        GitHub repository collaborators reader.

        Retrieves the list of collaborators in a GitHub repository and converts them to documents.

        Each collaborator is converted to a document by doing the following:

            - The text of the document is the login.
            - The title of the document is also the login.
            - The extra_info of the document is a dictionary with the following keys:
                - login: str, the login of the user
                - type: str, the type of user e.g. "User"
                - site_admin: bool, whether the user has admin permissions
                - role_name: str, e.g. "admin"
                - name: str, the name of the user, if available
                - email: str, the email of the user, if available
                - permissions: str, the permissions of the user, if available


        :return: list of documents
        """
        documents = []
        page = 1
        # Loop until there are no more collaborators
        while True:
            collaborators: Dict = self._loop.run_until_complete(
                self._github_client.get_collaborators(
                    self._owner, self._repo, page=page
                )
            )

            if len(collaborators) == 0:
                print_if_verbose(self._verbose, "No more collaborators found, stopping")

                break
            print_if_verbose(
                self._verbose,
                f"Found {len(collaborators)} collaborators in the repo page {page}",
            )
            page += 1
            for collab in collaborators:
                extra_info = {
                    "login": collab["login"],
                    "type": collab["type"],
                    "site_admin": collab["site_admin"],
                    "role_name": collab["role_name"],
                }
                if collab.get("name") is not None:
                    extra_info["name"] = collab["name"]
                if collab.get("email") is not None:
                    extra_info["email"] = collab["email"]
                if collab.get("permissions") is not None:
                    extra_info["permissions"] = collab["permissions"]
                document = Document(
                    doc_id=str(collab["login"]),
                    text=str(collab["login"]),  # unsure for this
                    extra_info=extra_info,
                )
                documents.append(document)

            print_if_verbose(self._verbose, f"Resulted in {len(documents)} documents")

        return documents


if __name__ == "__main__":
    """Load all collaborators in the repo labeled as bug."""
    github_client = GitHubCollaboratorsClient(verbose=True)

    reader = GitHubRepositoryCollaboratorsReader(
        github_client=github_client,
        owner="moncho",
        repo="dry",
        verbose=True,
    )

    documents = reader.load_data()
    print(f"Got {len(documents)} documents")
