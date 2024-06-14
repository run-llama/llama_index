from typing import Dict, List, Optional, Callable
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class AzureDevopsReader(BaseReader):
    """
    A loader class for Azure DevOps repositories. This class provides methods to authenticate with Azure DevOps,
    access repositories, and retrieve file content.

    Attributes:
        access_token (str): The personal access token for Azure DevOps.
        organization_name (str): The name of the organization in Azure DevOps.
        project_name (str): The name of the project in Azure DevOps.
        repo (str): The name of the repository in the project.
        organization_url (str): The URL to the organization in Azure DevOps.
        git_client: The Git client for interacting with Azure DevOps.
        repository_id: The ID of the repository in Azure DevOps.
    """

    def __init__(
        self,
        access_token: str,
        organization_name: str,
        project_name: str,
        repo: str,
        file_filter: Optional[Callable[[str], bool]] = False,
    ):
        """
        Initializes the AzureDevopsLoader with the necessary details to interact with an Azure DevOps repository.

        Parameters:
            access_token (str): The personal access token for Azure DevOps.
            organization_name (str): The name of the organization in Azure DevOps.
            project_name (str): The name of the project in Azure DevOps.
            repo (str): The name of the repository in the project.
            file_filter(callable): A function that can be used as file filter ex: `lambda file_path: file_path.endswith(".py")`
        """
        self.access_token = access_token
        self.project_name = project_name
        self.repo = repo
        self.organization_url = f"https://dev.azure.com/{organization_name}/"
        self.file_filter = file_filter

        self.git_client = self.create_git_client()
        self.repository_id = self._get_repository_id(repo_name=self.repo)

    def create_git_client(self):
        """
        Creates and returns a Git client for interacting with Azure DevOps.

        Returns:
            The Git client object for Azure DevOps.
        """
        try:
            from azure.devops.connection import Connection
            from msrest.authentication import BasicAuthentication
        except ImportError:
            raise ImportError(
                "Please install azure-devops to use the AzureDevopsLoader. "
                "You can do so by running `pip install azure-devops`."
            )
        credentials = BasicAuthentication("", self.access_token)
        connection = Connection(base_url=self.organization_url, creds=credentials)
        return connection.clients.get_git_client()

    def _get_repository_id(self, repo_name: str):
        """
        Retrieves the repository ID for a given repository name.

        Parameters:
            repo_name (str): The name of the repository.

        Returns:
            The ID of the repository.
        """
        repositories = self.git_client.get_repositories(project=self.project_name)
        return next((repo.id for repo in repositories if repo.name == repo_name), None)

    def _create_version_descriptor(self, branch: Optional[str]):
        """
        Creates a version descriptor for a given branch.

        Parameters:
            branch (Optional[str]): The name of the branch to create a version descriptor for.

        Returns:
            A version descriptor if a branch is specified, otherwise None.
        """
        if branch:
            from azure.devops.v7_0.git.models import GitVersionDescriptor

            version_descriptor = GitVersionDescriptor(
                version=branch, version_type="branch"
            )
        else:
            version_descriptor = None
        return version_descriptor

    def get_file_paths(self, folder: str = "/", version_descriptor=None) -> List[Dict]:
        """
        Retrieves the paths of all files within a given folder in the repository.

        Parameters:
            folder (str): The folder to retrieve file paths from, defaults to root.
            version_descriptor (Optional): The version descriptor to specify a version or branch.

        Returns:
            A list of paths of the files.
        """
        items = self.git_client.get_items(
            repository_id=self.repository_id,
            project=self.project_name,
            scope_path=folder,
            recursion_level="Full",
            version_descriptor=version_descriptor,
        )
        return [
            {"path": item.path, "url": item.url}
            for item in items
            if not (self.file_filter and not self.file_filter(item.path))
            and (item.git_object_type == "blob")
        ]

    def get_file_content_by_path(self, path: str, version_descriptor=None):
        """
        Retrieves the content of a file by its path in the repository.

        Parameters:
            path (str): The path of the file in the repository.
            version_descriptor (Optional): The version descriptor to specify a version or branch.

        Returns:
            The content of the file as a string.
        """
        try:
            stream = self.git_client.get_item_text(
                repository_id=self.repository_id,
                path=path,
                project=self.project_name,
                download=True,
                version_descriptor=version_descriptor,
            )
            file_content = ""
            # Iterate over the generator object
            for chunk in stream:
                # Assuming the content is encoded in UTF-8, decode each chunk and append to the file_content string
                file_content += chunk.decode("utf-8")
            return file_content
        except Exception as e:
            print(f"failed loading {path}")
            return None

    def load_data(self, folder: Optional[str] = "/", branch: Optional[str] = None):
        """
        Loads the documents from a specified folder and branch in the repository.

        Parameters:
            folder (Optional[str]): The folder to load documents from, defaults to root.
            branch (Optional[str]): The branch to load documents from.

        Returns:
            A list of Document objects representing the loaded documents.
        """
        documents = []
        version_descriptor = self._create_version_descriptor(branch=branch)
        files = self.get_file_paths(
            folder=folder, version_descriptor=version_descriptor
        )
        for file in files:
            path = file["path"]
            content = self.get_file_content_by_path(
                path=path, version_descriptor=version_descriptor
            )
            if content:
                metadata = {
                    "path": path,
                    "extension": path.split(".")[-1],
                    "source": file["url"],
                }
                documents.append(Document(text=content, extra_info=metadata))
        return documents
