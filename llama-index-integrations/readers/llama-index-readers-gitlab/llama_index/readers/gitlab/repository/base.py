""" GitLab repository reader. """

from typing import List, Optional

import gitlab
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class GitLabRepositoryReader(BaseReader):
    def __init__(
        self,
        gitlab_client: gitlab.Gitlab,
        project_id: int,
        use_parser: bool = False,
        verbose: bool = False,
    ):
        super().__init__()

        self._gl = gitlab_client
        self._use_parser = use_parser
        self._verbose = verbose
        self._project_url = f"{gitlab_client.api_url}/projects/{project_id}"

        self._project = gitlab_client.projects.get(project_id)

    def _parse_file_content(self, file_properties: dict, file_content: str) -> Document:
        raise NotImplementedError

    def _load_single_file(self, file_path: str, ref: Optional[str] = None) -> Document:
        file = self._project.files.get(file_path=file_path, ref=ref)
        file_properties = file.asdict()
        file_content = file.decode()

        if self._use_parser:
            return self._parse_file_content(file_properties, file_content)

        return Document(
            doc_id=file_properties["blob_id"],
            text=file_content,
            extra_info={
                "file_path": file_properties["file_path"],
                "file_name": file_properties["file_name"],
                "size": file_properties["size"],
                "url": f"{self._project_url}/projects/repository/files/{file_properties['file_path']}/raw",
            },
        )

    def load_data(
        self,
        ref: str,
        file_path: Optional[str] = None,
        path: Optional[str] = None,
        recursive: bool = False,
    ) -> List[Document]:
        """
        Load data from a GitLab repository.

        Args:
            ref: The name of a repository branch or commit id
            file_path: Path to the file to load.
            path: Path to the directory to load.
            recursive: Whether to load files recursively.

        Returns:
            List[Document]: List of documents loaded from the repository
        """
        if file_path:
            return [self._load_single_file(file_path, ref)]

        project = self._project

        params = {
            "ref": ref,
            "path": path,
            "recursive": recursive,
        }

        filtered_params = {k: v for k, v in params.items() if v is not None}

        repo_items = project.repository_tree(**filtered_params)

        documents = []

        for item in repo_items:
            if item["type"] == "blob":
                documents.append(self._load_single_file(item["path"], ref))

        return documents


if __name__ == "__main__":
    reader = GitLabRepositoryReader(
        gitlab_client=gitlab.Gitlab("https://gitlab.com"),
        project_id=25527353,
        verbose=True,
    )
    docs = reader.load_data(file_path="README.rst", ref="develop")
    print(docs)
