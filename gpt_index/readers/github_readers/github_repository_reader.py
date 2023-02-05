import asyncio
import base64
import binascii
import os
from typing import Any, List, Optional, Tuple

from gpt_index.readers.base import BaseReader
from gpt_index.readers.github_readers.github_api_client import (
    GitBranchResponseModel,
    GitCommitResponseModel,
    GitTreeResponseModel,
)
from gpt_index.readers.github_readers.utils import (
    BufferedGitBlobDataIterator,
    print_if_verbose,
)
from gpt_index.readers.schema.base import Document


class GithubRepositoryReader(BaseReader):
    """
    Github repository reader.

    Retrieves the contents of a Github repository and returns a list of documents.
    The documents are either the contents of the files in the repository or the text extracted from the files using the parser.


    Args:
        - owner (str): Owner of the repository.
        - repo (str): Name of the repository.
        - use_parser (bool): Whether to use the parser to extract the text from the files.
        - verbose (bool): Whether to print verbose messages.
        - github_token (str): Github token. If not provided, it will be read from the GITHUB_TOKEN environment variable.

    Examples:
        >>> reader = GithubRepositoryReader("owner", "repo")
        >>> branch_documents = reader.load_data(branch="branch")
        >>> commit_documents = reader.load_data(commit="commit")

    """

    def __init__(
        self,
        owner: str,
        repo: str,
        use_parser: bool = True,
        verbose: bool = False,
        github_token: Optional[str] = None,
    ):
        super().__init__(verbose)
        if github_token is None:
            self.github_token = os.getenv("GITHUB_TOKEN")
            if self.github_token is None:
                raise ValueError(
                    "Please provide a Github token. "
                    "You can do so by passing it as an argument or by setting the GITHUB_TOKEN environment variable."
                )
        else:
            self.github_token = github_token

        self.owner = owner
        self.repo = repo
        self.use_parser = use_parser

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.__client = GithubClient(github_token)  # type: ignore

    def __load_data_from_commit(self, commit: str, depth: int = 1) -> List[Document]:
        commit_response: GitCommitResponseModel = self.loop.run_until_complete(
            self.__client.get_commit(self.owner, self.repo, commit)
        )

        tree_sha = commit_response.tree.sha
        blobs_and_paths = self.loop.run_until_complete(self.recurse_tree(tree_sha))

        print_if_verbose(self.verbose, f"got {len(blobs_and_paths)} blobs")

        return self.loop.run_until_complete(
            self.__generate_documents(blobs_and_paths=blobs_and_paths)
        )

    def __load_data_from_branch(self, branch: str, depth: int = 1) -> List[Document]:
        branch_data: GitBranchResponseModel = self.loop.run_until_complete(
            self.__client.get_branch(self.owner, self.repo, branch)
        )

        tree_sha = branch_data.commit.commit.tree.sha
        blobs_and_paths = self.loop.run_until_complete(self.recurse_tree(tree_sha))

        print_if_verbose(self.verbose, f"got {len(blobs_and_paths)} blobs")

        return self.loop.run_until_complete(
            self.__generate_documents(blobs_and_paths=blobs_and_paths)
        )

    def load_data(
        self,
        commit: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> List[Document]:
        if commit is not None and branch is not None:
            raise ValueError("You can only specify one of commit or branch.")

        if commit is None and branch is None:
            raise ValueError("You must specify one of commit or branch.")

        if commit is not None:
            return self.__load_data_from_commit(commit)

        if branch is not None:
            return self.__load_data_from_branch(branch)

        raise ValueError("You must specify one of commit or branch.")

    async def recurse_tree(
        self, tree_sha: str, current_path: str = "", current_depth: int = 0
    ) -> Any:
        """
        Recursively get all files in a tree and construct their full path in the repo relative to the root of the repo

        :param `tree_sha`: sha of the tree to recurse
        :param `current_path`: current path of the tree
        :param `current_depth`: current depth of the tree
        :return: list of tuples of (tree object, file's full path in the repo realtive to the root of the repo)

        """
        blobs_and_full_paths: List[Tuple[GitTreeResponseModel.GitTreeObject, str]] = []
        print_if_verbose(
            self.verbose, "\t" * current_depth + f"current path: {current_path}"
        )

        tree_data: GitTreeResponseModel = await self.__client.get_tree(
            self.owner, self.repo, tree_sha
        )
        print_if_verbose(
            self.verbose, "\t" * current_depth + f"processing tree {tree_sha}"
        )
        for tree in tree_data.tree:
            file_path = os.path.join(current_path, tree.path)
            if tree.type == "tree":
                print_if_verbose(
                    self.verbose, "\t" * current_depth + f"recursing into {tree.path}"
                )
                blobs_and_full_paths.extend(
                    await self.recurse_tree(tree.sha, file_path, current_depth + 1)
                )
            elif tree.type == "blob":
                print_if_verbose(
                    self.verbose, "\t" * current_depth + f"found blob {tree.path}"
                )
                blobs_and_full_paths.append((tree, file_path))
        return blobs_and_full_paths

    async def __generate_documents(
        self, blobs_and_paths: List[Tuple[GitTreeResponseModel.GitTreeObject, str]]
    ):
        """
        Generate documents from a list of blobs and their full paths in the repo relative to the root of the repo

        :param `blobs_and_paths`: list of tuples of (tree object, file's full path in the repo realtive to the root of the repo)
        :return: list of documents
        """
        buffered_iterator = BufferedGitBlobDataIterator(
            blobs_and_paths=blobs_and_paths,
            github_client=self.__client,
            owner=self.owner,
            repo=self.repo,
            loop=self.loop,
            buffer_size=2,  # TODO: make this configurable
        )

        documents = []
        async for blob_data, full_path in buffered_iterator:
            print(f"generating document for {full_path}")
            assert (
                blob_data.encoding == "base64"
            ), f"blob encoding {blob_data.encoding} not supported"
            decoded_bytes = None
            try:
                decoded_bytes = base64.b64decode(blob_data.content)
                del blob_data.content
            except binascii.Error:
                print(f"could not decode {full_path} as base64")
                continue

            if self.use_parser:
                document = self.__parse_supported_file(
                    file_path=full_path,
                    file_content=decoded_bytes,
                    tree_sha=blob_data.sha,
                    tree_path=full_path,
                )
                if document is not None:
                    documents.append(document)
                else:
                    continue

            try:
                if decoded_bytes is None:
                    raise ValueError("decoded_bytes is None")
                decoded_text = decoded_bytes.decode("utf-8")
            except UnicodeDecodeError:
                print(f"could not decode {full_path} as utf-8")
                continue
            print(
                f"got {len(decoded_text)} characters - adding to documents - {full_path}"
            )
            document = Document(
                text=decoded_text,
                doc_id=blob_data.sha,
                extra_info={
                    "file_path": full_path,
                    "file_name": full_path.split("/")[-1],
                },
            )
            documents.append(document)
        return documents
