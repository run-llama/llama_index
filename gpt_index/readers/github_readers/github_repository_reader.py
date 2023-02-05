import asyncio
import os
from typing import List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.github_readers.github_api_client import (
    GitBranchResponseModel,
    GitCommitResponseModel,
)
from gpt_index.readers.github_readers.utils import print_if_verbose
from gpt_index.readers.schema.base import Document


class GithubRepositoryReader(BaseReader):
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
