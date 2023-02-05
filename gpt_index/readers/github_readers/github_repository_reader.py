import asyncio
import os
from typing import Optional

from gpt_index.readers.base import BaseReader


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
