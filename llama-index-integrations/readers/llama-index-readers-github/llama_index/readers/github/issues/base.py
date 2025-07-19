import os
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, override

from githubkit.auth.token import TokenAuthStrategy
from githubkit.github import GitHub
from githubkit.utils import UNSET
from githubkit.versions.latest.models import Issue, IssueComment
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document, MediaResource
from pydantic import Field

if TYPE_CHECKING:
    from githubkit.rest.paginator import Paginator


class GithubIssueCommentsClient(BasePydanticReader):
    owner: str = Field(description="The owner of the GitHub repository.")
    repo: str = Field(description="The name of the GitHub repository.")

    client: GitHub[Any] = Field(
        default_factory=lambda: GitHub(
            TokenAuthStrategy(token=os.environ["GITHUB_TOKEN"])
        ),
        description="The GitHub client.",
    )

    def _comment_to_document(self, comment: IssueComment) -> Document:
        metadata: dict[str, Any] = {}

        if user := comment.user:
            metadata["user.login"] = user.login
            metadata["user.association"] = comment.author_association

            if user.email:
                metadata["user.email"] = user.email

        if reactions := comment.reactions:
            metadata["reactions.total_count"] = reactions.total_count or 0
            metadata["reactions.plus_one_count"] = reactions.plus_one or 0
            metadata["reactions.minus_one_count"] = reactions.minus_one or 0
            metadata["reactions.laugh_count"] = reactions.laugh or 0
            metadata["reactions.hooray_count"] = reactions.hooray or 0

        return Document(
            text_resource=MediaResource(text=comment.body or ""),
            metadata={
                "id": comment.id,
                "type": "comment",
                "updated_at": comment.updated_at.isoformat(),
                "created_at": comment.created_at.isoformat(),
                **metadata,
            },
        )

    @override
    def load_data(
        self,
        issue_number: int,
        since: datetime | None = None,
        per_page: int = 100,
        page: int = 1,
    ) -> list[Document]:
        return list(self.lazy_load_data(issue_number, since, per_page, page))

    @override
    def lazy_load_data(
        self,
        issue_number: int,
        since: datetime | None = None,
        per_page: int = 100,
        page: int = 1,
    ) -> Generator[Document, Any, Any]:
        paginator: Paginator[IssueComment] = self.client.rest.paginate(
            self.client.rest.issues.list_comments,
            owner=self.owner,
            repo=self.repo,
            issue_number=issue_number,
            since=since if since else UNSET,
            per_page=per_page,
            page=page,
        )

        for comment in paginator:
            yield self._comment_to_document(comment)

    @override
    async def aload_data(
        self,
        issue_number: int,
        since: datetime | None = None,
        per_page: int = 100,
        page: int = 1,
    ) -> list[Document]:
        return [
            doc
            async for doc in self.alazy_load_data(issue_number, since, per_page, page)
        ]

    @override
    async def alazy_load_data(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        issue_number: int,
        since: datetime | None = None,
        per_page: int = 100,
        page: int = 1,
    ) -> AsyncGenerator[Document, Any]:
        paginator: Paginator[IssueComment] = self.client.rest.paginate(
            self.client.rest.issues.async_list_comments,
            owner=self.owner,
            repo=self.repo,
            issue_number=issue_number,
            since=since if since else UNSET,
            per_page=per_page,
            page=page,
        )

        async for comment in paginator:
            yield self._comment_to_document(comment)


class GithubIssuesClient(BasePydanticReader):
    owner: str = Field(description="The owner of the GitHub repository.")
    repo: str = Field(description="The name of the GitHub repository.")

    client: GitHub[Any] = Field(
        default_factory=lambda: GitHub(
            TokenAuthStrategy(token=os.environ["GITHUB_TOKEN"])
        ),
        description="The GitHub client.",
    )

    @cached_property
    def comments_client(self) -> GithubIssueCommentsClient:
        return GithubIssueCommentsClient(
            owner=self.owner, repo=self.repo, client=self.client
        )

    def _issue_to_document(self, issue: Issue) -> Document:
        metadata: dict[str, Any] = {}

        if user := issue.user:
            metadata["user.login"] = user.login
            metadata["user.association"] = issue.author_association

            if user.email:
                metadata["user.email"] = user.email

        if milestone := issue.milestone:
            metadata["milestone.title"] = milestone.title
            metadata["milestone.description"] = milestone.description
            metadata["milestone.state"] = milestone.state
            metadata["milestone.created_at"] = milestone.created_at.isoformat()
            metadata["milestone.updated_at"] = milestone.updated_at.isoformat()
            metadata["milestone.due_on"] = (
                milestone.due_on.isoformat() if milestone.due_on else None
            )

        if assignee := issue.assignee:
            metadata["assignee.login"] = assignee.login
            metadata["assignee.association"] = issue.author_association

            if assignee.email:
                metadata["assignee.email"] = assignee.email

        return Document(
            text_resource=MediaResource(text=issue.body or ""),
            metadata={
                "number": issue.number,
                "type": "issue",
                "title": issue.title,
                "url": issue.url,
                "state": issue.state,
                "comment_count": issue.comments,
                "created_at": issue.created_at.isoformat(),
                "updated_at": issue.updated_at.isoformat(),
                "labels": issue.labels,
                **metadata,
            },
        )

    @override
    def lazy_load_data(
        self,
        *,
        state: Literal["open", "closed", "all"] | None = None,
        milestone: str | None = None,
        labels: str | None = None,
        assignee: str | None = None,
        sort: Literal["created", "updated", "comments"] | None = None,
        direction: Literal["asc", "desc"] | None = None,
        creator: str | None = None,
        include_comments: bool = False,
    ) -> Generator[Document, Any, Any]:
        paginator: Paginator[Issue] = self.client.rest.paginate(
            self.client.rest.issues.list_for_repo,
            owner=self.owner,
            repo=self.repo,
            state=state if state else UNSET,
            milestone=milestone if milestone else UNSET,
            labels=labels if labels else UNSET,
            assignee=assignee if assignee else UNSET,
            sort=sort if sort else UNSET,
            direction=direction if direction else UNSET,
            creator=creator if creator else UNSET,
            per_page=100,
        )

        for issue in paginator:
            yield self._issue_to_document(issue)

            if issue.comments > 0 and include_comments:
                yield from self.comments_client.lazy_load_data(
                    issue_number=issue.number
                )

    @override
    def load_data(
        self,
        *,
        state: Literal["open", "closed", "all"] | None = None,
        milestone: str | None = None,
        labels: str | None = None,
        assignee: str | None = None,
        sort: Literal["created", "updated", "comments"] | None = None,
        direction: Literal["asc", "desc"] | None = None,
        creator: str | None = None,
        include_comments: bool = False,
    ) -> list[Document]:
        return list(
            self.lazy_load_data(
                state=state,
                milestone=milestone,
                labels=labels,
                assignee=assignee,
                sort=sort,
                direction=direction,
                creator=creator,
                include_comments=include_comments,
            )
        )

    @override
    async def alazy_load_data(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        state: Literal["open", "closed", "all"] | None = None,
        milestone: str | None = None,
        labels: str | None = None,
        assignee: str | None = None,
        sort: Literal["created", "updated", "comments"] | None = None,
        direction: Literal["asc", "desc"] | None = None,
        creator: str | None = None,
        include_comments: bool = False,
    ) -> AsyncGenerator[Document, Any]:
        paginator: Paginator[Issue] = self.client.rest.paginate(
            self.client.rest.issues.async_list_for_repo,
            owner=self.owner,
            repo=self.repo,
            state=state if state else UNSET,
            milestone=milestone if milestone else UNSET,
            labels=labels if labels else UNSET,
            assignee=assignee if assignee else UNSET,
            sort=sort if sort else UNSET,
            direction=direction if direction else UNSET,
            creator=creator if creator else UNSET,
            per_page=100,
        )

        async for issue in paginator:
            yield self._issue_to_document(issue)

            if issue.comments > 0 and include_comments:
                async for comment in self.comments_client.alazy_load_data(
                    issue_number=issue.number
                ):
                    yield comment

    @override
    async def aload_data(
        self,
        *,
        state: Literal["open", "closed", "all"] | None = None,
        milestone: str | None = None,
        labels: str | None = None,
        assignee: str | None = None,
        sort: Literal["created", "updated", "comments"] | None = None,
        direction: Literal["asc", "desc"] | None = None,
        creator: str | None = None,
        include_comments: bool = False,
    ) -> list[Document]:
        return [
            doc
            async for doc in self.alazy_load_data(
                state=state,
                milestone=milestone,
                labels=labels,
                assignee=assignee,
                sort=sort,
                direction=direction,
                creator=creator,
                include_comments=include_comments,
            )
        ]
