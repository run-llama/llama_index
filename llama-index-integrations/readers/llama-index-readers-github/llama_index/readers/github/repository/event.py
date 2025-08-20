from typing import Optional

from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.schema import Document


# GitHub-specific LlamaIndex events
class GitHubRepositoryProcessingStartedEvent(BaseEvent):
    """Event dispatched when GitHub repository processing starts."""

    repository_name: str
    branch_or_commit: str

    @classmethod
    def class_name(cls) -> str:
        return "GitHubRepositoryProcessingStartedEvent"


class GitHubRepositoryProcessingCompletedEvent(BaseEvent):
    """Event dispatched when GitHub repository processing completes."""

    repository_name: str
    branch_or_commit: str
    total_documents: int = 0

    @classmethod
    def class_name(cls) -> str:
        return "GitHubRepositoryProcessingCompletedEvent"


class GitHubTotalFilesToProcessEvent(BaseEvent):
    """Event dispatched with total number of files to process."""

    repository_name: str
    branch_or_commit: str
    total_files: int

    @classmethod
    def class_name(cls) -> str:
        return "GitHubTotalFilesToProcessEvent"


class GitHubFileProcessingStartedEvent(BaseEvent):
    """Event dispatched when file processing starts."""

    file_path: str
    file_type: str

    @classmethod
    def class_name(cls) -> str:
        return "GitHubFileProcessingStartedEvent"


class GitHubFileProcessedEvent(BaseEvent):
    """Event dispatched when a file is successfully processed."""

    file_path: str
    file_type: str
    file_size: Optional[int] = None
    document: Optional[Document] = None

    @classmethod
    def class_name(cls) -> str:
        return "GitHubFileProcessedEvent"


class GitHubFileSkippedEvent(BaseEvent):
    """Event dispatched when a file is skipped."""

    file_path: str
    file_type: str
    reason: str = ""

    @classmethod
    def class_name(cls) -> str:
        return "GitHubFileSkippedEvent"


class GitHubFileFailedEvent(BaseEvent):
    """Event dispatched when file processing fails."""

    file_path: str
    file_type: str
    error: str = ""

    @classmethod
    def class_name(cls) -> str:
        return "GitHubFileFailedEvent"
