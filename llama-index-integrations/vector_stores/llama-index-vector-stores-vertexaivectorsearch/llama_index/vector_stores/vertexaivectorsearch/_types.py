"""Utility types used for the VertexAI vector store."""

from dataclasses import dataclass, field

from typing_extensions import override


class VertexAIError(Exception):
    """Base Vertex AI exception type."""


class VertexAIInputError(VertexAIError, ValueError):
    """Errors related to invalid input data."""


class VertexAIQueryError(VertexAIError):
    """Errors during query operations."""


@dataclass(kw_only=True, slots=True)
class AddBatchResult:
    """Container for tracking individual or batch results for add/update operations."""

    added_ids: list[str] = field(default_factory=list)
    updated_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)
    exceptions: list[Exception] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """Indicates whether the batch operation succeeded."""
        return not self.failed_ids and not self.exceptions

    def __add__(self, other: "AddBatchResult") -> "AddBatchResult":
        """Combine the properties of two :py:class:`_AddResult` objects."""
        return AddBatchResult(
            added_ids=self.added_ids + other.added_ids,
            updated_ids=self.updated_ids + other.updated_ids,
            failed_ids=self.failed_ids + other.failed_ids,
            exceptions=[*self.exceptions, *other.exceptions],
        )

    @override
    def __str__(self) -> str:
        """Returns a summary count line for logging purposes."""
        return (
            f"added={len(self.added_ids)}, updated={len(self.updated_ids)}, "
            f"failed={len(self.failed_ids)}, exceptions={self.exceptions}"
        )


class VertexAIAddError(VertexAIError):
    """Raised for errors when adding or updating content in a vector store."""

    def __init__(self, result: AddBatchResult) -> None:
        """Initialize the exception."""
        super().__init__(f"Failed to add/update all requested objects, {result!s}")
        self.result = result


@dataclass(kw_only=True, slots=True)
class DeleteBatchResult:
    """Container for tracking individual or batch results for delete operations."""

    deleted: int = 0
    failed: int = 0
    not_found: int = 0
    exceptions: list[Exception] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """Indicates whether the batch operation succeeded."""
        return not self.failed and not self.not_found and not self.exceptions

    def __add__(self, other: "DeleteBatchResult") -> "DeleteBatchResult":
        """Combine the properties of two :py:class:`_DeleteResult` objects."""
        return DeleteBatchResult(
            deleted=self.deleted + other.deleted,
            failed=self.failed + other.failed,
            not_found=self.not_found + other.not_found,
            exceptions=[*self.exceptions, *other.exceptions],
        )

    @override
    def __str__(self) -> str:
        """Returns a summary count line for logging purposes."""
        return (
            f"deleted={self.deleted}, not_found={self.not_found}, "
            f"failed={self.failed}, exceptions={self.exceptions}"
        )


class VertexAIDeleteError(VertexAIError):
    """Raised for errors when indexing content into a vector store."""

    def __init__(self, result: DeleteBatchResult) -> None:
        """Initialize the exception."""
        super().__init__(f"Failed to delete all target data objects, {result!s}")
        self.result = result
