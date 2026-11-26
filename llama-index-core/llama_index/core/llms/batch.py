"""Provider-agnostic abstractions for bulk/batch LLM calls.

Several providers expose a *batch* endpoint (e.g. OpenAI's ``/v1/batches`` and
Anthropic's Message Batches) that runs a large set of requests asynchronously at
roughly half the cost of the synchronous path. These abstractions let
integrations share a single request/job/result contract while keeping each
provider's serialization and transport details inside its own package.

The scope here is intentionally narrow: request serialization, job submission,
status polling, cancellation and per-request result mapping. A provider
implements :class:`BaseBatchProcessor` and returns the provider-neutral types
defined below.
"""

import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.bridge.pydantic import BaseModel, ConfigDict, Field


class BatchJobStatus(str, Enum):
    """Normalized lifecycle state of a provider batch job."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether the job can no longer transition to another state."""
        return self in {
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.EXPIRED,
            BatchJobStatus.CANCELLED,
        }


class BatchRequestItem(BaseModel):
    """A single request to run inside a batch job.

    ``custom_id`` is optional; when omitted the processor assigns a collision-free
    id (``{run_id}:{index}``) so results can always be mapped back to their
    originating request, even across runs that reuse the same prompts.
    """

    custom_id: Optional[str] = Field(
        default=None,
        description="Optional caller-supplied id used to correlate results.",
    )
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="Chat messages that make up the request.",
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Model to run for this request. When None the processor's default "
            "model is used. Providers that require one model per input file "
            "sub-split the batch internally."
        ),
    )
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific per-request parameters.",
    )


class BatchJobHandle(BaseModel):
    """A handle to a submitted batch job.

    ``job_id`` is the provider-side identifier; ``run_id`` is the
    processor-generated correlation id used to namespace ``custom_id`` values
    so identical prompts across runs cannot collide.
    """

    job_id: str = Field(description="Provider-side batch job identifier.")
    provider: str = Field(description="Name of the provider, e.g. 'openai'.")
    run_id: str = Field(description="Processor-side correlation id for this run.")
    status: BatchJobStatus = Field(default=BatchJobStatus.VALIDATING)
    total_requests: int = Field(default=0, ge=0)
    completed_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def refresh(self, status: "BatchJobStatus", **counts: int) -> "BatchJobHandle":
        """Return a copy of this handle with an updated status and counters."""
        data = self.model_dump()
        data["status"] = status
        data.update(counts)
        return BatchJobHandle(**data)


class BatchResultItem(BaseModel):
    """The outcome of a single :class:`BatchRequestItem`.

    Exactly one of ``response`` / ``error`` is populated for a finished request;
    ``custom_id`` is always the provider-namespaced id echoed back by the batch.
    """

    custom_id: str = Field(description="Namespaced custom_id echoed by the provider.")
    request_index: int = Field(description="Index of the source request in the batch.")
    response: Optional[ChatResponse] = Field(
        default=None, description="Successful chat response, if any."
    )
    error: Optional[str] = Field(
        default=None, description="Provider error message, if the request failed."
    )
    raw: Optional[Any] = Field(
        default=None, description="Raw provider result payload, when useful."
    )

    @property
    def succeeded(self) -> bool:
        return self.error is None


class BatchJobResult(BaseModel):
    """Aggregated results for a finished batch job."""

    job_id: str
    provider: str
    run_id: str
    results: List[BatchResultItem] = Field(default_factory=list)

    def successful(self) -> List[BatchResultItem]:
        return [r for r in self.results if r.succeeded]

    def failed(self) -> List[BatchResultItem]:
        return [r for r in self.results if not r.succeeded]

    def by_index(self) -> Dict[int, BatchResultItem]:
        return {r.request_index: r for r in self.results}

    def by_custom_id(self) -> Dict[str, BatchResultItem]:
        return {r.custom_id: r for r in self.results}


def new_run_id() -> str:
    """Return a fresh correlation id used to namespace custom_ids."""
    return uuid.uuid4().hex


def namespaced_custom_id(run_id: str, index: int, custom_id: Optional[str] = None) -> str:
    """Compose a collision-free custom_id for a batch request.

    Namespacing under ``run_id`` guarantees uniqueness across runs even when the
    caller reuses prompts or ids. The provider receives this value verbatim and
    echoes it back in the result file.
    """
    if custom_id:
        return f"{run_id}:{custom_id}"
    return f"{run_id}:{index}"


class BaseBatchProcessor(BaseModel, ABC):
    """Submit and resolve provider batch jobs.

    Implementations live in the provider's own integration package and translate
    :class:`BatchRequestItem` sequences to the provider's wire format, handle
    any provider constraints (e.g. one model per input file) internally, and map
    provider results back onto :class:`BatchResultItem`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name reported on job handles and results."""

    @abstractmethod
    def submit(self, requests: Sequence[BatchRequestItem]) -> BatchJobHandle:
        """Submit ``requests`` as a batch job and return its handle."""

    @abstractmethod
    def get_status(self, handle: BatchJobHandle) -> BatchJobHandle:
        """Refresh and return the job's current status and counters."""

    @abstractmethod
    def get_results(self, handle: BatchJobHandle) -> BatchJobResult:
        """Fetch per-request results for a finished job."""

    @abstractmethod
    def cancel(self, handle: BatchJobHandle) -> BatchJobHandle:
        """Request cancellation of an in-flight job."""

    # ---- convenience async wrappers -------------------------------------------------

    async def asubmit(self, requests: Sequence[BatchRequestItem]) -> BatchJobHandle:
        """Async variant of :meth:`submit` (default runs the sync call)."""
        return self.submit(requests)

    async def aget_status(self, handle: BatchJobHandle) -> BatchJobHandle:
        """Async variant of :meth:`get_status`."""
        return self.get_status(handle)

    async def aget_results(self, handle: BatchJobHandle) -> BatchJobResult:
        """Async variant of :meth:`get_results`."""
        return self.get_results(handle)

    async def acancel(self, handle: BatchJobHandle) -> BatchJobHandle:
        """Async variant of :meth:`cancel`."""
        return self.cancel(handle)

    # ---- helpers --------------------------------------------------------------------

    @staticmethod
    def normalize_requests(
        requests: Sequence[BatchRequestItem], run_id: str
    ) -> List[Dict[str, Any]]:
        """Assign namespaced custom_ids and group requests by effective model.

        Returns a list of dicts, each with ``index`` (position in ``requests``),
        ``custom_id`` (namespaced), ``model`` (effective model or None) and the
        source ``item``. Providers that require a single model per file partition
        on the ``model`` key.
        """
        normalized: List[Dict[str, Any]] = []
        for index, item in enumerate(requests):
            normalized.append(
                {
                    "index": index,
                    "custom_id": namespaced_custom_id(run_id, index, item.custom_id),
                    "model": item.model,
                    "item": item,
                }
            )
        return normalized

    @staticmethod
    def index_map(normalized: List[Dict[str, Any]]) -> Dict[str, int]:
        """Build a ``custom_id -> request index`` lookup for a normalized batch.

        Stored on the job handle's ``metadata`` so result mapping survives even
        when callers supply their own ``custom_id`` values (which cannot be
        parsed back into an index).
        """
        return {e["custom_id"]: e["index"] for e in normalized}
