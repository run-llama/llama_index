"""OpenAI Batch API support for bulk LLM calls.

OpenAI's ``/v1/batches`` endpoint runs a set of chat-completion requests
asynchronously at ~50% of the synchronous cost. This module adapts the
provider-agnostic :class:`~llama_index.core.llms.batch.BaseBatchProcessor`
contract to OpenAI:

- Requests are serialized to the OpenAI chat-completions JSONL format and
  uploaded in-memory via ``io.BytesIO`` (no temp files on disk).
- OpenAI requires every request in an input file to target the same model, so a
  mixed-model batch is transparently sub-split into one sub-batch per model and
  tracked under a single :class:`BatchJobHandle` (comma-joined ``job_id``).
- ``custom_id`` is namespaced ``{run_id}:{index|custom_id}`` so results map back
  to the originating request without collisions across runs.
- Optionally, requests that error inside a completed batch can be retried
  through the synchronous chat path (``sync_fallback``).
"""

import io
import json
import time
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    TextBlock,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.batch import (
    BaseBatchProcessor,
    BatchJobHandle,
    BatchJobResult,
    BatchJobStatus,
    BatchRequestItem,
    BatchResultItem,
    new_run_id,
)
from llama_index.llms.openai.base import OpenAI, SyncOpenAI
from llama_index.llms.openai.utils import to_openai_message_dicts

# OpenAI batch states -> normalized status.
_OPENAI_STATUS_MAP: Dict[str, BatchJobStatus] = {
    "validating": BatchJobStatus.VALIDATING,
    "in_progress": BatchJobStatus.IN_PROGRESS,
    "finalizing": BatchJobStatus.FINALIZING,
    "completed": BatchJobStatus.COMPLETED,
    "failed": BatchJobStatus.FAILED,
    "expired": BatchJobStatus.EXPIRED,
    "cancelling": BatchJobStatus.CANCELLING,
    "cancelled": BatchJobStatus.CANCELLED,
}

_TERMINAL = {
    BatchJobStatus.COMPLETED,
    BatchJobStatus.FAILED,
    BatchJobStatus.EXPIRED,
    BatchJobStatus.CANCELLED,
}


class OpenAIBatchProcessor(BaseBatchProcessor):
    """Submit and resolve OpenAI ``/v1/batches`` jobs.

    Args:
        llm: The :class:`OpenAI` LLM used for default model, credentials and the
            optional synchronous fallback.
        sync_fallback: When True, requests that error inside a finished batch are
            retried once through the synchronous ``llm.chat`` path.
        poll_interval: Seconds between ``get_status`` polls in :meth:`wait`.

    Examples:
        ```python
        from llama_index.llms.openai import OpenAI, OpenAIBatchProcessor
        from llama_index.core.llms.batch import BatchRequestItem
        from llama_index.core.base.llms.types import ChatMessage

        llm = OpenAI(model="gpt-4o-mini")
        processor = OpenAIBatchProcessor(llm=llm)
        handle = processor.submit([
            BatchRequestItem(messages=[ChatMessage(role="user", content="Hi")])
        ])
        handle = processor.wait(handle)
        for item in processor.get_results(handle).results:
            print(item.request_index, item.response.message.content)
        ```
    """

    llm: OpenAI = Field(description="OpenAI LLM used for model and credentials.")
    sync_fallback: bool = Field(
        default=False,
        description="Retry failed batch entries via the synchronous chat path.",
    )
    poll_interval: float = Field(
        default=10.0, ge=0.0, description="Seconds between status polls in wait()."
    )

    _client: Optional[SyncOpenAI] = PrivateAttr(default=None)
    _requests: Dict[int, BatchRequestItem] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        llm: Optional[OpenAI] = None,
        sync_fallback: bool = False,
        poll_interval: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            llm=llm or OpenAI(),
            sync_fallback=sync_fallback,
            poll_interval=poll_interval,
            **kwargs,
        )

    @property
    def provider(self) -> str:
        return "openai"

    def _get_client(self) -> SyncOpenAI:
        if self._client is None:
            # Reuse the LLM's authenticated client when available.
            self._client = self.llm._get_client()
        return self._client

    # ---- serialization ------------------------------------------------------------

    def _effective_model(self, item: BatchRequestItem) -> str:
        return item.model or self.llm.model

    def _serialize_request(self, custom_id: str, item: BatchRequestItem) -> str:
        message_dicts = to_openai_message_dicts(item.messages, model=item.model)
        body: Dict[str, Any] = {
            "model": self._effective_model(item),
            "messages": message_dicts,
        }
        # Allow per-request overrides (max_tokens, temperature, tools, ...).
        for key, value in item.extra_params.items():
            body[key] = value
        line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    # ---- lifecycle ----------------------------------------------------------------

    def submit(self, requests: Sequence[BatchRequestItem]) -> BatchJobHandle:
        if not requests:
            raise ValueError("submit() requires at least one BatchRequestItem")

        run_id = new_run_id()
        normalized = BaseBatchProcessor.normalize_requests(requests, run_id)

        # Keep the source requests so per-entry sync fallback can re-send them.
        self._requests = {e["index"]: e["item"] for e in normalized}
        index_map = BaseBatchProcessor.index_map(normalized)

        # Group by effective model: OpenAI requires one model per input file.
        by_model: Dict[str, List[Dict[str, Any]]] = {}
        for entry in normalized:
            model = self._effective_model(entry["item"])
            by_model.setdefault(model, []).append(entry)

        client = self._get_client()
        job_ids: List[str] = []
        for entries in by_model.values():
            lines = [
                self._serialize_request(e["custom_id"], e["item"]) for e in entries
            ]
            payload = "\n".join(lines).encode("utf-8")
            upload = client.files.create(
                file=io.BytesIO(payload), purpose="batch"
            )
            batch = client.batches.create(
                input_file_id=upload.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            job_ids.append(batch.id)

        return BatchJobHandle(
            job_id=",".join(job_ids),
            provider=self.provider,
            run_id=run_id,
            status=BatchJobStatus.VALIDATING,
            total_requests=len(requests),
            metadata={"index_map": index_map},
        )

    def _job_ids(self, handle: BatchJobHandle) -> List[str]:
        return [j for j in handle.job_id.split(",") if j]

    def get_status(self, handle: BatchJobHandle) -> BatchJobHandle:
        client = self._get_client()
        statuses: List[BatchJobStatus] = []
        completed = 0
        failed = 0
        for job_id in self._job_ids(handle):
            batch = client.batches.retrieve(job_id)
            statuses.append(
                _OPENAI_STATUS_MAP.get(batch.status, BatchJobStatus.IN_PROGRESS)
            )
            counts = getattr(batch, "request_counts", None)
            if counts is not None:
                completed += getattr(counts, "completed", 0) or 0
                failed += getattr(counts, "failed", 0) or 0

        status = self._aggregate_status(statuses)
        return handle.refresh(
            status, completed_requests=completed, failed_requests=failed
        )

    @staticmethod
    def _aggregate_status(statuses: List[BatchJobStatus]) -> BatchJobStatus:
        """Collapse per-sub-batch statuses into one handle-level status."""
        if not statuses:
            return BatchJobStatus.VALIDATING
        if all(s == BatchJobStatus.COMPLETED for s in statuses):
            return BatchJobStatus.COMPLETED
        if any(s == BatchJobStatus.FAILED for s in statuses):
            return BatchJobStatus.FAILED
        if any(s == BatchJobStatus.CANCELLING for s in statuses):
            return BatchJobStatus.CANCELLING
        if all(s in _TERMINAL for s in statuses):
            # Mixed terminal states that aren't all-completed/all-failed.
            if any(s == BatchJobStatus.CANCELLED for s in statuses):
                return BatchJobStatus.CANCELLED
            return BatchJobStatus.EXPIRED
        if any(s == BatchJobStatus.IN_PROGRESS for s in statuses):
            return BatchJobStatus.IN_PROGRESS
        if any(s == BatchJobStatus.FINALIZING for s in statuses):
            return BatchJobStatus.FINALIZING
        return BatchJobStatus.VALIDATING

    def get_results(self, handle: BatchJobHandle) -> BatchJobResult:
        client = self._get_client()
        results: List[BatchResultItem] = []

        for job_id in self._job_ids(handle):
            batch = client.batches.retrieve(job_id)
            output_file_id = getattr(batch, "output_file_id", None)
            error_file_id = getattr(batch, "error_file_id", None)

            for file_id in (output_file_id, error_file_id):
                if not file_id:
                    continue
                content = client.files.content(file_id)
                text = content.text if hasattr(content, "text") else content.read()
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                for line in str(text).splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    results.append(self._parse_result_line(line, handle))

        # Map back to source request index and apply optional sync fallback.
        results = self._finalize_results(handle, results)
        return BatchJobResult(
            job_id=handle.job_id,
            provider=self.provider,
            run_id=handle.run_id,
            results=results,
        )

    def _parse_result_line(
        self, line: str, handle: Optional[BatchJobHandle] = None
    ) -> BatchResultItem:
        record = json.loads(line)
        custom_id = record.get("custom_id", "")
        index = self._index_for(custom_id, handle)

        error = record.get("error")
        response_obj = record.get("response") or {}
        status_code = response_obj.get("status_code")
        body = response_obj.get("body") or {}

        if error or (status_code and status_code >= 400):
            message = ""
            if error:
                message = error.get("message", json.dumps(error))
            elif isinstance(body, dict) and body.get("error"):
                message = body["error"].get("message", json.dumps(body["error"]))
            return BatchResultItem(
                custom_id=custom_id,
                request_index=index,
                error=message or "unknown batch error",
                raw=record,
            )

        # Successful chat completion payload.
        chat_response = self._chat_response_from_body(body)
        return BatchResultItem(
            custom_id=custom_id,
            request_index=index,
            response=chat_response,
            raw=record,
        )

    @staticmethod
    def _index_for(custom_id: str, handle: Optional[BatchJobHandle]) -> int:
        """Resolve the source request index for a result's custom_id.

        Prefers the ``index_map`` recorded on the job handle at submit time
        (robust to caller-supplied custom_ids); falls back to parsing the
        trailing ``{index}`` segment of a namespaced id.
        """
        if handle is not None:
            index_map = (handle.metadata or {}).get("index_map") or {}
            if custom_id in index_map:
                return index_map[custom_id]
        try:
            return int(custom_id.rsplit(":", 1)[-1])
        except (ValueError, IndexError):
            return -1

    def _chat_response_from_body(self, body: Dict[str, Any]) -> ChatResponse:
        choices = body.get("choices") or []
        message_dict = choices[0].get("message", {}) if choices else {}
        content = message_dict.get("content")
        role = message_dict.get("role", "assistant")
        try:
            role_enum = MessageRole(role)
        except ValueError:
            role_enum = MessageRole.ASSISTANT
        blocks: List[Any] = []
        if content:
            blocks.append(TextBlock(text=content))
        message = ChatMessage(role=role_enum, blocks=blocks)
        usage = body.get("usage") or {}
        additional_kwargs = {
            k: usage[k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            if k in usage
        }
        return ChatResponse(
            message=message,
            raw=body,
            additional_kwargs=additional_kwargs,
        )

    def _finalize_results(
        self, handle: BatchJobHandle, results: List[BatchResultItem]
    ) -> List[BatchResultItem]:
        # Entries that errored inside the batch still carry their namespaced
        # custom_id, so request_index is already correct.
        if self.sync_fallback:
            results = [
                self._sync_fallback(item) if not item.succeeded else item
                for item in results
            ]
        # Order by request index for deterministic downstream consumption.
        return sorted(results, key=lambda r: r.request_index)

    def _sync_fallback(self, item: BatchResultItem) -> BatchResultItem:
        """Re-run a failed request synchronously; keep the error on failure."""
        source = self._requests.get(item.request_index)
        if source is None:
            return item
        try:
            response = self.llm.chat(source.messages)
        except Exception:
            return item
        return BatchResultItem(
            custom_id=item.custom_id,
            request_index=item.request_index,
            response=response,
            raw=item.raw,
        )

    def cancel(self, handle: BatchJobHandle) -> BatchJobHandle:
        client = self._get_client()
        statuses: List[BatchJobStatus] = []
        for job_id in self._job_ids(handle):
            batch = client.batches.cancel(job_id)
            statuses.append(
                _OPENAI_STATUS_MAP.get(batch.status, BatchJobStatus.CANCELLING)
            )
        return handle.refresh(self._aggregate_status(statuses))

    # ---- convenience ----------------------------------------------------------------

    def wait(
        self, handle: BatchJobHandle, timeout: Optional[float] = None
    ) -> BatchJobHandle:
        """Poll until the job reaches a terminal state or ``timeout`` elapses."""
        start = time.monotonic()
        current = self.get_status(handle)
        while not current.status.is_terminal:
            if timeout is not None and (time.monotonic() - start) > timeout:
                raise TimeoutError(
                    f"Batch job {handle.job_id} did not finish within {timeout}s"
                )
            time.sleep(self.poll_interval)
            current = self.get_status(current)
        return current

    def run(
        self,
        requests: Sequence[BatchRequestItem],
        timeout: Optional[float] = None,
    ) -> BatchJobResult:
        """Submit, wait for completion and return results in one call."""
        handle = self.submit(requests)
        handle = self.wait(handle, timeout=timeout)
        return self.get_results(handle)
