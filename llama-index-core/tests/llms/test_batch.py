"""Unit tests for the provider-agnostic batch abstractions in core."""

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.batch import (
    BaseBatchProcessor,
    BatchJobHandle,
    BatchJobResult,
    BatchJobStatus,
    BatchRequestItem,
    BatchResultItem,
    namespaced_custom_id,
    new_run_id,
)


def test_job_status_terminal_states() -> None:
    assert BatchJobStatus.COMPLETED.is_terminal
    assert BatchJobStatus.FAILED.is_terminal
    assert BatchJobStatus.CANCELLED.is_terminal
    assert not BatchJobStatus.IN_PROGRESS.is_terminal
    assert not BatchJobStatus.VALIDATING.is_terminal


def test_namespaced_custom_id_uses_run_prefix() -> None:
    assert namespaced_custom_id("run1", 4) == "run1:4"
    assert namespaced_custom_id("run1", 4, "my-id") == "run1:my-id"


def test_new_run_id_unique() -> None:
    assert new_run_id() != new_run_id()


def test_normalize_requests_assigns_namespaced_ids() -> None:
    reqs = [
        BatchRequestItem(messages=[ChatMessage(role="user", content="a")]),
        BatchRequestItem(
            messages=[ChatMessage(role="user", content="b")], custom_id="cid"
        ),
    ]
    normalized = BaseBatchProcessor.normalize_requests(reqs, "r1")
    assert normalized[0]["custom_id"] == "r1:0"
    assert normalized[1]["custom_id"] == "r1:cid"
    assert normalized[0]["index"] == 0
    assert normalized[1]["index"] == 1


def test_index_map_covers_custom_ids() -> None:
    reqs = [
        BatchRequestItem(messages=[ChatMessage(role="user", content="a")]),
        BatchRequestItem(
            messages=[ChatMessage(role="user", content="b")], custom_id="cid"
        ),
    ]
    normalized = BaseBatchProcessor.normalize_requests(reqs, "r1")
    mapping = BaseBatchProcessor.index_map(normalized)
    assert mapping["r1:0"] == 0
    assert mapping["r1:cid"] == 1


def test_job_handle_refresh() -> None:
    handle = BatchJobHandle(job_id="j", provider="p", run_id="r", total_requests=2)
    updated = handle.refresh(BatchJobStatus.IN_PROGRESS, completed_requests=1)
    assert updated.status == BatchJobStatus.IN_PROGRESS
    assert updated.completed_requests == 1
    # Original is unchanged.
    assert handle.status == BatchJobStatus.VALIDATING


def test_result_helpers_partition_and_index() -> None:
    ok = BatchResultItem(custom_id="r:0", request_index=0, response=None)
    ok.error = None
    bad = BatchResultItem(custom_id="r:1", request_index=1, error="boom")
    result = BatchJobResult(job_id="j", provider="p", run_id="r", results=[ok, bad])
    assert result.successful() == [ok]
    assert result.failed() == [bad]
    assert result.by_index()[1] is bad
    assert result.by_custom_id()["r:0"] is ok
