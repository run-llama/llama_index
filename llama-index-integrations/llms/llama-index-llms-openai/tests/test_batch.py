import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.batch import (
    BatchJobHandle,
    BatchJobStatus,
    BatchRequestItem,
    namespaced_custom_id,
)
from llama_index.llms.openai import OpenAI, OpenAIBatchProcessor


def _request(content: str, **kwargs: Any) -> BatchRequestItem:
    return BatchRequestItem(
        messages=[ChatMessage(role="user", content=content)], **kwargs
    )


def _upload(file_id: str) -> MagicMock:
    upload = MagicMock()
    upload.id = file_id
    return upload


def _batch(
    batch_id: str,
    status: str = "completed",
    output_file_id: str = "file-out",
    error_file_id: str = None,
    completed: int = 0,
    failed: int = 0,
) -> MagicMock:
    batch = MagicMock()
    batch.id = batch_id
    batch.status = status
    batch.output_file_id = output_file_id
    batch.error_file_id = error_file_id
    counts = MagicMock()
    counts.completed = completed
    counts.failed = failed
    batch.request_counts = counts
    return batch


def _content(text: str) -> MagicMock:
    content = MagicMock()
    content.text = text
    return content


def _result_line(
    custom_id: str,
    content: str,
    status_code: int = 200,
    error: Any = None,
) -> str:
    body: Dict[str, Any] = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    record: Dict[str, Any] = {
        "custom_id": custom_id,
        "response": {"status_code": status_code, "body": body},
        "error": error,
    }
    return json.dumps(record)


@patch("llama_index.llms.openai.batch.SyncOpenAI")
def test_serialize_request_uses_namespaced_custom_id(MockSyncOpenAI: MagicMock) -> None:
    proc = OpenAIBatchProcessor(llm=OpenAI(model="gpt-4o", api_key="sk-test"))
    cid = namespaced_custom_id("run1", 3)
    line = proc._serialize_request(cid, _request("hello"))
    record = json.loads(line)
    assert record["custom_id"] == "run1:3"
    assert record["method"] == "POST"
    assert record["url"] == "/v1/chat/completions"
    assert record["body"]["model"] == "gpt-4o"
    assert record["body"]["messages"][0]["role"] == "user"


@patch("llama_index.llms.openai.batch.SyncOpenAI")
def test_submit_single_model_creates_one_batch(MockSyncOpenAI: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.files.create.return_value = _upload("file-in")
    mock_client.batches.create.return_value = _batch("batch_1")
    MockSyncOpenAI.return_value = mock_client

    proc = OpenAIBatchProcessor(llm=OpenAI(model="gpt-4o", api_key="sk-test"))
    proc._client = mock_client
    handle = proc.submit([_request("a"), _request("b")])

    assert mock_client.files.create.call_count == 1
    assert mock_client.batches.create.call_count == 1
    assert handle.job_id == "batch_1"
    assert handle.provider == "openai"
    assert handle.total_requests == 2
    # JSONL payload uploaded via BytesIO.
    upload_arg = mock_client.files.create.call_args.kwargs["file"]
    payload = upload_arg.read().decode("utf-8")
    assert payload.count("\n") == 1  # two JSONL lines, one newline separator


@patch("llama_index.llms.openai.batch.SyncOpenAI")
def test_submit_mixed_models_splits_sub_batches(MockSyncOpenAI: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.files.create.side_effect = [_upload("f1"), _upload("f2")]
    mock_client.batches.create.side_effect = [
        _batch("b_a"),
        _batch("b_b"),
    ]
    MockSyncOpenAI.return_value = mock_client

    proc = OpenAIBatchProcessor(llm=OpenAI(model="gpt-4o", api_key="sk-test"))
    proc._client = mock_client
    handle = proc.submit(
        [
            _request("a"),  # default model gpt-4o
            _request("b", model="gpt-4o-mini"),
            _request("c"),  # default model gpt-4o
        ]
    )

    # Two distinct models -> two sub-batches joined under one handle.
    assert mock_client.batches.create.call_count == 2
    assert handle.job_id == "b_a,b_b"
    assert handle.total_requests == 3


@patch("llama_index.llms.openai.batch.SyncOpenAI")
def test_get_status_aggregates_sub_batches(MockSyncOpenAI: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.batches.retrieve.side_effect = [
        _batch("b_a", status="completed", completed=2),
        _batch("b_b", status="in_progress", completed=1),
    ]
    MockSyncOpenAI.return_value = mock_client

    proc = OpenAIBatchProcessor(llm=OpenAI(model="gpt-4o", api_key="sk-test"))
    proc._client = mock_client
    handle = BatchJobHandle(
        job_id="b_a,b_b", provider="openai", run_id="r", total_requests=3
    )
    updated = proc.get_status(handle)

    assert updated.status == BatchJobStatus.IN_PROGRESS
    assert updated.completed_requests == 3


@patch("llama_index.llms.openai.batch.SyncOpenAI")
def test_get_results_maps_custom_ids_and_orders(MockSyncOpenAI: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.batches.retrieve.return_value = _batch("b1")
    # Results arrive out of order; mapping must still be by request_index.
    out = "\n".join(
        [
            _result_line("run1:1", "second"),
            _result_line("run1:0", "first"),
        ]
    )
    mock_client.files.content.return_value = _content(out)
    MockSyncOpenAI.return_value = mock_client

    proc = OpenAIBatchProcessor(llm=OpenAI(model="gpt-4o", api_key="sk-test"))
    proc._client = mock_client
    handle = BatchJobHandle(
        job_id="b1", provider="openai", run_id="run1", total_requests=2
    )
    result = proc.get_results(handle)

    assert [r.request_index for r in result.results] == [0, 1]
    assert result.results[0].response.message.content == "first"
    assert result.results[1].response.message.content == "second"
    assert all(r.succeeded for r in result.results)


@patch("llama_index.llms.openai.batch.SyncOpenAI")
def test_get_results_marks_per_entry_errors(MockSyncOpenAI: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.batches.retrieve.return_value = _batch("b1")
    err_line = json.dumps(
        {
            "custom_id": "run1:0",
            "response": {
                "status_code": 500,
                "body": {"error": {"message": "upstream boom"}},
            },
            "error": None,
        }
    )
    mock_client.files.content.return_value = _content(err_line)
    MockSyncOpenAI.return_value = mock_client

    proc = OpenAIBatchProcessor(llm=OpenAI(model="gpt-4o", api_key="sk-test"))
    proc._client = mock_client
    handle = BatchJobHandle(
        job_id="b1", provider="openai", run_id="run1", total_requests=1
    )
    result = proc.get_results(handle)

    assert not result.results[0].succeeded
    assert "boom" in result.results[0].error


@patch("llama_index.llms.openai.batch.SyncOpenAI")
def test_sync_fallback_retries_failed_entry(MockSyncOpenAI: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.batches.retrieve.return_value = _batch("b1")
    err_line = json.dumps(
        {
            "custom_id": "run1:0",
            "response": {"status_code": 500, "body": {"error": {"message": "x"}}},
            "error": None,
        }
    )
    mock_client.files.content.return_value = _content(err_line)
    MockSyncOpenAI.return_value = mock_client

    llm = OpenAI(model="gpt-4o", api_key="sk-test")
    llm.chat = MagicMock(return_value="sync-response")

    proc = OpenAIBatchProcessor(llm=llm, sync_fallback=True)
    proc._client = mock_client
    proc._requests = {0: _request("redo me")}

    handle = BatchJobHandle(
        job_id="b1", provider="openai", run_id="run1", total_requests=1
    )
    result = proc.get_results(handle)

    assert result.results[0].succeeded
    assert result.results[0].response == "sync-response"


@patch("llama_index.llms.openai.batch.SyncOpenAI")
def test_cancel_calls_each_sub_batch(MockSyncOpenAI: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.batches.cancel.side_effect = [
        _batch("b_a", status="cancelling"),
        _batch("b_b", status="cancelling"),
    ]
    MockSyncOpenAI.return_value = mock_client

    proc = OpenAIBatchProcessor(llm=OpenAI(model="gpt-4o", api_key="sk-test"))
    proc._client = mock_client
    handle = BatchJobHandle(
        job_id="b_a,b_b", provider="openai", run_id="r", total_requests=2
    )
    updated = proc.cancel(handle)

    assert mock_client.batches.cancel.call_count == 2
    assert updated.status == BatchJobStatus.CANCELLING


@patch("llama_index.llms.openai.batch.SyncOpenAI")
def test_submit_empty_raises(MockSyncOpenAI: MagicMock) -> None:
    proc = OpenAIBatchProcessor(llm=OpenAI(model="gpt-4o", api_key="sk-test"))
    with pytest.raises(ValueError):
        proc.submit([])
