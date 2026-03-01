"""Tests for OpenAIFinetuneEngine with mocked OpenAI client."""

import json
import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(tmp_path: str, **kwargs: Any):
    """Create an OpenAIFinetuneEngine with a mocked OpenAI client."""
    from llama_index.finetuning.openai.base import OpenAIFinetuneEngine

    with patch("llama_index.finetuning.openai.base.SyncOpenAI") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        engine = OpenAIFinetuneEngine(
            base_model="gpt-3.5-turbo",
            data_path=tmp_path,
            validate_json=False,
            **kwargs,
        )
        engine._client = mock_client
    return engine, mock_client


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


def test_engine_constructor_sets_attributes() -> None:
    """Engine should store base_model, data_path, and other init params."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        engine, _ = _make_engine(tmp_path, verbose=True)
        assert engine.base_model == "gpt-3.5-turbo"
        assert engine.data_path == tmp_path
        assert engine._verbose is True
    finally:
        os.unlink(tmp_path)


def test_engine_constructor_with_start_job_id() -> None:
    """When start_job_id is provided, the engine should fetch the job."""
    from llama_index.finetuning.openai.base import OpenAIFinetuneEngine

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        mock_job = MagicMock()
        mock_job.id = "ftjob-123"

        with patch("llama_index.finetuning.openai.base.SyncOpenAI") as MockClient:
            mock_client = MagicMock()
            mock_client.fine_tuning.jobs.retrieve.return_value = mock_job
            MockClient.return_value = mock_client

            engine = OpenAIFinetuneEngine(
                base_model="gpt-3.5-turbo",
                data_path=tmp_path,
                validate_json=False,
                start_job_id="ftjob-123",
            )
            assert engine._start_job is not None
            assert engine._start_job.id == "ftjob-123"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# finetune() tests
# ---------------------------------------------------------------------------


def test_finetune_uploads_file_and_creates_job() -> None:
    """finetune() should upload the file and create a fine-tuning job."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write('{"messages": []}\n')
        tmp_path = tmp.name

    try:
        engine, mock_client = _make_engine(tmp_path, validate_json=False)

        mock_file = MagicMock()
        mock_file.id = "file-abc"
        mock_client.files.create.return_value = mock_file

        mock_job = MagicMock()
        mock_job.id = "ftjob-xyz"
        mock_client.fine_tuning.jobs.create.return_value = mock_job

        engine.finetune()

        mock_client.files.create.assert_called_once()
        mock_client.fine_tuning.jobs.create.assert_called_once_with(
            training_file="file-abc", model="gpt-3.5-turbo"
        )
        assert engine._start_job is not None
    finally:
        os.unlink(tmp_path)


def test_finetune_calls_validate_json_when_enabled(capsys: pytest.CaptureFixture) -> None:
    """When validate_json=True, finetune() should call the validator."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        record = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        tmp.write(json.dumps(record) + "\n")
        tmp_path = tmp.name

    try:
        from llama_index.finetuning.openai.base import OpenAIFinetuneEngine

        with patch("llama_index.finetuning.openai.base.SyncOpenAI") as MockClient:
            mock_client = MagicMock()
            mock_file = MagicMock()
            mock_file.id = "file-val"
            mock_client.files.create.return_value = mock_file
            mock_job = MagicMock()
            mock_client.fine_tuning.jobs.create.return_value = mock_job
            MockClient.return_value = mock_client

            engine = OpenAIFinetuneEngine(
                base_model="gpt-3.5-turbo",
                data_path=tmp_path,
                validate_json=True,
            )
            engine.finetune()

        captured = capsys.readouterr()
        # validate_json prints "No errors found" for valid data
        assert "No errors found" in captured.out
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# get_current_job() tests
# ---------------------------------------------------------------------------


def test_get_current_job_raises_without_finetune() -> None:
    """get_current_job() should raise ValueError if finetune() was never called."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        engine, _ = _make_engine(tmp_path)
        with pytest.raises(ValueError, match="Must call finetune\\(\\) first"):
            engine.get_current_job()
    finally:
        os.unlink(tmp_path)


def test_get_current_job_after_finetune() -> None:
    """get_current_job() should return the job retrieved from the API."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write("{}\n")
        tmp_path = tmp.name

    try:
        engine, mock_client = _make_engine(tmp_path)

        mock_file = MagicMock()
        mock_file.id = "file-1"
        mock_client.files.create.return_value = mock_file

        mock_start_job = MagicMock()
        mock_start_job.id = "ftjob-1"
        mock_client.fine_tuning.jobs.create.return_value = mock_start_job

        mock_current_job = MagicMock()
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_current_job

        engine.finetune()
        job = engine.get_current_job()
        assert job is mock_current_job
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# get_finetuned_model() tests
# ---------------------------------------------------------------------------


def test_get_finetuned_model_raises_if_no_model_id() -> None:
    """get_finetuned_model() should raise if fine_tuned_model is None."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write("{}\n")
        tmp_path = tmp.name

    try:
        engine, mock_client = _make_engine(tmp_path)

        mock_file = MagicMock()
        mock_file.id = "file-2"
        mock_client.files.create.return_value = mock_file

        mock_start_job = MagicMock()
        mock_start_job.id = "ftjob-2"
        mock_client.fine_tuning.jobs.create.return_value = mock_start_job

        mock_current_job = MagicMock()
        mock_current_job.id = "ftjob-2"
        mock_current_job.status = "running"
        mock_current_job.fine_tuned_model = None
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_current_job

        engine.finetune()

        with pytest.raises(ValueError, match="does not have a finetuned model id"):
            engine.get_finetuned_model()
    finally:
        os.unlink(tmp_path)


def test_get_finetuned_model_raises_if_not_succeeded() -> None:
    """get_finetuned_model() should raise if job status is not 'succeeded'."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write("{}\n")
        tmp_path = tmp.name

    try:
        engine, mock_client = _make_engine(tmp_path)

        mock_file = MagicMock()
        mock_file.id = "file-3"
        mock_client.files.create.return_value = mock_file

        mock_start_job = MagicMock()
        mock_start_job.id = "ftjob-3"
        mock_client.fine_tuning.jobs.create.return_value = mock_start_job

        mock_current_job = MagicMock()
        mock_current_job.id = "ftjob-3"
        mock_current_job.status = "failed"
        mock_current_job.fine_tuned_model = "ft:gpt-3.5-turbo:my-model"
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_current_job

        engine.finetune()

        with pytest.raises(ValueError, match="has status failed"):
            engine.get_finetuned_model()
    finally:
        os.unlink(tmp_path)


def test_get_finetuned_model_returns_openai_llm() -> None:
    """get_finetuned_model() should return an OpenAI LLM when job succeeded."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write("{}\n")
        tmp_path = tmp.name

    try:
        engine, mock_client = _make_engine(tmp_path)

        mock_file = MagicMock()
        mock_file.id = "file-4"
        mock_client.files.create.return_value = mock_file

        mock_start_job = MagicMock()
        mock_start_job.id = "ftjob-4"
        mock_client.fine_tuning.jobs.create.return_value = mock_start_job

        ft_model_id = "ft:gpt-3.5-turbo:org:custom:abc123"
        mock_current_job = MagicMock()
        mock_current_job.id = "ftjob-4"
        mock_current_job.status = "succeeded"
        mock_current_job.fine_tuned_model = ft_model_id
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_current_job

        engine.finetune()

        from llama_index.llms.openai import OpenAI

        with patch("llama_index.finetuning.openai.base.OpenAI") as MockOpenAI:
            mock_llm = MagicMock(spec=OpenAI)
            MockOpenAI.return_value = mock_llm
            model = engine.get_finetuned_model()

        MockOpenAI.assert_called_once_with(model=ft_model_id)
        assert model is mock_llm
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# from_finetuning_handler() class method
# ---------------------------------------------------------------------------


def test_from_finetuning_handler() -> None:
    """from_finetuning_handler should save events and create a new engine."""
    from llama_index.finetuning.openai.base import OpenAIFinetuneEngine
    from llama_index.finetuning.callbacks.finetuning_handler import (
        OpenAIFineTuningHandler,
    )

    handler = MagicMock(spec=OpenAIFineTuningHandler)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with patch("llama_index.finetuning.openai.base.SyncOpenAI") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            engine = OpenAIFinetuneEngine.from_finetuning_handler(
                finetuning_handler=handler,
                base_model="gpt-3.5-turbo",
                data_path=tmp_path,
                validate_json=False,
            )

        handler.save_finetuning_events.assert_called_once_with(tmp_path)
        assert isinstance(engine, OpenAIFinetuneEngine)
        assert engine.base_model == "gpt-3.5-turbo"
    finally:
        os.unlink(tmp_path)
