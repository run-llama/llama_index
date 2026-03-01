"""Tests for validate_json utility in llama-index-finetuning."""

import json
import os
import tempfile
from typing import List

import pytest


def _write_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_valid_record(
    include_system: bool = True, include_user: bool = True
) -> dict:
    messages = []
    if include_system:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    if include_user:
        messages.append({"role": "user", "content": "What is 2+2?"})
    messages.append({"role": "assistant", "content": "4"})
    return {"messages": messages}


# ---------------------------------------------------------------------------
# validate_json – happy-path (no format errors, function completes without raising)
# ---------------------------------------------------------------------------


def test_validate_json_valid_file(capsys: pytest.CaptureFixture) -> None:
    """validate_json should accept a well-formed JSONL file without raising."""
    from llama_index.finetuning.openai.validate_json import validate_json

    records = [_make_valid_record() for _ in range(5)]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        for r in records:
            tmp.write(json.dumps(r) + "\n")
        tmp_path = tmp.name

    try:
        validate_json(tmp_path)
        captured = capsys.readouterr()
        assert "No errors found" in captured.out
        assert "Num examples: 5" in captured.out
    finally:
        os.unlink(tmp_path)


def test_validate_json_missing_assistant_message(capsys: pytest.CaptureFixture) -> None:
    """validate_json should detect records without an assistant message."""
    from llama_index.finetuning.openai.validate_json import validate_json

    record = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello?"},
        ]
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write(json.dumps(record) + "\n")
        tmp_path = tmp.name

    try:
        validate_json(tmp_path)
        captured = capsys.readouterr()
        assert "example_missing_assistant_message" in captured.out
    finally:
        os.unlink(tmp_path)


def test_validate_json_unrecognized_role(capsys: pytest.CaptureFixture) -> None:
    """validate_json should flag messages with roles other than system/user/assistant."""
    from llama_index.finetuning.openai.validate_json import validate_json

    record = {
        "messages": [
            {"role": "moderator", "content": "Keep it civil."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write(json.dumps(record) + "\n")
        tmp_path = tmp.name

    try:
        validate_json(tmp_path)
        captured = capsys.readouterr()
        assert "unrecognized_role" in captured.out
    finally:
        os.unlink(tmp_path)


def test_validate_json_missing_content(capsys: pytest.CaptureFixture) -> None:
    """validate_json should flag messages whose content is empty or absent."""
    from llama_index.finetuning.openai.validate_json import validate_json

    record = {
        "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Hello"},
        ]
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write(json.dumps(record) + "\n")
        tmp_path = tmp.name

    try:
        validate_json(tmp_path)
        captured = capsys.readouterr()
        assert "missing_content" in captured.out
    finally:
        os.unlink(tmp_path)


def test_validate_json_missing_messages_list(capsys: pytest.CaptureFixture) -> None:
    """validate_json should flag records that have no 'messages' key."""
    from llama_index.finetuning.openai.validate_json import validate_json

    record = {"not_messages": []}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write(json.dumps(record) + "\n")
        tmp_path = tmp.name

    try:
        validate_json(tmp_path)
        captured = capsys.readouterr()
        assert "missing_messages_list" in captured.out
    finally:
        os.unlink(tmp_path)


def test_validate_json_token_statistics_printed(capsys: pytest.CaptureFixture) -> None:
    """validate_json should print token distribution statistics for valid data."""
    from llama_index.finetuning.openai.validate_json import validate_json

    records = [_make_valid_record() for _ in range(10)]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        for r in records:
            tmp.write(json.dumps(r) + "\n")
        tmp_path = tmp.name

    try:
        validate_json(tmp_path)
        captured = capsys.readouterr()
        assert "Distribution of num_messages_per_example" in captured.out
        assert "Distribution of num_total_tokens_per_example" in captured.out
        assert "Distribution of num_assistant_tokens_per_example" in captured.out
    finally:
        os.unlink(tmp_path)


def test_validate_json_missing_system_user_counts(
    capsys: pytest.CaptureFixture,
) -> None:
    """validate_json should count examples missing system/user messages."""
    from llama_index.finetuning.openai.validate_json import validate_json

    # Record missing system message
    record_no_system = _make_valid_record(include_system=False)
    # Valid record
    record_valid = _make_valid_record()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write(json.dumps(record_no_system) + "\n")
        tmp.write(json.dumps(record_valid) + "\n")
        tmp_path = tmp.name

    try:
        validate_json(tmp_path)
        captured = capsys.readouterr()
        assert "Num examples missing system message: 1" in captured.out
        assert "Num examples missing user message: 0" in captured.out
    finally:
        os.unlink(tmp_path)
