import builtins
import json
from pathlib import Path
from typing import Any

import pytest

from llama_index.core.evaluation.benchmarks import hotpotqa


class _Response:
    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [b'{"answer": ', b'"ok"}']


def test_download_datasets_closes_download_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened_files: list[Any] = []
    real_open = builtins.open

    def tracking_open(*args: Any, **kwargs: Any) -> Any:
        file_obj = real_open(*args, **kwargs)
        opened_files.append(file_obj)
        return file_obj

    monkeypatch.setattr(hotpotqa, "get_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(hotpotqa.requests, "get", lambda *_, **__: _Response())
    monkeypatch.setattr(builtins, "open", tracking_open)

    paths = hotpotqa.HotpotQAEvaluator()._download_datasets()

    assert Path(paths["hotpot_dev_distractor"]).read_text() == '{"answer": "ok"}'
    assert opened_files
    assert all(file_obj.closed for file_obj in opened_files)


def test_run_closes_dataset_file_before_query_engine_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dev_distractor.json"
    dataset_path.write_text(
        json.dumps([{"question": "Where?", "answer": "Here", "context": []}])
    )
    opened_files: list[Any] = []
    real_open = builtins.open

    def tracking_open(*args: Any, **kwargs: Any) -> Any:
        file_obj = real_open(*args, **kwargs)
        opened_files.append(file_obj)
        return file_obj

    class Evaluator(hotpotqa.HotpotQAEvaluator):
        def _download_datasets(self) -> dict[str, str]:
            return {"hotpot_dev_distractor": str(dataset_path)}

    monkeypatch.setattr(builtins, "open", tracking_open)

    with pytest.raises(AssertionError, match="query_engine must be"):
        Evaluator().run(query_engine=object())  # type: ignore[arg-type]

    assert opened_files
    assert all(file_obj.closed for file_obj in opened_files)
