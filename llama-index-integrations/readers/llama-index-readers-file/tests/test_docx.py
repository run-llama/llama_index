import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from llama_index.readers.file.docs import base


@pytest.mark.parametrize("path_type", [str, PurePosixPath])
def test_docx_preserves_remote_paths_on_windows(monkeypatch, path_type) -> None:
    fs = MemoryFileSystem()
    file_path = "docx-reader/reports/example.docx"
    fs.pipe(file_path, b"document contents")
    process = Mock(side_effect=lambda stream: stream.read().decode())
    monkeypatch.setitem(sys.modules, "docx2txt", SimpleNamespace(process=process))
    monkeypatch.setattr(base, "Path", PureWindowsPath)

    documents = base.DocxReader().load_data(
        path_type(file_path), extra_info={"source": "remote"}, fs=fs
    )

    assert len(documents) == 1
    assert documents[0].text == "document contents"
    assert documents[0].metadata == {"file_name": "example.docx", "source": "remote"}
    process.assert_called_once()


@pytest.mark.parametrize("path_type", [str, Path])
def test_docx_preserves_local_paths(monkeypatch, tmp_path, path_type) -> None:
    file_path = tmp_path / "example.docx"
    process = Mock(return_value="local document")
    monkeypatch.setitem(sys.modules, "docx2txt", SimpleNamespace(process=process))

    documents = base.DocxReader().load_data(path_type(file_path))

    process.assert_called_once_with(file_path)
    assert documents[0].text == "local document"
    assert documents[0].metadata == {"file_name": "example.docx"}
