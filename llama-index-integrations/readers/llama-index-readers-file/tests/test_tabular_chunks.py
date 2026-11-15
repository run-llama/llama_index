from pathlib import Path
from typing import Any

from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
import pandas as pd
import pytest

from llama_index.readers.file.tabular import PandasCSVReader


@pytest.mark.parametrize("pandas_config", [{}, {"chunksize": 2}, {"iterator": True}])
@pytest.mark.parametrize("concat_rows", [True, False])
@pytest.mark.parametrize("use_fs", [True, False])
@pytest.mark.parametrize("header_only", [True, False])
def test_pandas_csv_options(
    tmp_path: Path,
    pandas_config: dict,
    concat_rows: bool,
    use_fs: bool,
    header_only: bool,
) -> None:
    file = tmp_path / "rows.csv"
    content = "name,count\n"
    rows = []
    if not header_only:
        content += "alpha,1\nbeta,2\ngamma,3\n"
        rows = ["alpha | 1", "beta | 2", "gamma | 3"]

    fs = None
    if use_fs:
        fs = MemoryFileSystem()
        fs.pipe(str(file), content.encode())
    else:
        file.write_text(content)

    metadata = {"source": "fixture"}
    reader = PandasCSVReader(
        pandas_config=pandas_config,
        concat_rows=concat_rows,
        col_joiner=" | ",
        row_joiner=" / ",
    )
    documents = reader.load_data(file, extra_info=metadata, fs=fs)

    assert [document.text for document in documents] == (
        [" / ".join(rows)] if concat_rows else rows
    )
    assert all(document.metadata == metadata for document in documents)


@pytest.mark.parametrize("pandas_config", [{"chunksize": 2}, {"iterator": True}])
@pytest.mark.parametrize("malformed", [True, False])
def test_pandas_csv_closes_filesystem_handle(
    tmp_path: Path,
    pandas_config: dict,
    malformed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = tmp_path / "rows.csv"
    last_row = 'gamma,"unterminated' if malformed else "gamma,3"
    file.write_text(f"name,count\nalpha,1\nbeta,2\n{last_row}\n")
    fs = LocalFileSystem()
    opened = []
    original_open = fs.open

    def track_open(*args: Any, **kwargs: Any) -> Any:
        handle = original_open(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(fs, "open", track_open)
    reader = PandasCSVReader(pandas_config=pandas_config)

    if malformed:
        with pytest.raises(pd.errors.ParserError):
            reader.load_data(file, fs=fs)
    else:
        documents = reader.load_data(file, fs=fs)
        assert [document.text for document in documents] == [
            "alpha, 1\nbeta, 2\ngamma, 3"
        ]

    assert opened and all(handle.closed for handle in opened)
