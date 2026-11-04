from pathlib import Path
from typing import List, Optional

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import IPYNBReader

pytest.importorskip("nbconvert")
nbformat = pytest.importorskip("nbformat")


def write_notebook(file: Path, cells: List, language: str = "python") -> None:
    notebook = nbformat.v4.new_notebook(cells=cells)
    if language == "python":
        notebook.metadata.language_info = {
            "name": "python",
            "version": "3.12",
            "mimetype": "text/x-python",
            "file_extension": ".py",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "codemirror_mode": {"name": "ipython", "version": 3},
        }
    elif language:
        notebook.metadata.language_info = {
            "name": language,
            "mimetype": "text/plain",
            "file_extension": ".txt",
        }
    nbformat.write(notebook, file)


@pytest.mark.parametrize("concatenate", [False, True])
@pytest.mark.parametrize("execution_count", [None, 1])
def test_code_without_requiring_execution(
    tmp_path: Path, concatenate: bool, execution_count: Optional[int]
) -> None:
    file = tmp_path / "code.ipynb"
    write_notebook(
        file,
        [
            nbformat.v4.new_code_cell(
                'print("payload_marker")', execution_count=execution_count
            )
        ],
    )

    docs = IPYNBReader(concatenate=concatenate).load_data(
        file, extra_info={"source": "fixture"}
    )

    assert len(docs) == 1
    assert docs[0].text.strip() == 'print("payload_marker")'
    assert docs[0].metadata == {"source": "fixture"}


@pytest.mark.parametrize("concatenate", [False, True])
def test_markdown_only_notebook(tmp_path: Path, concatenate: bool) -> None:
    file = tmp_path / "markdown.ipynb"
    write_notebook(file, [nbformat.v4.new_markdown_cell("# 中文 notebook notes")])

    docs = IPYNBReader(concatenate=concatenate).load_data(file)

    assert len(docs) == 1
    assert "# 中文 notebook notes" in docs[0].text
    assert "#!/usr/bin/env python" not in docs[0].text


@pytest.mark.parametrize("concatenate", [False, True])
def test_mixed_cells_preserve_order_and_exported_code(
    tmp_path: Path, concatenate: bool
) -> None:
    file = tmp_path / "mixed.ipynb"
    write_notebook(
        file,
        [
            nbformat.v4.new_markdown_cell("Introduction"),
            nbformat.v4.new_code_cell("first = 1", execution_count=4),
            nbformat.v4.new_markdown_cell("Explanation"),
            nbformat.v4.new_code_cell("%time second = 2"),
            nbformat.v4.new_raw_cell("Raw notes"),
        ],
    )

    docs = IPYNBReader(concatenate=concatenate).load_data(file)

    expected = [
        "Introduction",
        "first = 1",
        "Explanation",
        "get_ipython().run_line_magic('time', 'second = 2')",
        "Raw notes",
    ]
    assert len(docs) == (1 if concatenate else 5)
    if concatenate:
        offsets = [docs[0].text.index(text) for text in expected]
        assert offsets == sorted(offsets)
    else:
        for doc, text in zip(docs, expected):
            assert text in doc.text
    assert all("#!/usr/bin/env python" not in doc.text for doc in docs)


def test_prompt_text_inside_a_code_cell_is_not_a_separator(tmp_path: Path) -> None:
    file = tmp_path / "literal.ipynb"
    source = 'text = "In[1]:"\nblock = """\n# In[2]:\n\nstill inside the string\n"""'
    write_notebook(
        file,
        [nbformat.v4.new_code_cell(source, execution_count=3)],
    )

    docs = IPYNBReader().load_data(file)

    assert len(docs) == 1
    assert docs[0].text.strip() == source


@pytest.mark.parametrize("language", ["javascript", ""])
def test_notebook_without_python_prompts(tmp_path: Path, language: str) -> None:
    file = tmp_path / "plain.ipynb"
    write_notebook(
        file,
        [
            nbformat.v4.new_markdown_cell("Notebook notes"),
            nbformat.v4.new_code_cell("const value = 1;"),
        ],
        language=language,
    )

    docs = IPYNBReader().load_data(file)

    assert len(docs) == 2
    assert "Notebook notes" in docs[0].text
    assert docs[1].text.strip() == "const value = 1;"


def test_read_notebook_from_fsspec(tmp_path: Path) -> None:
    file = tmp_path / "remote.ipynb"
    write_notebook(file, [nbformat.v4.new_code_cell('print("中文")')])
    fs = MemoryFileSystem()
    fs.pipe("/reader-test/remote.ipynb", file.read_bytes())

    docs = IPYNBReader().load_data(
        Path("/reader-test/remote.ipynb"), fs=fs, extra_info={"source": "remote"}
    )

    assert len(docs) == 1
    assert docs[0].text.strip() == 'print("中文")'
    assert docs[0].metadata == {"source": "remote"}


def test_directory_reader_keeps_unexecuted_notebook(tmp_path: Path) -> None:
    file = tmp_path / "directory.ipynb"
    write_notebook(file, [nbformat.v4.new_code_cell("value = 1")])

    docs = SimpleDirectoryReader(
        input_files=[file],
        file_extractor={".ipynb": IPYNBReader()},
        raise_on_error=True,
    ).load_data()

    assert len(docs) == 1
    assert docs[0].text.strip() == "value = 1"


@pytest.mark.parametrize("concatenate", [False, True])
def test_empty_notebook(tmp_path: Path, concatenate: bool) -> None:
    file = tmp_path / "empty.ipynb"
    write_notebook(file, [])

    docs = IPYNBReader(concatenate=concatenate).load_data(file)

    assert len(docs) == (1 if concatenate else 0)
    assert all(doc.text == "" for doc in docs)


@pytest.mark.parametrize("concatenate", [False, True])
def test_empty_cells_do_not_produce_script_headers(
    tmp_path: Path, concatenate: bool
) -> None:
    file = tmp_path / "empty_cells.ipynb"
    write_notebook(
        file,
        [
            nbformat.v4.new_markdown_cell("\n  "),
            nbformat.v4.new_code_cell(""),
            nbformat.v4.new_raw_cell("\n"),
        ],
    )

    docs = IPYNBReader(concatenate=concatenate).load_data(file)

    assert len(docs) == (1 if concatenate else 0)
    assert all(doc.text == "" for doc in docs)


@pytest.mark.parametrize("cell_type", ["markdown", "code"])
def test_exporter_remove_source_metadata_is_respected(
    tmp_path: Path, cell_type: str
) -> None:
    file = tmp_path / "excluded.ipynb"
    new_cell = (
        nbformat.v4.new_markdown_cell
        if cell_type == "markdown"
        else nbformat.v4.new_code_cell
    )
    write_notebook(
        file,
        [
            new_cell(
                source="excluded source",
                metadata={"transient": {"remove_source": True}},
            )
        ],
    )

    assert IPYNBReader().load_data(file) == []
