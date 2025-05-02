"""Test file reader."""

from multiprocessing import cpu_count
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Type, Union
import hashlib
from pathlib import Path

import pytest
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.schema import Document

try:
    from llama_index.readers.file import PDFReader
except ImportError:
    PDFReader = None  # type: ignore


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_recursive(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Test simple directory reader in recursive mode."""
    # test recursive
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with TemporaryDirectory(dir=tmp_dir) as tmp_sub_dir:
            with open(f"{tmp_sub_dir}/test2.txt", "w") as f:
                f.write("test2")
            with TemporaryDirectory(dir=tmp_sub_dir) as tmp_sub_sub_dir:
                with open(f"{tmp_sub_sub_dir}/test3.txt", "w") as f:
                    f.write("test3")
                with open(f"{tmp_sub_sub_dir}/test4.txt", "w") as f:
                    f.write("test4")

                    reader = SimpleDirectoryReader(tmp_dir, recursive=True)
                    input_file_names = [f.name for f in reader.input_files]
                    assert len(reader.input_files) == 4
                    assert set(input_file_names) == {
                        "test1.txt",
                        "test2.txt",
                        "test3.txt",
                        "test4.txt",
                    }

    # test that recursive=False works
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with TemporaryDirectory(dir=tmp_dir) as tmp_sub_dir:
            with open(f"{tmp_sub_dir}/test2.txt", "w") as f:
                f.write("test2")
            with TemporaryDirectory(dir=tmp_sub_dir) as tmp_sub_sub_dir:
                with open(f"{tmp_sub_sub_dir}/test3.txt", "w") as f:
                    f.write("test3")
                with open(f"{tmp_sub_sub_dir}/test4.txt", "w") as f:
                    f.write("test4")

                    reader = SimpleDirectoryReader(tmp_dir, recursive=False)
                    input_file_names = [f.name for f in reader.input_files]
                    print(reader.input_files)
                    assert len(reader.input_files) == 1
                    assert set(input_file_names) == {
                        "test1.txt",
                    }

    # test recursive with .md files
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.md", "w") as f:
            f.write("test1")
        with TemporaryDirectory(dir=tmp_dir) as tmp_sub_dir:
            with open(f"{tmp_sub_dir}/test2.txt", "w") as f:
                f.write("test2")
            with TemporaryDirectory(dir=tmp_sub_dir) as tmp_sub_sub_dir:
                with open(f"{tmp_sub_sub_dir}/test3.md", "w") as f:
                    f.write("test3")
                with open(f"{tmp_sub_sub_dir}/test4.txt", "w") as f:
                    f.write("test4")

                    reader = SimpleDirectoryReader(
                        tmp_dir, recursive=True, required_exts=[".md"]
                    )
                    input_file_names = [f.name for f in reader.input_files]
                    assert len(reader.input_files) == 2
                    assert set(input_file_names) == {
                        "test1.md",
                        "test3.md",
                    }


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_nonrecursive(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Test simple non-recursive directory reader."""
    # test nonrecursive
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")
        with open(f"{tmp_dir}/test3.txt", "w") as f:
            f.write("test3")
        with open(f"{tmp_dir}/test4.txt", "w") as f:
            f.write("test4")
        with open(f"{tmp_dir}/.test5.txt", "w") as f:
            f.write("test5")

        # test exclude hidden
        reader = SimpleDirectoryReader(tmp_dir, recursive=False)
        input_file_names = [f.name for f in reader.input_files]
        assert len(reader.input_files) == 4
        assert input_file_names == ["test1.txt", "test2.txt", "test3.txt", "test4.txt"]

        # test include hidden
        reader = SimpleDirectoryReader(tmp_dir, recursive=False, exclude_hidden=False)
        input_file_names = [f.name for f in reader.input_files]
        assert len(reader.input_files) == 5
        assert input_file_names == [
            ".test5.txt",
            "test1.txt",
            "test2.txt",
            "test3.txt",
            "test4.txt",
        ]


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_required_exts(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Test extension filter."""
    # test nonrecursive
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with open(f"{tmp_dir}/test2.md", "w") as f:
            f.write("test2")
        with open(f"{tmp_dir}/test3.tmp", "w") as f:
            f.write("test3")
        with open(f"{tmp_dir}/test4.json", "w") as f:
            f.write("test4")
        with open(f"{tmp_dir}/test5.json", "w") as f:
            f.write("test5")

        # test exclude hidden
        reader = SimpleDirectoryReader(tmp_dir, required_exts=[".json"])
        input_file_names = [f.name for f in reader.input_files]
        assert len(reader.input_files) == 2
        assert input_file_names == ["test4.json", "test5.json"]


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_num_files_limit(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Test num files limit."""
    # test num_files_limit (with recursion)
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with TemporaryDirectory(dir=tmp_dir) as tmp_sub_dir:
            with open(f"{tmp_sub_dir}/test2.txt", "w") as f:
                f.write("test2")
            with open(f"{tmp_sub_dir}/test3.txt", "w") as f:
                f.write("test3")
            with TemporaryDirectory(dir=tmp_sub_dir) as tmp_sub_sub_dir:
                with open(f"{tmp_sub_sub_dir}/test4.txt", "w") as f:
                    f.write("test4")

                    reader = SimpleDirectoryReader(
                        tmp_dir, recursive=True, num_files_limit=2
                    )
                    input_file_names = [f.name for f in reader.input_files]
                    assert len(reader.input_files) == 2
                    assert set(input_file_names) == {
                        "test1.txt",
                        "test2.txt",
                    }

                    reader = SimpleDirectoryReader(
                        tmp_dir, recursive=True, num_files_limit=3
                    )
                    input_file_names = [f.name for f in reader.input_files]
                    assert len(reader.input_files) == 3
                    assert set(input_file_names) == {
                        "test1.txt",
                        "test2.txt",
                        "test3.txt",
                    }

                    reader = SimpleDirectoryReader(
                        tmp_dir, recursive=True, num_files_limit=4
                    )
                    input_file_names = [f.name for f in reader.input_files]
                    assert len(reader.input_files) == 4
                    assert set(input_file_names) == {
                        "test1.txt",
                        "test2.txt",
                        "test3.txt",
                        "test4.txt",
                    }


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_file_metadata(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Test if file metadata is added to Document."""
    # test file_metadata
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")
        with open(f"{tmp_dir}/test3.txt", "w") as f:
            f.write("test3")

        test_author = "Bruce Wayne"

        def filename_to_metadata(filename: str) -> Dict[str, Any]:
            return {"filename": filename, "author": test_author}

        # test default file_metadata
        reader = SimpleDirectoryReader(tmp_dir)

        documents = reader.load_data()

        for doc in documents:
            assert "file_path" in doc.metadata

        # test customized file_metadata
        reader = SimpleDirectoryReader(tmp_dir, file_metadata=filename_to_metadata)

        documents = reader.load_data()

        for doc in documents:
            assert doc.metadata is not None and doc.metadata["author"] == test_author


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_excluded_files(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Tests if files are excluded properly."""
    # test recursive
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with TemporaryDirectory(dir=tmp_dir) as tmp_sub_dir:
            with open(f"{tmp_sub_dir}/test2.txt", "w") as f:
                f.write("test2")
            with TemporaryDirectory(dir=tmp_sub_dir) as tmp_sub_sub_dir:
                with open(f"{tmp_sub_sub_dir}/test3.txt", "w") as f:
                    f.write("test3")
                with open(f"{tmp_sub_sub_dir}/test4.txt", "w") as f:
                    f.write("test4")

                    reader = SimpleDirectoryReader(
                        tmp_dir, recursive=True, exclude=["test3.txt"]
                    )
                    input_file_names = [f.name for f in reader.input_files]
                    assert len(reader.input_files) == 3
                    assert set(input_file_names) == {
                        "test1.txt",
                        "test2.txt",
                        "test4.txt",
                    }

    # test nonrecursive exclude *.py
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.py", "w") as f:
            f.write("test1.py")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")
        with open(f"{tmp_dir}/test3.txt", "w") as f:
            f.write("test3")
        with open(f"{tmp_dir}/test4.txt", "w") as f:
            f.write("test4")
        with open(f"{tmp_dir}/test5.txt", "w") as f:
            f.write("test5")

        reader = SimpleDirectoryReader(tmp_dir, recursive=False, exclude=["*.py"])
        input_file_names = [f.name for f in reader.input_files]
        assert len(reader.input_files) == 4
        assert input_file_names == ["test2.txt", "test3.txt", "test4.txt", "test5.txt"]

    # test recursive exclude *.md
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.md", "w") as f:
            f.write("test1")
        with TemporaryDirectory(dir=tmp_dir) as tmp_sub_dir:
            with open(f"{tmp_sub_dir}/test2.txt", "w") as f:
                f.write("test2")
            with TemporaryDirectory(dir=tmp_sub_dir) as tmp_sub_sub_dir:
                with open(f"{tmp_sub_sub_dir}/test3.md", "w") as f:
                    f.write("test3")
                with open(f"{tmp_sub_sub_dir}/test4.txt", "w") as f:
                    f.write("test4")

                    reader = SimpleDirectoryReader(
                        tmp_dir, recursive=True, exclude=["*.md"]
                    )
                    input_file_names = [f.name for f in reader.input_files]
                    assert len(reader.input_files) == 2
                    assert set(input_file_names) == {
                        "test2.txt",
                        "test4.txt",
                    }


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_exclude_hidden(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Test if exclude_hidden flag excludes hidden files and files in hidden directories."""
    # test recursive exclude hidden
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with TemporaryDirectory(dir=tmp_dir) as tmp_sub_dir:
            # hidden file
            with open(f"{tmp_sub_dir}/.test2.txt", "w") as f:
                f.write("test2")
            with TemporaryDirectory(dir=tmp_sub_dir) as tmp_sub_sub_a_dir:
                with open(f"{tmp_sub_sub_a_dir}/test3.txt", "w") as f:
                    f.write("test3")
                # hidden directory
                with TemporaryDirectory(
                    dir=tmp_sub_dir, prefix="."
                ) as tmp_sub_sub_b_dir:
                    with open(f"{tmp_sub_sub_b_dir}/test4.txt", "w") as f:
                        f.write("test4")
                    with open(f"{tmp_sub_sub_b_dir}/test5.txt", "w") as f:
                        f.write("test5")

                        reader = SimpleDirectoryReader(
                            tmp_dir, recursive=True, exclude_hidden=True
                        )
                        input_file_names = [f.name for f in reader.input_files]
                        assert len(reader.input_files) == 2
                        assert set(input_file_names) == {"test1.txt", "test3.txt"}

    # test non-recursive exclude hidden files
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.py", "w") as f:
            f.write("test1.py")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")
        with open(f"{tmp_dir}/.test3.txt", "w") as f:
            f.write("test3")
        with open(f"{tmp_dir}/test4.txt", "w") as f:
            f.write("test4")
        with open(f"{tmp_dir}/.test5.py", "w") as f:
            f.write("test5")

        reader = SimpleDirectoryReader(tmp_dir, recursive=False, exclude_hidden=True)
        input_file_names = [f.name for f in reader.input_files]
        assert len(reader.input_files) == 3
        assert input_file_names == ["test1.py", "test2.txt", "test4.txt"]

    # test non-recursive exclude hidden directory
    # - i.e., user passes hidden root directory and tries to use exclude_hidden
    with TemporaryDirectory(prefix=".") as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.py", "w") as f:
            f.write("test1.py")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")
        with open(f"{tmp_dir}/.test3.txt", "w") as f:
            f.write("test3")
        with open(f"{tmp_dir}/test4.txt", "w") as f:
            f.write("test4")
        with open(f"{tmp_dir}/.test5.txt", "w") as f:
            f.write("test5")

        # correct behaviour is to raise ValueError as defined in SimpleDirectoryReader._add_files
        try:
            reader = SimpleDirectoryReader(
                tmp_dir, recursive=False, exclude_hidden=True
            )
        except ValueError as e:
            assert e.args[0] == f"No files found in {tmp_dir}."


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_filename_as_doc_id(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Test if file metadata is added to Document."""
    # test file_metadata
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")
        with open(f"{tmp_dir}/test3.txt", "w") as f:
            f.write("test3")
        with open(f"{tmp_dir}/test4.md", "w") as f:
            f.write("test4")
        with open(f"{tmp_dir}/test5.json", "w") as f:
            f.write('{"test_1": {"test_2": [1, 2, 3]}}')

        reader = SimpleDirectoryReader(tmp_dir, filename_as_id=True)

        documents = reader.load_data()

        doc_paths = [
            f"{tmp_dir}{os.sep}test1.txt",
            f"{tmp_dir}{os.sep}test2.txt",
            f"{tmp_dir}{os.sep}test3.txt",
            f"{tmp_dir}{os.sep}test4.md",
            f"{tmp_dir}{os.sep}test5.json",
        ]

        # check paths. Split handles path_part_X doc_ids from md and json files
        for doc in documents:
            assert str(doc.node_id).split("_part")[0] in doc_paths


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_specifying_encoding(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Test if file metadata is added to Document."""
    # test file_metadata
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w", encoding="latin-1") as f:
            f.write("test1á")
        with open(f"{tmp_dir}/test2.txt", "w", encoding="latin-1") as f:
            f.write("test2â")
        with open(f"{tmp_dir}/test3.txt", "w", encoding="latin-1") as f:
            f.write("test3ã")
        with open(f"{tmp_dir}/test4.json", "w", encoding="latin-1") as f:
            f.write('{"test_1á": {"test_2ã": ["â"]}}')

        reader = SimpleDirectoryReader(
            tmp_dir, filename_as_id=True, errors="strict", encoding="latin-1"
        )

        documents = reader.load_data()

        doc_paths = [
            f"{tmp_dir}{os.sep}test1.txt",
            f"{tmp_dir}{os.sep}test2.txt",
            f"{tmp_dir}{os.sep}test3.txt",
            f"{tmp_dir}{os.sep}test4.json",
        ]

        # check paths. Split handles path_part_X doc_ids from md and json files
        for doc in documents:
            assert str(doc.node_id).split("_part")[0] in doc_paths


@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_error_if_not_dir_or_file() -> None:
    with pytest.raises(ValueError, match="Directory"):
        SimpleDirectoryReader("not_a_dir")
    with pytest.raises(ValueError, match="File"):
        SimpleDirectoryReader(input_files=["not_a_file"])
    with TemporaryDirectory() as tmp_dir, pytest.raises(ValueError, match="No files"):
        SimpleDirectoryReader(tmp_dir)


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_parallel_load(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Test parallel load."""
    # test nonrecursive
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with open(f"{tmp_dir}/test2.md", "w") as f:
            f.write("test2")
        with open(f"{tmp_dir}/test3.tmp", "w") as f:
            f.write("test3")
        with open(f"{tmp_dir}/test4.json", "w") as f:
            f.write("test4")
        with open(f"{tmp_dir}/test5.json", "w") as f:
            f.write("test5")

        reader = SimpleDirectoryReader(tmp_dir, filename_as_id=True)
        num_workers = min(2, cpu_count())
        documents = reader.load_data(num_workers=num_workers)

        doc_paths = [
            f"{tmp_dir}{os.sep}test1.txt",
            f"{tmp_dir}{os.sep}test2.md",
            f"{tmp_dir}{os.sep}test3.tmp",
            f"{tmp_dir}{os.sep}test4.json",
            f"{tmp_dir}{os.sep}test5.json",
        ]

        # check paths. Split handles path_part_X doc_ids from md and json files
        for doc in documents:
            assert str(doc.node_id).split("_part")[0] in doc_paths


def _compare_document_lists(
    documents1: List[Document], documents2: List[Document]
) -> None:
    assert len(documents1) == len(documents2)
    hashes_1 = {doc.hash for doc in documents1}
    hashes_2 = {doc.hash for doc in documents2}
    assert hashes_1 == hashes_2


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_list_and_read_file_workflow(tmp_dir_type: Type[Union[Path, str]]) -> None:
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")

        reader = SimpleDirectoryReader(tmp_dir)
        original_docs = reader.load_data()

        files = reader.list_resources()
        assert len(files) == 2

        new_docs: List[Document] = []
        for file in files:
            file_info = reader.get_resource_info(file)
            assert file_info is not None
            assert len(file_info) == 4

            new_docs.extend(reader.load_resource(file))

        _compare_document_lists(original_docs, new_docs)

        new_docs = reader.load_resources(files)
        _compare_document_lists(original_docs, new_docs)


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_read_file_content(tmp_dir_type: Type[Union[Path, str]]) -> None:
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")

        files_checksum = {
            f"{tmp_dir}/test1.txt": hashlib.md5(
                open(f"{tmp_dir}/test1.txt", "rb").read()
            ).hexdigest(),
            f"{tmp_dir}/test2.txt": hashlib.md5(
                open(f"{tmp_dir}/test2.txt", "rb").read()
            ).hexdigest(),
        }

        reader = SimpleDirectoryReader(tmp_dir)

        for file in files_checksum:
            content = reader.read_file_content(file)
            checksum = hashlib.md5(content).hexdigest()
            assert checksum == files_checksum[file]


@pytest.mark.parametrize("tmp_dir_type", [Path, str])
@pytest.mark.skipif(PDFReader is None, reason="llama-index-readers-file not installed")
def test_exclude_empty(tmp_dir_type: Type[Union[Path, str]]) -> None:
    """Test if exclude_empty flag excludes empty files."""
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = tmp_dir_type(tmp_dir)

        # Create non-empty files
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")

        # Create empty files
        open(f"{tmp_dir}/empty1.txt", "w").close()
        open(f"{tmp_dir}/empty2.txt", "w").close()

        # Test with exclude_empty=True
        reader_exclude = SimpleDirectoryReader(tmp_dir, exclude_empty=True)
        documents_exclude = reader_exclude.load_data()

        assert len(documents_exclude) == 2
        assert [doc.metadata["file_name"] for doc in documents_exclude] == [
            "test1.txt",
            "test2.txt",
        ]

        # Test with exclude_empty=False (default behavior)
        reader_include = SimpleDirectoryReader(tmp_dir, exclude_empty=False)
        documents_include = reader_include.load_data()

        assert len(documents_include) == 4
        assert [doc.metadata["file_name"] for doc in documents_include] == [
            "empty1.txt",
            "empty2.txt",
            "test1.txt",
            "test2.txt",
        ]
