"""Test file reader."""

from tempfile import TemporaryDirectory
from typing import Any, Dict
import pytest

from llama_index.readers.file.base import SimpleDirectoryReader


def test_recursive() -> None:
    """Test simple directory reader in recursive mode."""
    # test recursive
    with TemporaryDirectory() as tmp_dir:
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


def test_nonrecursive() -> None:
    """Test simple non-recursive directory reader."""
    # test nonrecursive
    with TemporaryDirectory() as tmp_dir:
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


def test_required_exts() -> None:
    """Test extension filter."""
    # test nonrecursive
    with TemporaryDirectory() as tmp_dir:
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


def test_num_files_limit() -> None:
    """Test num files limit."""
    # test num_files_limit (with recursion)
    with TemporaryDirectory() as tmp_dir:
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


def test_file_metadata() -> None:
    """Test if file metadata is added to Document."""
    # test file_metadata
    with TemporaryDirectory() as tmp_dir:
        with open(f"{tmp_dir}/test1.txt", "w") as f:
            f.write("test1")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")
        with open(f"{tmp_dir}/test3.txt", "w") as f:
            f.write("test3")

        test_author = "Bruce Wayne"

        def filename_to_metadata(filename: str) -> Dict[str, Any]:
            return {"filename": filename, "author": test_author}

        reader = SimpleDirectoryReader(tmp_dir, file_metadata=filename_to_metadata)

        documents = reader.load_data()

        for d in documents:
            assert d.metadata is not None and d.metadata["author"] == test_author


def test_excluded_files() -> None:
    """Tests if files are excluded properly."""
    # test recursive
    with TemporaryDirectory() as tmp_dir:
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

    # test nonrecursive
    with TemporaryDirectory() as tmp_dir:
        with open(f"{tmp_dir}/test1.py", "w") as f:
            f.write("test1.py")
        with open(f"{tmp_dir}/test2.txt", "w") as f:
            f.write("test2")
        with open(f"{tmp_dir}/test3.txt", "w") as f:
            f.write("test3")
        with open(f"{tmp_dir}/test4.txt", "w") as f:
            f.write("test4")
        with open(f"{tmp_dir}/.test5.txt", "w") as f:
            f.write("test5")

        # test exclude hidden
        reader = SimpleDirectoryReader(tmp_dir, recursive=False, exclude=["*.py"])
        input_file_names = [f.name for f in reader.input_files]
        assert len(reader.input_files) == 3
        assert input_file_names == ["test2.txt", "test3.txt", "test4.txt"]

    # test recursive exclude *.md
    with TemporaryDirectory() as tmp_dir:
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


def test_filename_as_doc_id() -> None:
    """Test if file metadata is added to Document."""
    # test file_metadata
    with TemporaryDirectory() as tmp_dir:
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
            f"{tmp_dir}/test1.txt",
            f"{tmp_dir}/test2.txt",
            f"{tmp_dir}/test3.txt",
            f"{tmp_dir}/test4.md",
            f"{tmp_dir}/test5.json",
        ]

        # check paths. Split handles path_part_X doc_ids from md and json files
        for doc in documents:
            assert str(doc.node_id).split("_part")[0] in doc_paths


def test_error_if_not_dir_or_file() -> None:
    with pytest.raises(ValueError):
        SimpleDirectoryReader("not_a_dir")
    with pytest.raises(ValueError):
        SimpleDirectoryReader(input_files=["not_a_file"])
