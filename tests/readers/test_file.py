"""Test file reader."""

from tempfile import TemporaryDirectory

from gpt_index.readers.file import SimpleDirectoryReader


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
