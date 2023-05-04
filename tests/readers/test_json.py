"""Test file reader."""

from tempfile import TemporaryDirectory

from llama_index.readers.json import JSONReader


def test_basic() -> None:
    """Test JSON reader in basic mode."""
    with TemporaryDirectory() as tmp_dir:
        file_name = f"{tmp_dir}/test1.json"

        with open(file_name, "w") as f:
            f.write('{"test1": "test1"}')

        reader = JSONReader()
        data = reader.load_data(file_name)
        assert len(data) == 1
        assert isinstance(data[0].text, str)
        assert data[0].text.index("test1") is not None


def test_levels_back0() -> None:
    """Test JSON reader using the levels_back function."""
    with TemporaryDirectory() as tmp_dir:
        file_name = f"{tmp_dir}/test2.json"
        with open(file_name, "w") as f:
            f.write('{ "a": { "b": "c" } }')

        reader1 = JSONReader(levels_back=0)
        data1 = reader1.load_data(file_name)
        assert data1[0].text == "a b c"

        reader2 = JSONReader(levels_back=1)
        data2 = reader2.load_data(file_name)
        assert data2[0].text == "b c"


def test_collapse_length() -> None:
    """Test JSON reader using the collapse_length function."""
    with TemporaryDirectory() as tmp_dir:
        file_name = f"{tmp_dir}/test3.json"
        with open(file_name, "w") as f:
            f.write('{ "a": { "b": "c" } }')

        reader1 = JSONReader(levels_back=0, collapse_length=100)
        data1 = reader1.load_data(file_name)
        assert isinstance(data1[0].text, str)
        assert data1[0].text.index('"a":') is not None

        reader2 = JSONReader(levels_back=0, collapse_length=10)
        data2 = reader2.load_data(file_name)
        assert isinstance(data2[0].text, str)
        assert data2[0].text.index("a ") is not None
