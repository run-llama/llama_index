"""Test file reader."""

import json
import sys
from tempfile import TemporaryDirectory

import pytest
from llama_index.core.readers.json import JSONReader


def test_basic() -> None:
    """Test JSON reader in basic mode."""
    with TemporaryDirectory() as tmp_dir:
        file_name = f"{tmp_dir}/test1.json"

        with open(file_name, "w") as f:
            f.write('{"test1": "test1"}')

        reader = JSONReader()
        data = reader.load_data(file_name)
        assert len(data) == 1
        assert isinstance(data[0].get_content(), str)
        assert data[0].get_content().index("test1") is not None


def test_levels_back0() -> None:
    """Test JSON reader using the levels_back function."""
    with TemporaryDirectory() as tmp_dir:
        file_name = f"{tmp_dir}/test2.json"
        with open(file_name, "w") as f:
            f.write('{ "a": { "b": ["c"] } }')

        reader1 = JSONReader(levels_back=0)
        data1 = reader1.load_data(file_name)
        assert data1[0].get_content() == "a b c"

        reader2 = JSONReader(levels_back=1)
        data2 = reader2.load_data(file_name)
        assert data2[0].get_content() == "b c"


def test_collapse_length() -> None:
    """Test JSON reader using the collapse_length function."""
    with TemporaryDirectory() as tmp_dir:
        file_name = f"{tmp_dir}/test3.json"
        with open(file_name, "w") as f:
            f.write('{ "a": { "b": "c" } }')

        reader1 = JSONReader(levels_back=0, collapse_length=100)
        data1 = reader1.load_data(file_name)
        assert isinstance(data1[0].get_content(), str)
        assert data1[0].get_content().index('"a":') is not None

        reader2 = JSONReader(levels_back=0, collapse_length=10)
        data2 = reader2.load_data(file_name)
        assert isinstance(data2[0].get_content(), str)
        assert data2[0].get_content().index("a ") is not None


def test_jsonl() -> None:
    """Test JSON reader using the is_jsonl function."""
    with TemporaryDirectory() as tmp_dir:
        file_name = f"{tmp_dir}/test4.json"
        with open(file_name, "w") as f:
            f.write('{"test1": "test1"}\n{"test2": "test2"}\n{"test3": "test3"}\n')

        reader = JSONReader(is_jsonl=True)
        data = reader.load_data(file_name)
        assert len(data) == 3
        assert isinstance(data[0].get_content(), str)
        assert data[0].get_content().index("test1") is not None
        assert isinstance(data[1].get_content(), str)
        assert data[1].get_content().index("test2") is not None
        assert isinstance(data[2].get_content(), str)
        assert data[2].get_content().index("test3") is not None


def test_clean_json() -> None:
    """Test JSON reader using the clean_json function."""
    with TemporaryDirectory() as tmp_dir:
        file_name = f"{tmp_dir}/test5.json"
        with open(file_name, "w") as f:
            f.write('{ "a": { "b": "c" } }')

        # If levels back is set clean_json is ignored
        reader1 = JSONReader(levels_back=0, clean_json=False)
        data1 = reader1.load_data(file_name)
        assert data1[0].get_content() == "a b c"

        # If clean_json is false the full json should be contained in a document
        reader1 = JSONReader(clean_json=False)
        data1 = reader1.load_data(file_name)
        assert data1[0].get_content() == '{"a": {"b": "c"}}'

        # If clean_json is True the full json should be contained in a document
        reader1 = JSONReader(clean_json=True)
        data1 = reader1.load_data(file_name)
        assert data1[0].get_content() == '"a": {\n"b": "c"'


def test_max_recursion_attack(tmp_path):
    original_limit = sys.getrecursionlimit()
    try:
        nested_dict = {}
        current_level = nested_dict
        sys.setrecursionlimit(5000)

        for i in range(1, 2001):  # Create 2000 levels of nesting
            if i == 2000:
                current_level[f"level{i}"] = "final_value"
            else:
                current_level[f"level{i}"] = {}
                current_level = current_level[f"level{i}"]

        file_name = tmp_path / "test_nested.json"
        with open(file_name, "w") as f:
            f.write(json.dumps(nested_dict))

        # Force a recursion error
        sys.setrecursionlimit(500)
        reader = JSONReader(levels_back=1)
        with pytest.warns(UserWarning):
            data = reader.load_data(file_name)
            assert data == []

    finally:
        sys.setrecursionlimit(original_limit)
