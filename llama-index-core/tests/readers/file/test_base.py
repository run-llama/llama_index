from pathlib import Path
from unittest import mock

import pytest
from fsspec.implementations.local import LocalFileSystem
from llama_index.core import Document
from llama_index.core.readers.file.base import (
    SimpleDirectoryReader,
    _DefaultFileMetadataFunc,
    _format_file_timestamp,
    default_file_metadata_func,
    is_default_fs,
)


@pytest.fixture()
def data_path():
    return Path(__file__).resolve().parent / "data"


def test__format_file_timestamp():
    assert _format_file_timestamp(None) is None  # type: ignore
    assert _format_file_timestamp(0, include_time=False) == "1970-01-01"
    assert _format_file_timestamp(0, include_time=True) == "1970-01-01T00:00:00Z"


def test_default_file_metadata_func():
    meta = default_file_metadata_func(__file__)
    assert list(meta.keys()) == [
        "file_path",
        "file_name",
        "file_type",
        "file_size",
        "creation_date",
        "last_modified_date",
    ]

    with mock.patch("llama_index.core.readers.file.base.os") as mock_os:
        mock_os.path.basename.side_effect = (ValueError, "test_path")
        meta = default_file_metadata_func(__file__)
        assert meta["file_name"] == "test_path"


def test__DefaultFileMetadataFunc():
    func = _DefaultFileMetadataFunc()
    meta = func(__file__)
    assert meta["file_type"] == "text/x-python"


def test_is_default_fs():
    assert is_default_fs(LocalFileSystem()) is True
    assert is_default_fs(mock.MagicMock(auto_mkdir=True)) is False


def test_SimpleDirectoryReader_init(data_path):
    with pytest.raises(
        ValueError, match="Must provide either `input_dir` or `input_files`."
    ):
        SimpleDirectoryReader()

    with pytest.raises(ValueError, match="File does_not_exist does not exist."):
        SimpleDirectoryReader(input_files=["does_not_exist"])

    r = SimpleDirectoryReader(input_files=[__file__])
    assert r.input_files == [Path(__file__)]

    with pytest.raises(ValueError, match="Directory does_not_exist does not exist."):
        SimpleDirectoryReader(input_dir="does_not_exist")

    r = SimpleDirectoryReader(input_dir=data_path)
    assert r.input_files[0].name == "excluded_0.txt"


def test_SimpleDirectoryReader_recursive(data_path):
    r = SimpleDirectoryReader(input_dir=data_path, recursive=True)
    assert [f.name for f in r.input_files] == [
        "excluded_1.txt",
        "excluded_0.txt",
        "file_0.md",
        "file_0.xyz",
        "file_1.txt",
    ]


def test_SimpleDirectoryReader_excluded(data_path):
    r = SimpleDirectoryReader(input_dir=data_path, exclude=["excluded*"])
    assert [f.name for f in r.input_files] == ["file_0.md", "file_0.xyz"]

    r = SimpleDirectoryReader(
        input_dir=data_path, recursive=True, exclude=["excluded*"]
    )
    assert [f.name for f in r.input_files] == ["file_0.md", "file_0.xyz", "file_1.txt"]


def test_SimpleDirectoryReader_empty(data_path):
    with pytest.raises(ValueError, match="No files found in"):
        SimpleDirectoryReader(input_dir=data_path / "empty")


def test_SimpleDirectoryReader_file_limit(data_path):
    r = SimpleDirectoryReader(input_dir=data_path, recursive=True, num_files_limit=2)
    assert [f.name for f in r.input_files] == ["excluded_1.txt", "excluded_0.txt"]


def test_SimpleDirectoryReader_list_resources(data_path):
    r = SimpleDirectoryReader(input_dir=data_path, exclude=["excluded*"])
    res = r.list_resources()
    assert len(res) == 2
    assert "file_0.md" in res[0]


def test_SimpleDirectoryReader_get_resource_info(data_path):
    r = SimpleDirectoryReader(input_dir=data_path, exclude=["excluded*"])
    res = r.get_resource_info(str(data_path / "file_0.md"))
    assert res["file_path"].endswith("file_0.md")


def test_SimpleDirectoryReader_load_resource(data_path):
    r = SimpleDirectoryReader(input_dir=data_path)
    res = r.load_resource(str(data_path / "file_0.md"))
    assert len(res) == 1
    assert isinstance(res[0], Document)


@pytest.mark.asyncio()
async def test_SimpleDirectoryReader_aload_resource(data_path):
    r = SimpleDirectoryReader(input_dir=data_path)
    res = await r.aload_resource(str(data_path / "file_0.md"))
    assert len(res) == 1
    assert isinstance(res[0], Document)


def test_SimpleDirectoryReader_read_file_content(data_path):
    r = SimpleDirectoryReader(input_dir=data_path)
    content = r.read_file_content(data_path / "file_0.md")
    assert content.decode("utf-8").startswith("# Hello")


def test_SimpleDirectoryReader__exclude_metadata(data_path):
    r = SimpleDirectoryReader(input_dir=data_path)
    doc = mock.MagicMock()
    doc.excluded_embed_metadata_keys = []
    doc.excluded_llm_metadata_keys = []
    res = r._exclude_metadata([doc])  # type:ignore
    assert res[0].excluded_embed_metadata_keys == [
        "file_name",
        "file_type",
        "file_size",
        "creation_date",
        "last_modified_date",
        "last_accessed_date",
    ]
    assert res[0].excluded_llm_metadata_keys == [
        "file_name",
        "file_type",
        "file_size",
        "creation_date",
        "last_modified_date",
        "last_accessed_date",
    ]


def test_SimpleDirectoryReader_load_file(data_path):
    docs = SimpleDirectoryReader.load_file(
        input_file=data_path / "file_0.md",
        file_metadata=lambda x: {},
        file_extractor={},
        filename_as_id=True,
    )
    assert len(docs) == 1


def test_SimpleDirectoryReader_load_file_extractor(data_path):
    extractor = mock.MagicMock()
    extractor.load_data.return_value = [Document()]
    docs = SimpleDirectoryReader.load_file(
        input_file=data_path / "file_0.md",
        file_metadata=lambda x: {},
        file_extractor={".md": extractor},
        filename_as_id=True,
        fs=mock.MagicMock(),
    )
    assert len(docs) == 1


@pytest.mark.asyncio()
async def test_SimpleDirectoryReader_aload_file_extractor(data_path):
    extractor = mock.AsyncMock()
    extractor.aload_data.return_value = [Document()]
    docs = await SimpleDirectoryReader.aload_file(
        input_file=data_path / "file_0.md",
        file_metadata=lambda x: {},
        file_extractor={".md": extractor},
        filename_as_id=True,
        fs=mock.MagicMock(),
    )
    assert len(docs) == 1


def test_SimpleDirectoryReader_load_file_error(data_path):
    extractor = mock.MagicMock()
    extractor.load_data.side_effect = ValueError

    # Raise on error
    with pytest.raises(Exception, match="Error loading file"):
        SimpleDirectoryReader.load_file(
            input_file=data_path / "file_0.md",
            file_metadata=lambda x: {},
            file_extractor={".md": extractor},
            raise_on_error=True,
        )

    # Continue on error
    docs = SimpleDirectoryReader.load_file(
        input_file=data_path / "file_0.md",
        file_metadata=lambda x: {},
        file_extractor={".md": extractor},
        raise_on_error=False,
    )
    assert not docs

    # Always raise ImportError
    extractor.load_data.side_effect = ImportError("Module Foo not found.")
    with pytest.raises(ImportError, match="Module Foo not found."):
        SimpleDirectoryReader.load_file(
            input_file=data_path / "file_0.md",
            file_metadata=lambda x: {},
            file_extractor={".md": extractor},
            raise_on_error=False,
        )


def test_SimpleDirectoryReader_load_file_unknown(data_path):
    docs = SimpleDirectoryReader.load_file(
        input_file=data_path / "file_0.xyz",
        file_metadata=lambda x: {},
        file_extractor={},
        filename_as_id=True,
    )
    assert len(docs) == 1
    assert docs[0].text.startswith("Lorem ipsum")


def test_SimpleDirectoryReader_load_data(data_path):
    r = SimpleDirectoryReader(input_dir=data_path, recursive=True)
    docs = r.load_data()
    assert len(docs) == 5


@pytest.mark.asyncio()
async def test_SimpleDirectoryReader_aload_data(data_path):
    r = SimpleDirectoryReader(input_dir=data_path, recursive=True)
    docs = await r.aload_data(num_workers=2)
    assert len(docs) == 5


def test_SimpleDirectoryReader_iter_data(data_path):
    r = SimpleDirectoryReader(input_dir=data_path, recursive=True)
    docs = list(r.iter_data())
    assert len(docs) == 5
