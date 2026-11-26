from io import BytesIO
from pathlib import PurePosixPath
from unittest.mock import Mock
from xml.etree.ElementTree import ParseError

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file.xml import XMLReader


@pytest.mark.parametrize("split_level", [0, 1])
@pytest.mark.parametrize("as_path", [False, True])
def test_xml_filesystem_matches_local(tmp_path, split_level, as_path):
    content = '<?xml version="1.0" encoding="iso-8859-1"?><items><item>café</item><item>tea</item></items>'.encode(
        "iso-8859-1"
    )
    local_file = tmp_path / "data.xml"
    local_file.write_bytes(content)
    fs = MemoryFileSystem()
    remote_file = f"/xml-reader/{tmp_path.name}/data.xml"
    fs.pipe(remote_file, content)
    reader = XMLReader(tree_level_split=split_level)
    metadata = {"source": "fixture"}

    expected = reader.load_data(local_file, extra_info=metadata)
    actual = reader.load_data(
        PurePosixPath(remote_file) if as_path else remote_file,
        extra_info=metadata,
        fs=fs,
    )

    assert [doc.text for doc in actual] == [doc.text for doc in expected]
    assert len(actual) == (1 if split_level == 0 else 2)
    assert "café" in actual[0].text
    assert all(doc.metadata == metadata for doc in actual)


def test_xml_filesystem_through_directory_reader(tmp_path):
    fs = MemoryFileSystem()
    remote_file = f"/xml-reader/{tmp_path.name}/directory.xml"
    fs.pipe(remote_file, b"<items><item>Apple</item><item>Tea</item></items>")
    reader = SimpleDirectoryReader(
        input_files=[remote_file],
        fs=fs,
        file_extractor={".xml": XMLReader(tree_level_split=1)},
        raise_on_error=True,
    )

    documents = reader.load_data()

    assert [doc.text for doc in documents] == ["<item>Apple</item>", "<item>Tea</item>"]
    assert all(doc.metadata["file_name"] == "directory.xml" for doc in documents)


@pytest.mark.parametrize("content", [b"<root>ok</root>", b"<root>"])
def test_xml_filesystem_closes_handle(tmp_path, content):
    fs = MemoryFileSystem()
    remote_file = f"/xml-reader/{tmp_path.name}/close.xml"
    fs.pipe(remote_file, content)
    handle = BytesIO(fs.cat(remote_file))
    filesystem = Mock(wraps=fs)
    filesystem.open.return_value = handle

    if content == b"<root>":
        with pytest.raises(ParseError):
            XMLReader().load_data(remote_file, fs=filesystem)
    else:
        assert (
            XMLReader().load_data(remote_file, fs=filesystem)[0].text
            == "<root>ok</root>"
        )

    assert handle.closed
