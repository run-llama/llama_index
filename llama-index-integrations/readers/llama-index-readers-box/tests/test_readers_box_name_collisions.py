import io
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from llama_index.readers.box import BoxReader
from llama_index.readers.box.BoxAPI.box_api import download_file_by_id
from llama_index.readers.box.BoxReader import base as box_reader_base


def _box_file(file_id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(id=file_id, name=name)


def _client() -> mock.MagicMock:
    client = mock.MagicMock()
    client.downloads.download_file.side_effect = lambda file_id: io.BytesIO(
        f"content of {file_id}".encode()
    )
    return client


def test_download_same_named_files_keeps_both_contents(tmp_path):
    client = _client()
    first = _box_file("111", "report.txt")
    second = _box_file("222", "report.txt")

    first_path = download_file_by_id(client, first, str(tmp_path))
    second_path = download_file_by_id(client, second, str(tmp_path))

    assert first_path != second_path
    assert Path(first_path).read_text() == "content of 111"
    assert Path(second_path).read_text() == "content of 222"


def test_load_data_returns_all_same_named_files():
    client = _client()
    reader = BoxReader(box_client=client)
    reader._box_client = client
    box_files = [_box_file("111", "report.txt"), _box_file("222", "report.txt")]

    with (
        mock.patch.object(box_reader_base, "box_check_connection"),
        mock.patch.object(
            box_reader_base, "get_box_files_details", return_value=box_files
        ),
        mock.patch.object(
            box_reader_base,
            "box_file_to_llama_document_metadata",
            side_effect=lambda file: {"box_file_id": file.id},
        ),
    ):
        documents = reader.load_data(file_ids=["111", "222"])

    assert len(documents) == 2
    assert {(doc.text, doc.metadata["box_file_id"]) for doc in documents} == {
        ("content of 111", "111"),
        ("content of 222", "222"),
    }
