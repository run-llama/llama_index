import io
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from box_sdk_gen import BoxClient

from llama_index.readers.box import BoxReader
from llama_index.readers.box.BoxAPI.box_api import download_file_by_id
from llama_index.readers.box.BoxReader import base as box_reader_base


def _box_file(file_id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(id=file_id, name=name)


def _mock_client() -> mock.MagicMock:
    client = mock.MagicMock(spec=BoxClient)
    client.downloads = mock.MagicMock()
    client.downloads.download_file.side_effect = lambda file_id: io.BytesIO(
        f"content of {file_id}".encode()
    )
    return client


def test_download_file_by_id_same_name_files_do_not_collide(tmp_path):
    client = _mock_client()
    file_a = _box_file("111", "report.txt")
    file_b = _box_file("222", "report.txt")

    path_a = download_file_by_id(
        box_client=client, box_file=file_a, temp_dir=str(tmp_path)
    )
    path_b = download_file_by_id(
        box_client=client, box_file=file_b, temp_dir=str(tmp_path)
    )

    assert path_a == str(tmp_path / "111" / "report.txt")
    assert path_b == str(tmp_path / "222" / "report.txt")
    assert Path(path_a).read_text() == "content of 111"
    assert Path(path_b).read_text() == "content of 222"


def test_load_data_returns_all_same_named_files():
    client = _mock_client()
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
            side_effect=lambda f: {"box_file_id": f.id},
        ),
    ):
        documents = reader.load_data(file_ids=["111", "222"])

    assert len(documents) == 2
    assert {(doc.text, doc.metadata["box_file_id"]) for doc in documents} == {
        ("content of 111", "111"),
        ("content of 222", "222"),
    }
