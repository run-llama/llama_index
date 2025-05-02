import pytest
from unittest import mock
from llama_index.readers.gitlab import GitLabRepositoryReader
from llama_index.core import Document


@pytest.fixture()
def mock_gitlab_client():
    client = mock.Mock()
    client.api_url = "https://gitlab.com/api/v4"
    client.projects.get.return_value = mock.Mock()
    return client


def test_initialization(mock_gitlab_client):
    reader = GitLabRepositoryReader(gitlab_client=mock_gitlab_client, project_id=12345)
    assert reader._gl == mock_gitlab_client
    assert reader._project_url == "https://gitlab.com/api/v4/projects/12345"
    assert reader._project == mock_gitlab_client.projects.get.return_value


def test_load_single_file(mock_gitlab_client):
    mock_project = mock_gitlab_client.projects.get.return_value
    mock_file = mock.Mock()
    mock_file.asdict.return_value = {
        "blob_id": "123",
        "file_path": "path/to/file",
        "file_name": "file",
        "size": 100,
    }
    mock_file.decode.return_value = "file content"
    mock_project.files.get.return_value = mock_file

    reader = GitLabRepositoryReader(gitlab_client=mock_gitlab_client, project_id=12345)
    document = reader._load_single_file("path/to/file", "main")

    assert document.doc_id == "123"
    assert document.text == "file content"
    assert document.extra_info["file_path"] == "path/to/file"
    assert document.extra_info["file_name"] == "file"
    assert document.extra_info["size"] == 100
    assert (
        document.extra_info["url"]
        == "https://gitlab.com/api/v4/projects/12345/projects/repository/files/path/to/file/raw"
    )


def test_load_data(mock_gitlab_client):
    mock_project = mock_gitlab_client.projects.get.return_value
    mock_project.repository_tree.return_value = [
        {"type": "blob", "path": "path/to/file1"},
        {"type": "blob", "path": "path/to/file2"},
    ]

    reader = GitLabRepositoryReader(gitlab_client=mock_gitlab_client, project_id=12345)
    reader._load_single_file = mock.Mock(
        side_effect=[
            Document(doc_id="1", text="content1", extra_info={}),
            Document(doc_id="2", text="content2", extra_info={}),
        ]
    )

    documents = reader.load_data(ref="main", path="path/to", recursive=True)

    assert len(documents) == 2
    assert documents[0].doc_id == "1"
    assert documents[0].text == "content1"
    assert documents[1].doc_id == "2"
    assert documents[1].text == "content2"
