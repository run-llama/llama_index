from typing import Optional
import tempfile
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.schema import Document
import os
import pytest
from uuid import uuid4

base_url = os.environ.get("LLAMA_CLOUD_BASE_URL", None)
api_key = os.environ.get("LLAMA_CLOUD_API_KEY", None)
openai_api_key = os.environ.get("OPENAI_API_KEY", None)
organization_id = os.environ.get("LLAMA_CLOUD_ORGANIZATION_ID", None)


def test_class():
    names_of_base_classes = [b.__name__ for b in LlamaCloudIndex.__mro__]
    assert BaseManagedIndex.__name__ in names_of_base_classes


@pytest.mark.skipif(
    not base_url or not api_key, reason="No platform base url or api key set"
)
@pytest.mark.skipif(not openai_api_key, reason="No openai api key set")
@pytest.mark.integration()
def test_retrieve():
    os.environ["OPENAI_API_KEY"] = openai_api_key
    index = LlamaCloudIndex(
        name="test",  # assumes this pipeline exists
        project_name="Default",
        api_key=api_key,
        base_url=base_url,
    )
    query = "test"
    nodes = index.as_retriever().retrieve(query)
    assert nodes is not None and len(nodes) > 0

    response = index.as_query_engine().query(query)
    assert response is not None and len(response.response) > 0


@pytest.mark.parametrize("organization_id", [None, organization_id])
@pytest.mark.skipif(
    not base_url or not api_key, reason="No platform base url or api key set"
)
@pytest.mark.skipif(not openai_api_key, reason="No openai api key set")
@pytest.mark.integration()
def test_documents_crud(organization_id: Optional[str]):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    documents = [
        Document(text="Hello world.", doc_id="1", metadata={"source": "test"}),
    ]
    index = LlamaCloudIndex.from_documents(
        documents=documents,
        name=f"test pipeline {uuid4()}",
        api_key=api_key,
        base_url=base_url,
        organization_id=organization_id,
        verbose=True,
    )
    docs = index.ref_doc_info
    assert len(docs) == 1
    assert docs["1"].metadata["source"] == "test"
    nodes = index.as_retriever().retrieve("Hello world.")
    assert len(nodes) > 0
    assert all(n.node.ref_doc_id == "1" for n in nodes)
    assert all(n.node.metadata["source"] == "test" for n in nodes)

    index.insert(
        Document(text="Hello world.", doc_id="2", metadata={"source": "inserted"}),
        verbose=True,
    )
    docs = index.ref_doc_info
    assert len(docs) == 2
    assert docs["2"].metadata["source"] == "inserted"
    nodes = index.as_retriever().retrieve("Hello world.")
    assert len(nodes) > 0
    assert all(n.node.ref_doc_id in ["1", "2"] for n in nodes)
    assert any(n.node.ref_doc_id == "1" for n in nodes)
    assert any(n.node.ref_doc_id == "2" for n in nodes)

    index.update_ref_doc(
        Document(text="Hello world.", doc_id="2", metadata={"source": "updated"}),
        verbose=True,
    )
    docs = index.ref_doc_info
    assert len(docs) == 2
    assert docs["2"].metadata["source"] == "updated"

    index.refresh_ref_docs(
        [
            Document(text="Hello world.", doc_id="1", metadata={"source": "refreshed"}),
            Document(text="Hello world.", doc_id="3", metadata={"source": "refreshed"}),
        ]
    )
    docs = index.ref_doc_info
    assert len(docs) == 3
    assert docs["3"].metadata["source"] == "refreshed"
    assert docs["1"].metadata["source"] == "refreshed"

    index.delete_ref_doc("3", verbose=True)
    docs = index.ref_doc_info
    assert len(docs) == 2
    assert "3" not in docs


@pytest.mark.skipif(
    not base_url or not api_key, reason="No platform base url or api key set"
)
@pytest.mark.skipif(not openai_api_key, reason="No openai api key set")
@pytest.mark.integration()
def test_upload_file():
    os.environ["OPENAI_API_KEY"] = openai_api_key
    index = LlamaCloudIndex(
        name="test",  # assumes this pipeline exists
        project_name="Default",
        api_key=api_key,
        base_url=base_url,
    )

    # Create a temporary file to upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Sample content for testing upload.")
        temp_file_path = temp_file.name

    try:
        # Upload the file
        file_id = index.upload_file(temp_file_path, verbose=True)
        assert file_id is not None

        # Verify the file is part of the index
        docs = index.ref_doc_info
        temp_file_name = os.path.basename(temp_file_path)
        assert any(
            temp_file_name == doc.metadata.get("file_name") for doc in docs.values()
        )

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


@pytest.mark.skipif(
    not base_url or not api_key, reason="No platform base url or api key set"
)
@pytest.mark.skipif(not openai_api_key, reason="No openai api key set")
@pytest.mark.integration()
def test_upload_file_from_url():
    os.environ["OPENAI_API_KEY"] = openai_api_key
    index = LlamaCloudIndex(
        name="test",  # assumes this pipeline exists
        project_name="Default",
        api_key=api_key,
        base_url=base_url,
    )

    # Define a URL to a file for testing
    test_file_url = "https://www.google.com/robots.txt"
    test_file_name = "google_robots.txt"

    # Upload the file from the URL
    file_id = index.upload_file_from_url(
        file_name=test_file_name, url=test_file_url, verbose=True
    )
    assert file_id is not None

    # Verify the file is part of the index
    docs = index.ref_doc_info
    assert any(test_file_name == doc.metadata.get("file_name") for doc in docs.values())
