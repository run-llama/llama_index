from typing import Optional
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.schema import Document
import os
import pytest
from uuid import uuid4

base_url = os.environ.get("LLAMA_CLOUD_BASE_URL", None)
api_key = os.environ.get("LLAMA_CLOUD_API_KEY", None)
openai_api_key = os.environ.get("OPENAI_API_KEY", None)


def test_class():
    names_of_base_classes = [b.__name__ for b in LlamaCloudIndex.__mro__]
    assert BaseManagedIndex.__name__ in names_of_base_classes


@pytest.mark.skipif(
    not base_url or not api_key, reason="No platform base url or api keyset"
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
    not base_url or not api_key, reason="No platform base url or api keyset"
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
