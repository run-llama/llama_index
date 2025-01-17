from typing import Tuple
from llama_cloud import (
    AutoTransformConfig,
    PipelineCreate,
    PipelineFileCreate,
    ProjectCreate,
    CompositeRetrievalMode,
)
from llama_index.indices.managed.llama_cloud import (
    LlamaCloudIndex,
    LlamaCloudCompositeRetriever,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document
import os
import pytest
from uuid import uuid4
from llama_cloud.client import LlamaCloud
from llama_index.core.indices.managed.base import BaseManagedIndex
import tempfile

base_url = os.environ.get("LLAMA_CLOUD_BASE_URL", None)
api_key = os.environ.get("LLAMA_CLOUD_API_KEY", None)
openai_api_key = os.environ.get("LLAMA_CLOUD_OPENAI_API_KEY", None)
organization_id = os.environ.get("LLAMA_CLOUD_ORGANIZATION_ID", None)
project_name = os.environ.get("LLAMA_CLOUD_PROJECT_NAME", "framework_integration_test")


@pytest.fixture()
def remote_file() -> Tuple[str, str]:
    test_file_url = "https://www.google.com/robots.txt"
    test_file_name = "google_robots.txt"
    return test_file_url, test_file_name


def _setup_empty_index(
    client: LlamaCloud,
) -> LlamaCloudIndex:
    # create project if it doesn't exist
    project_create = ProjectCreate(name=project_name)
    project = client.projects.upsert_project(
        organization_id=organization_id, request=project_create
    )

    # create pipeline
    pipeline_create = PipelineCreate(
        name="test_empty_index_" + str(uuid4()),
        embedding_config={"type": "OPENAI_EMBEDDING", "component": OpenAIEmbedding()},
        transform_config=AutoTransformConfig(),
    )
    return client.pipelines.upsert_pipeline(
        project_id=project.id, request=pipeline_create
    )


def _setup_index_with_file(
    client: LlamaCloud, remote_file: Tuple[str, str]
) -> LlamaCloudIndex:
    # create project if it doesn't exist
    project_create = ProjectCreate(name=project_name)
    project = client.projects.upsert_project(
        organization_id=organization_id, request=project_create
    )

    # create pipeline
    pipeline_create = PipelineCreate(
        name="test_index_with_file_" + str(uuid4()),
        embedding_config={"type": "OPENAI_EMBEDDING", "component": OpenAIEmbedding()},
        transform_config=AutoTransformConfig(),
    )
    pipeline = client.pipelines.upsert_pipeline(
        project_id=project.id, request=pipeline_create
    )

    # upload file to pipeline
    test_file_url, test_file_name = remote_file
    file = client.files.upload_file_from_url(
        project_id=project.id, url=test_file_url, name=test_file_name
    )

    # add file to pipeline
    pipeline_file_create = PipelineFileCreate(file_id=file.id)
    client.pipelines.add_files_to_pipeline(
        pipeline_id=pipeline.id, request=[pipeline_file_create]
    )

    return pipeline


def test_class():
    names_of_base_classes = [b.__name__ for b in LlamaCloudIndex.__mro__]
    assert BaseManagedIndex.__name__ in names_of_base_classes


def test_conflicting_index_identifiers():
    with pytest.raises(ValueError):
        LlamaCloudIndex(name="test", pipeline_id="test", index_id="test")


@pytest.mark.skipif(
    not base_url or not api_key, reason="No platform base url or api key set"
)
@pytest.mark.skipif(not openai_api_key, reason="No openai api key set")
@pytest.mark.integration()
def test_resolve_index_with_id(remote_file):
    """Test that we can instantiate an index with a given id."""
    client = LlamaCloud(token=api_key, base_url=base_url)
    pipeline = _setup_index_with_file(client, remote_file)

    index = LlamaCloudIndex(
        pipeline_id=pipeline.id,
        api_key=api_key,
        base_url=base_url,
    )
    assert index is not None

    index.wait_for_completion()
    retriever = index.as_retriever()

    nodes = retriever.retrieve("Hello world.")
    assert len(nodes) > 0


@pytest.mark.skipif(
    not base_url or not api_key, reason="No platform base url or api key set"
)
@pytest.mark.skipif(not openai_api_key, reason="No openai api key set")
@pytest.mark.integration()
def test_resolve_index_with_name(remote_file):
    """Test that we can instantiate an index with a given name."""
    client = LlamaCloud(token=api_key, base_url=base_url)
    pipeline = _setup_index_with_file(client, remote_file)

    index = LlamaCloudIndex(
        name=pipeline.name,
        project_name=project_name,
        organization_id=organization_id,
        api_key=api_key,
        base_url=base_url,
    )
    assert index is not None

    index.wait_for_completion()
    retriever = index.as_retriever()

    nodes = retriever.retrieve("Hello world.")
    assert len(nodes) > 0


@pytest.mark.skipif(
    not base_url or not api_key, reason="No platform base url or api key set"
)
@pytest.mark.skipif(not openai_api_key, reason="No openai api key set")
@pytest.mark.integration()
def test_upload_file():
    pipeline = _setup_empty_index(LlamaCloud(token=api_key, base_url=base_url))

    index = LlamaCloudIndex(
        name=pipeline.name,
        project_name=project_name,
        organization_id=organization_id,
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
def test_upload_file_from_url(remote_file):
    pipeline = _setup_empty_index(LlamaCloud(token=api_key, base_url=base_url))

    index = LlamaCloudIndex(
        name=pipeline.name,
        project_name=project_name,
        organization_id=organization_id,
        api_key=api_key,
        base_url=base_url,
    )

    # Define a URL to a file for testing
    test_file_url, test_file_name = remote_file

    # Upload the file from the URL
    file_id = index.upload_file_from_url(
        file_name=test_file_name, url=test_file_url, verbose=True
    )
    assert file_id is not None

    # Verify the file is part of the index
    docs = index.ref_doc_info
    assert any(test_file_name == doc.metadata.get("file_name") for doc in docs.values())


@pytest.mark.skipif(
    not base_url or not api_key, reason="No platform base url or api key set"
)
@pytest.mark.skipif(not openai_api_key, reason="No openai api key set")
@pytest.mark.integration()
def test_index_from_documents():
    documents = [
        Document(text="Hello world.", doc_id="1", metadata={"source": "test"}),
    ]
    index = LlamaCloudIndex.from_documents(
        documents=documents,
        name=f"test pipeline {uuid4()}",
        project_name=project_name,
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
def test_composite_retriever():
    """Test the LlamaCloudCompositeRetriever with multiple indices."""
    # Create first index with documents
    documents1 = [
        Document(
            text="Hello world from index 1.", doc_id="1", metadata={"source": "index1"}
        ),
    ]
    index1 = LlamaCloudIndex.from_documents(
        documents=documents1,
        name=f"test pipeline 1 {uuid4()}",
        project_name=project_name,
        api_key=api_key,
        base_url=base_url,
        organization_id=organization_id,
        verbose=True,
    )

    # Create second index with documents
    documents2 = [
        Document(
            text="Hello world from index 2.", doc_id="2", metadata={"source": "index2"}
        ),
    ]
    index2 = LlamaCloudIndex.from_documents(
        documents=documents2,
        name=f"test pipeline 2 {uuid4()}",
        project_name=project_name,
        api_key=api_key,
        base_url=base_url,
        organization_id=organization_id,
        verbose=True,
    )

    # Create a composite retriever
    retriever = LlamaCloudCompositeRetriever(
        name="composite_retriever_test",
        project_name=project_name,
        api_key=api_key,
        base_url=base_url,
        create_if_not_exists=True,
        mode=CompositeRetrievalMode.FULL,
        rerank_top_n=5,
    )

    # Attach indices to the composite retriever
    retriever.add_index(index1, description="Information from index 1.")
    retriever.add_index(index2, description="Information from index 2.")

    # Retrieve nodes using the composite retriever
    nodes = retriever.retrieve("Hello world.")

    # Assertions to verify the retrieval
    assert len(nodes) >= 2
    assert any(n.node.metadata["pipeline_id"] == index1.id for n in nodes)
    assert any(n.node.metadata["pipeline_id"] == index1.id for n in nodes)
