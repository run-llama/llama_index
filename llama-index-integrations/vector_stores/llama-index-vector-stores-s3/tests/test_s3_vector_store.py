import boto3
import os
from llama_index.core.vector_stores.types import VectorStoreQuery
import pytest
import uuid
from typing import List

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter
from llama_index.vector_stores.s3 import S3VectorStore

bucket_name = os.getenv("S3_BUCKET_NAME", "test-bucket")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")
region_name = os.getenv("AWS_REGION", "us-east-2")

should_skip = not all([aws_access_key_id, aws_secret_access_key])


@pytest.fixture
def vector_store():
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )

    index_name = str(uuid.uuid4())
    s3_vector_store = S3VectorStore.create_index_from_bucket(
        bucket_name_or_arn=bucket_name,
        index_name=index_name,
        dimension=1536,
        sync_session=session,
    )
    try:
        yield s3_vector_store
    finally:
        session.client("s3vectors").delete_index(
            vectorBucketName=bucket_name, indexName=index_name
        )


@pytest.fixture
def nodes() -> List[TextNode]:
    return [
        TextNode(
            id_="1",
            text="Hello, world 1!",
            metadata={"key": "1"},
            embedding=[0.1] + [0.0] * 1535,
        ),
        TextNode(
            id_="2",
            text="Hello, world 2!",
            metadata={"key": "2"},
            embedding=[0.0] * 1535 + [0.5],
        ),
        TextNode(
            id_="3",
            text="Hello, world 3!",
            metadata={"key": "3"},
            embedding=[0.9] + [0.3] * 1535,
        ),
    ]


@pytest.mark.skipif(should_skip, reason="AWS credentials not set")
def test_basic_flow(vector_store: S3VectorStore, nodes: List[TextNode]):
    vector_store.add(nodes)

    nodes = vector_store.get_nodes(node_ids=["1", "2", "3"])
    assert len(nodes) == 3

    query = VectorStoreQuery(
        query_embedding=[0.1] + [0.0] * 1535,
        similarity_top_k=2,
    )

    results = vector_store.query(query)
    assert len(results.nodes) == 2
    assert results.nodes[0].text == "Hello, world 1!"
    assert results.nodes[1].text == "Hello, world 3!"
    assert results.nodes[0].metadata["key"] == "1"
    assert results.nodes[1].metadata["key"] == "3"
    assert results.similarities[0] > results.similarities[1]


@pytest.mark.skipif(should_skip, reason="AWS credentials not set")
@pytest.mark.asyncio
async def test_async_flow(vector_store: S3VectorStore, nodes: List[TextNode]):
    await vector_store.async_add(nodes)

    nodes = await vector_store.aget_nodes(node_ids=["1", "2", "3"])
    assert len(nodes) == 3

    query = VectorStoreQuery(
        query_embedding=[0.1] + [0.0] * 1535,
        similarity_top_k=2,
    )

    results = await vector_store.aquery(query)
    assert len(results.nodes) == 2
    assert results.nodes[0].text == "Hello, world 1!"
    assert results.nodes[1].text == "Hello, world 3!"
    assert results.nodes[0].metadata["key"] == "1"
    assert results.nodes[1].metadata["key"] == "3"
    assert results.similarities[0] > results.similarities[1]


@pytest.mark.skipif(should_skip, reason="AWS credentials not set")
def test_text_field(vector_store: S3VectorStore, nodes: List[TextNode]):
    vectors = [
        {
            "key": node.id_,
            "metadata": {
                "my_text": node.text,
                **node.metadata,
            },
            "data": {"float32": node.embedding},
        }
        for node in nodes
    ]

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )

    # populate with custom vectors
    session.client("s3vectors").put_vectors(
        vectorBucketName=bucket_name,
        indexName=vector_store.index_name_or_arn,
        vectors=vectors,
    )

    vector_store.text_field = "my_text"
    results = vector_store.get_nodes(node_ids=["1", "2", "3"])
    assert len(results) == 3


@pytest.mark.skipif(should_skip, reason="AWS credentials not set")
@pytest.mark.asyncio
async def test_filtering(vector_store: S3VectorStore, nodes: List[TextNode]):
    await vector_store.async_add(nodes)
    results = await vector_store.aquery(
        VectorStoreQuery(
            filters=MetadataFilters(filters=[MetadataFilter(key="key", value="1")]),
            query_embedding=[0.1] + [0.0] * 1535,
            similarity_top_k=2,
        )
    )
    assert len(results.nodes) == 1
    assert results.nodes[0].text == "Hello, world 1!"
