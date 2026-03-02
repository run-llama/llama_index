import pytest
from typing import List
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter

# Using existing Valkey/Redis container instead of creating a new one
# Make sure Valkey or Redis Stack is running on localhost:6379


@pytest.fixture()
def dummy_embedding() -> List:
    return [0.0] * 1536


@pytest.fixture()
def turtle_test() -> dict:
    return {
        "text": "something about turtles",
        "metadata": {"animal": "turtle"},
        "question": "turtle stuff",
        "doc_id": "1234",
    }


@pytest.fixture()
def documents(turtle_test, dummy_embedding) -> List[Document]:
    """
    List of documents represents data to be embedded in the datastore.
    Minimum requirements for Documents in the /upsert endpoint's UpsertRequest.
    """
    return [
        Document(
            text=turtle_test["text"],
            metadata=turtle_test["metadata"],
            doc_id=turtle_test["doc_id"],
            embedding=dummy_embedding,
        ),
        Document(
            text="something about whales",
            metadata={"animal": "whale"},
            doc_id="5678",
            embedding=dummy_embedding,
        ),
    ]


@pytest.fixture()
def test_nodes(documents) -> List[TextNode]:
    parser = SentenceSplitter()
    return parser.get_nodes_from_documents(documents)


@pytest.fixture()
def valkey_client():
    """Fixture that provides a synchronous Valkey client."""
    try:
        from glide_sync import GlideClient as SyncGlideClient
        from glide_sync import GlideClientConfiguration as SyncGlideClientConfiguration
        from glide_shared import NodeAddress

        config = SyncGlideClientConfiguration(
            addresses=[NodeAddress("localhost", 6379)]
        )
        client = SyncGlideClient.create(config)
        yield client
        # Don't call close() as it may not be properly initialized
        # The client will be garbage collected
    except ImportError:
        pytest.skip("valkey-glide not installed")
    except Exception as e:
        pytest.skip(f"Could not create sync client: {e}")


@pytest.fixture()
async def valkey_client_async():
    """Fixture that provides an asynchronous Valkey client."""
    try:
        from glide import GlideClient, GlideClientConfiguration
        from glide_shared import NodeAddress

        config = GlideClientConfiguration(addresses=[NodeAddress("localhost", 6379)])
        client = await GlideClient.create(config)
        yield client
        await client.close()
    except ImportError:
        pytest.skip("valkey-glide not installed")
