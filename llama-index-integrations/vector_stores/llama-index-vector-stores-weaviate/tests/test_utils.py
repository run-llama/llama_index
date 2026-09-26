"""
Unit tests for the Weaviate <-> LlamaIndex serializers.

These do not need a running Weaviate instance: ``to_node`` is a pure function
over the ``__dict__`` of a ``weaviate.outputs.query.Object``, which is exactly
what ``WeaviateVectorStore.parse_query_result`` passes it.
"""

import json
from uuid import UUID

from llama_index.core.schema import NodeRelationship, TextNode
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.weaviate.utils import to_node

from weaviate.outputs.query import MetadataReturn, Object

TEST_UUID = UUID("11111111-2222-3333-4444-555555555555")


def make_entry(properties: dict) -> dict:
    """Build the dict that ``parse_query_result`` hands to ``to_node``."""
    return Object(
        uuid=TEST_UUID,
        metadata=MetadataReturn(score=0.5),
        properties=properties,
        references=None,
        vector={"default": [0.1, 0.2, 0.3]},
        collection="TestCollection",
    ).__dict__


def test_to_node_llama_index_collection():
    """A collection written by llama-index takes the non-legacy path."""
    node = TextNode(
        text="Hello world.",
        id_=str(TEST_UUID),
        metadata={"title": "Paper A"},
    )
    properties = {"text": node.text}
    properties.update(
        node_to_metadata_dict(node, remove_text=True, flat_metadata=False)
    )

    parsed = to_node(make_entry(properties), text_key="text")

    assert parsed.node_id == str(TEST_UUID)
    assert parsed.text == "Hello world."
    assert parsed.metadata == {"title": "Paper A"}
    assert parsed.embedding == [0.1, 0.2, 0.3]


def test_to_node_pre_existing_collection():
    """
    A collection not written by llama-index has no ``_node_content``.

    Regression test for #14857: the legacy fallback used to be handed the whole
    Weaviate object wrapper, so every node came back with
    ``{'collection', 'metadata', 'properties', 'references', 'uuid', 'vector'}``
    as its metadata while the real properties were buried one level down.
    """
    entry = make_entry({"text": "Paper A body", "title": "Paper A", "year": 2020})

    node = to_node(entry, text_key="text")

    assert node.metadata == {"title": "Paper A", "year": 2020}
    assert node.node_id == str(TEST_UUID)
    assert node.text == "Paper A body"
    assert node.embedding == [0.1, 0.2, 0.3]


def test_to_node_legacy_llama_index_collection():
    """Legacy llama-index properties are still unpacked, not returned as metadata."""
    entry = make_entry(
        {
            "text": "Paper A body",
            "title": "Paper A",
            "node_info": json.dumps({"start": 0, "end": 12}),
            "relationships": json.dumps(
                {NodeRelationship.PARENT.value: "parent-doc-id"}
            ),
            "doc_id": "parent-doc-id",
        }
    )

    node = to_node(entry, text_key="text")

    assert node.metadata == {"title": "Paper A"}
    assert node.start_char_idx == 0
    assert node.end_char_idx == 12
    assert node.relationships[NodeRelationship.PARENT].node_id == "parent-doc-id"
