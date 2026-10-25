"""Serialize URL locators without resolving or fetching their media."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from llama_index.core.bridge.pydantic import AnyUrl
from llama_index.core.schema import Document, IndexNode, MediaResource, Node
from llama_index.core.storage.docstore import SimpleDocumentStore

URL = "https://example.invalid/image.png"
RESOURCE_FIELDS = (
    "text_resource",
    "image_resource",
    "audio_resource",
    "video_resource",
)


def assert_same_node(actual: Node, expected: Node) -> None:
    assert type(actual) is type(expected)
    assert actual.id_ == expected.id_
    assert actual.hash == expected.hash
    for field in RESOURCE_FIELDS:
        actual_resource = getattr(actual, field)
        expected_resource = getattr(expected, field)
        assert actual_resource == expected_resource
        if actual_resource is not None and actual_resource.url is not None:
            assert isinstance(actual_resource.url, AnyUrl)


@pytest.mark.parametrize("node_type", [Node, Document])
@pytest.mark.parametrize("field", RESOURCE_FIELDS)
def test_default_json_round_trip(node_type: type[Node], field: str) -> None:
    node = node_type(
        id_="media",
        metadata={"count": 2, "tags": ["local"], "enabled": True},
        **{field: MediaResource(url=URL, mimetype="image/png")},
    )
    encoded = node.to_json()
    assert json.loads(encoded)[field]["url"] == URL
    assert_same_node(node_type.from_json(encoded), node)
    assert json.loads(node.json()) == json.loads(encoded)
    assert json.loads(node.to_json(mode="json")) == json.loads(encoded)


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", [Node, Document])
@pytest.mark.parametrize("async_insert", [False, True], ids=["sync", "async"])
async def test_docstore_round_trip(
    node_type: type[Node], async_insert: bool, tmp_path: Path
) -> None:
    node = node_type(
        id_="media",
        metadata={"source": "local", "count": 2, "optional": None},
        **{
            field: MediaResource(url=f"{URL}?resource={field}", mimetype="image/png")
            for field in RESOURCE_FIELDS
        },
    )
    store = SimpleDocumentStore()
    if async_insert:
        await store.async_add_documents([node])
    else:
        store.add_documents([node])
    stored = store.get_node(node.id_)
    assert isinstance(stored, Node)
    assert_same_node(stored, node)
    path = str(tmp_path / "docstore.json")
    store.persist(path)
    restored = SimpleDocumentStore.from_persist_path(path)
    actual = restored.get_node(node.id_)
    assert isinstance(actual, Node)
    assert_same_node(actual, node)


@pytest.mark.parametrize("node_type", [Node, Document])
def test_python_dump_and_in_memory_metadata_types(node_type: type[Node]) -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5)
    node = node_type(
        id_="media",
        image_resource=MediaResource(url=URL),
        metadata={"pair": (1, 2), "timestamp": timestamp},
    )
    for data in (node.to_dict(), node.model_dump()):
        assert isinstance(data["image_resource"]["url"], AnyUrl)
        assert isinstance(data["metadata"]["pair"], tuple)
        assert data["metadata"]["timestamp"] is timestamp
    store = SimpleDocumentStore()
    store.add_documents([node])
    stored = store.get_node(node.id_)
    assert isinstance(stored.metadata["pair"], tuple)
    assert stored.metadata["timestamp"] == timestamp
    assert isinstance(stored.metadata["timestamp"], datetime)
    assert isinstance(stored, Node)
    assert_same_node(stored, node)


def test_no_url_path_and_dump_options() -> None:
    node = Node(
        id_="local",
        text_resource=MediaResource(text="local text"),
        image_resource=MediaResource(path=Path("unopened.png"), mimetype="image/png"),
        metadata={"hidden": "value"},
    )
    assert_same_node(Node.from_json(node.to_json()), node)
    assert node.to_dict()["image_resource"]["path"] == "unopened.png"
    excluded = json.loads(node.to_json(exclude={"metadata"}, mode="python"))
    assert "metadata" not in excluded
    assert excluded["image_resource"]["path"] == "unopened.png"
    url_node = Node(image_resource=MediaResource(url=URL))
    assert "image_resource" not in json.loads(
        url_node.to_json(exclude={"image_resource"}, mode="python")
    )


def test_index_node_with_nested_media_node(tmp_path: Path) -> None:
    media = Document(id_="inner", image_resource=MediaResource(url=URL))
    node = IndexNode(id_="outer", text="index", index_id="index", obj=media)
    restored = IndexNode.from_json(node.to_json())
    assert isinstance(restored.obj, Document)
    assert_same_node(restored.obj, media)
    store = SimpleDocumentStore()
    store.add_documents([node])
    path = str(tmp_path / "docstore.json")
    store.persist(path)
    stored = SimpleDocumentStore.from_persist_path(path).get_node(node.id_)
    assert isinstance(stored, IndexNode)
    assert isinstance(stored.obj, Document)
    assert_same_node(stored.obj, media)
