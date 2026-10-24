from pathlib import Path
from typing import Any, Sequence

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.ingestion.cache import DEFAULT_CACHE_NAME
from llama_index.core.ingestion.pipeline import get_transformation_hash
from llama_index.core.schema import BaseNode, TextNode, TransformComponent


class CountingTransform(TransformComponent):
    _calls: int = PrivateAttr(default=0)

    @property
    def calls(self) -> int:
        return self._calls

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        self._calls += 1
        return [
            TextNode(id_=node.node_id, text=node.get_content() + " transformed")
            for node in nodes
        ]


@pytest.mark.parametrize("collection", [DEFAULT_CACHE_NAME, "tenant"])
@pytest.mark.parametrize("explicit_fs", [False, True])
@pytest.mark.parametrize("same_pipeline", [False, True])
def test_load_preserves_configured_cache_collection(
    tmp_path: Path, collection: str, explicit_fs: bool, same_pipeline: bool
) -> None:
    transform = CountingTransform()
    pipeline = IngestionPipeline(
        transformations=[transform], cache=IngestionCache(collection=collection)
    )
    nodes = [TextNode(id_="source", text="same input")]
    cache_key = get_transformation_hash(nodes, transform)
    pipeline.run(nodes=nodes)
    pipeline.run(nodes=nodes)
    assert transform.calls == 1
    assert get_transformation_hash(nodes, transform) == cache_key

    fs = MemoryFileSystem() if explicit_fs else None
    persist_dir = str(tmp_path / "pipeline")
    pipeline.persist(persist_dir=persist_dir, fs=fs)
    if same_pipeline:
        target = pipeline
        target_transform = transform
    else:
        target_transform = CountingTransform()
        target = IngestionPipeline(
            transformations=[target_transform],
            cache=IngestionCache(collection=collection),
        )

    target.load(persist_dir=persist_dir, fs=fs)
    persisted_nodes = target.cache.get(cache_key, collection=collection)
    assert persisted_nodes is not None
    assert [node.get_content() for node in persisted_nodes] == [
        "same input transformed"
    ]
    calls_before = target_transform.calls
    result = target.run(nodes=nodes)

    assert [node.get_content() for node in result] == ["same input transformed"]
    assert target_transform.calls == calls_before
    assert target.cache.get(cache_key) is not None
    assert target.cache.collection == collection
