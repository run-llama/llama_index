import pytest

from llama_index.core.utils import concat_dirs


@pytest.mark.parametrize("basename", ["docstore.json", "nested/cache.json", ""])
def test_empty_directory_prefix_preserves_relative_filename(basename: str) -> None:
    assert concat_dirs("", basename) == basename


@pytest.mark.parametrize("dirname", ["storage", "storage/"])
def test_nonempty_directory_prefix_is_unchanged(dirname: str) -> None:
    assert concat_dirs(dirname, "docstore.json") == "storage/docstore.json"


def test_pipeline_persist_and_load_with_empty_directory(tmp_path, monkeypatch) -> None:
    from fsspec.implementations.local import LocalFileSystem
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.schema import TextNode
    from llama_index.core.storage.docstore import SimpleDocumentStore

    monkeypatch.chdir(tmp_path)
    node = TextNode(id_="node-1", text="synthetic cached passage")
    docstore = SimpleDocumentStore()
    docstore.add_documents([node])
    pipeline = IngestionPipeline(transformations=[], docstore=docstore)
    pipeline.cache.put("cache-key", [node])
    filesystem = LocalFileSystem()
    pipeline.persist(persist_dir="", fs=filesystem)

    restored = IngestionPipeline(transformations=[])
    restored.load(persist_dir="", fs=filesystem)
    assert restored.cache.get("cache-key")[0].text == node.text
    assert restored.docstore.get_node(node.node_id).text == node.text
