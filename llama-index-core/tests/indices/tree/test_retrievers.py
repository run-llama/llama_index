from typing import Dict, List

from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.schema import Document, QueryBundle


def test_query(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Dict,
) -> None:
    """Test query."""
    index_kwargs, query_kwargs = struct_kwargs
    tree = TreeIndex.from_documents(documents, **index_kwargs)

    # test default query
    query_str = "What is?"
    retriever = tree.as_retriever()
    nodes = retriever.retrieve(query_str)
    assert len(nodes) == 1


def test_query_response_includes_selected_leaf_source_nodes(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Dict,
) -> None:
    """Test select leaf query response preserves selected leaf source nodes."""
    index_kwargs, query_kwargs = struct_kwargs
    tree = TreeIndex.from_documents(documents, **index_kwargs)

    query_str = "What is?"
    retriever = tree.as_retriever(**query_kwargs)
    response = retriever._query(QueryBundle(query_str))

    assert len(response.source_nodes) == 1
    assert response.source_nodes[0].node.get_content() == "Hello world."


def test_summarize_query(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Dict,
) -> None:
    """Test summarize query."""
    # create tree index without building tree
    index_kwargs, orig_query_kwargs = struct_kwargs
    index_kwargs = index_kwargs.copy()
    index_kwargs.update({"build_tree": False})
    tree = TreeIndex.from_documents(documents, **index_kwargs)

    # test retrieve all leaf
    query_str = "What is?"
    retriever = tree.as_retriever(retriever_mode="all_leaf")
    nodes = retriever.retrieve(query_str)
    assert len(nodes) == 4


def test_select_leaf_retriever_and_inserter_reject_zero_index(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Dict,
    monkeypatch,
) -> None:
    """Test that a 0 index in LLM response is treated as out-of-range rather than indexing -1."""
    index_kwargs, query_kwargs = struct_kwargs
    tree = TreeIndex.from_documents(documents, **index_kwargs)
    retriever = tree.as_retriever(**query_kwargs)

    # Patch LLM predict on the class to return "0" (1-indexed out-of-range lower bound).
    monkeypatch.setattr(type(retriever._llm), "predict", lambda *args, **kwargs: "0")
    assert retriever.retrieve("What is?") == []
    resp = retriever._query(QueryBundle("What is?"))
    assert resp.response == "0"
    assert resp.source_nodes == []

    # Inserting when LLM returns "0" should fall back to inserting under parent rather than child[-1].
    graph = tree.index_struct
    last_root_id = graph.root_nodes[max(graph.root_nodes)]

    def descendants(node_id: str) -> set:
        out = set()
        for child_id in graph.node_id_to_children_ids.get(node_id, []):
            out |= {child_id} | descendants(child_id)
        return out

    tree.insert(Document(text="Inserted with zero response.", id_="zero_doc"))
    new_ids = {
        node_id
        for node_id in graph.all_nodes.values()
        if tree.docstore.get_node(node_id).ref_doc_id == "zero_doc"
    }
    assert new_ids
    assert not new_ids & descendants(last_root_id)
