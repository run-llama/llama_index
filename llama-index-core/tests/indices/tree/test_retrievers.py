from typing import Dict, List

from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.schema import Document


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
