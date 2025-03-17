"""Embedded Tables Retriever w/ Unstructured.IO."""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.readers.file.flat import FlatReader


class EmbeddedTablesUnstructuredRetrieverPack(BaseLlamaPack):
    """
    Embedded Tables + Unstructured.io Retriever pack.

    Use unstructured.io to parse out embedded tables from an HTML document, build
    a node graph, and then run our recursive retriever against that.

    **NOTE**: must take in a single HTML file.

    """

    def __init__(
        self,
        html_path: str,
        nodes_save_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.reader = FlatReader()

        docs = self.reader.load_data(Path(html_path))

        self.node_parser = UnstructuredElementNodeParser()
        if nodes_save_path is None or not os.path.exists(nodes_save_path):
            raw_nodes = self.node_parser.get_nodes_from_documents(docs)
            pickle.dump(raw_nodes, open(nodes_save_path, "wb"))
        else:
            raw_nodes = pickle.load(open(nodes_save_path, "rb"))

        base_nodes, node_mappings = self.node_parser.get_base_nodes_and_mappings(
            raw_nodes
        )
        # construct top-level vector index + query engine
        vector_index = VectorStoreIndex(base_nodes)
        vector_retriever = vector_index.as_retriever(similarity_top_k=1)
        self.recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=node_mappings,
            verbose=True,
        )
        self.query_engine = RetrieverQueryEngine.from_args(self.recursive_retriever)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "node_parser": self.node_parser,
            "recursive_retriever": self.recursive_retriever,
            "query_engine": self.query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)
