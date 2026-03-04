"""Data struct for document summary index."""

from dataclasses import dataclass, field
from typing import Dict, List

from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.data_structs.struct_type import IndexStructType
from llama_index.core.schema import BaseNode


@dataclass
class IndexDocumentSummary(IndexStruct):
    """
    A simple struct containing a mapping from summary node_id to doc node_ids.

    Also mapping vice versa.

    """

    summary_id_to_node_ids: Dict[str, List[str]] = field(default_factory=dict)
    node_id_to_summary_id: Dict[str, str] = field(default_factory=dict)

    # track mapping from doc id to node summary id
    doc_id_to_summary_id: Dict[str, str] = field(default_factory=dict)

    def add_summary_and_nodes(
        self,
        summary_node: BaseNode,
        nodes: List[BaseNode],
    ) -> str:
        """Add node and summary."""
        summary_id = summary_node.node_id
        ref_doc_id = summary_node.ref_doc_id
        if ref_doc_id is None:
            raise ValueError(
                "ref_doc_id of node cannot be None when building a document "
                "summary index"
            )
        self.doc_id_to_summary_id[ref_doc_id] = summary_id

        for node in nodes:
            node_id = node.node_id
            if summary_id not in self.summary_id_to_node_ids:
                self.summary_id_to_node_ids[summary_id] = []
            self.summary_id_to_node_ids[summary_id].append(node_id)

            self.node_id_to_summary_id[node_id] = summary_id

        return summary_id

    @property
    def summary_ids(self) -> List[str]:
        """Get summary ids."""
        return list(self.summary_id_to_node_ids.keys())

    def delete(self, doc_id: str) -> None:
        """Delete a document and its nodes."""
        summary_id = self.doc_id_to_summary_id[doc_id]
        del self.doc_id_to_summary_id[doc_id]
        node_ids = self.summary_id_to_node_ids[summary_id]
        for node_id in node_ids:
            del self.node_id_to_summary_id[node_id]
        del self.summary_id_to_node_ids[summary_id]

    def delete_nodes(self, node_ids: List[str]) -> None:
        for node_id in node_ids:
            summary_id = self.node_id_to_summary_id[node_id]
            self.summary_id_to_node_ids[summary_id].remove(node_id)
            del self.node_id_to_summary_id[node_id]

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.DOCUMENT_SUMMARY
