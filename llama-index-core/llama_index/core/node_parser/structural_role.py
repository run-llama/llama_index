"""Structural Role Node Parser.

Annotates TextNode chunks with a `structural_role` metadata field during parsing.
This enables per-role retrieval strategies in heterogeneous document corpora
(e.g., financial filings where a single document contains narrative sections,
tables, and footnotes that benefit from different retrieval approaches).

Addresses: https://github.com/run-llama/llama_index/issues/22032

Example::

    from llama_index.core.node_parser.structural_role import (
        StructuralRoleNodeParser,
        StructuralRole,
    )
    from llama_index.core.node_parser import SentenceSplitter

    # Wrap any existing node parser to add structural role annotation
    parser = StructuralRoleNodeParser(
        base_parser=SentenceSplitter(chunk_size=512),
        role_classifier=None,  # Uses heuristic classifier by default
    )

    nodes = parser.get_nodes_from_documents(documents)
    # Each node now has node.metadata["structural_role"] set to one of:
    # "narrative", "table", "footnote", "header", "list", "unknown"

    # Use structural_role in retriever config:
    # retriever_config_by_role = {
    #     "table": {"retrieval_mode": "exact_match"},
    #     "narrative": {"retrieval_mode": "semantic"},
    #     "footnote": {"retrieval_mode": "hybrid", "alpha": 0.3},
    # }
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence

from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, Document, TextNode


class StructuralRole(str, Enum):
    """Structural role of a text chunk within a document.

    Used to select per-role retrieval strategies in heterogeneous corpora.
    """

    NARRATIVE = "narrative"
    """Prose/narrative text: MD&A, risk factors, earnings call discussions."""

    TABLE = "table"
    """Tabular data: financial statements, data tables, comparison matrices."""

    FOOTNOTE = "footnote"
    """Footnotes, endnotes, or parenthetical disclosures."""

    HEADER = "header"
    """Section headers, titles, or document metadata."""

    LIST = "list"
    """Bullet lists, numbered lists, or enumerated items."""

    UNKNOWN = "unknown"
    """Role could not be determined reliably."""


# Metadata key used on TextNode
STRUCTURAL_ROLE_KEY = "structural_role"

# Metadata key for the hierarchical provenance path (e.g., "10k/md_and_a/paragraph_3")
PROVENANCE_PATH_KEY = "provenance_path"


def _heuristic_role_classifier(text: str) -> StructuralRole:
    """Lightweight heuristic classifier for structural role.

    Determines role from surface features of the text chunk.
    For production use, replace with a trained classifier via the
    `role_classifier` parameter of `StructuralRoleNodeParser`.
    """
    stripped = text.strip()

    if not stripped:
        return StructuralRole.UNKNOWN

    # Table detection: presence of pipe characters or tab-separated numeric rows
    lines = stripped.splitlines()
    if len(lines) >= 2:
        pipe_lines = sum(1 for line in lines if "|" in line)
        if pipe_lines / len(lines) > 0.4:
            return StructuralRole.TABLE

        # TSV-style numeric table (e.g., financial statements)
        numeric_tab_lines = sum(
            1
            for line in lines
            if re.search(r"\t[\d,.$%-]+", line) or re.search(r"  {2,}[\d,.$%-]+", line)
        )
        if numeric_tab_lines / len(lines) > 0.5:
            return StructuralRole.TABLE

    # Footnote detection: starts with superscript-like marker or "Note"
    if re.match(r"^(\(\d+\)|\d+\.\s|Note\s\d|\*+\s)", stripped, re.IGNORECASE):
        return StructuralRole.FOOTNOTE

    # Header detection: short, no period, all-caps or title case
    if len(stripped) < 120 and not stripped.endswith("."):
        word_count = len(stripped.split())
        if word_count <= 12:
            if stripped.isupper() or stripped.istitle():
                return StructuralRole.HEADER

    # List detection: majority of lines start with bullet or number
    list_lines = sum(
        1 for line in lines if re.match(r"^(\s*[-*•]\s|\s*\d+[.):]\s)", line)
    )
    if len(lines) > 1 and list_lines / len(lines) > 0.5:
        return StructuralRole.LIST

    return StructuralRole.NARRATIVE


class StructuralRoleNodeParser(NodeParser):
    """Wraps any NodeParser to annotate nodes with structural role metadata.

    This parser is a thin decorator: it delegates actual chunking to a
    `base_parser` and then annotates each resulting `TextNode` with a
    `structural_role` metadata field.

    The `structural_role` field enables retrieval pipelines to select
    different retrieval strategies per chunk type without re-indexing:

    - ``narrative``  → semantic (vector) search
    - ``table``      → exact/keyword (BM25) search
    - ``footnote``   → hybrid with low alpha (BM25-heavy)
    - ``header``     → metadata filter or skip

    Args:
        base_parser: Any NodeParser to use for chunking.
        role_classifier: Callable that takes a text string and returns a
            StructuralRole. Defaults to the built-in heuristic classifier.
        include_provenance_path: If True, also sets a ``provenance_path``
            metadata field using available document/node path metadata.
    """

    base_parser: NodeParser
    role_classifier: Optional[Callable[[str], StructuralRole]] = None
    include_provenance_path: bool = False

    class Config:
        arbitrary_types_allowed = True

    def _classify_node(self, node: TextNode) -> StructuralRole:
        """Classify a single TextNode and return its structural role."""
        classifier = self.role_classifier or _heuristic_role_classifier
        return classifier(node.get_content())

    def _annotate_provenance(
        self, node: TextNode, source_doc: Optional[Document] = None
    ) -> None:
        """Set provenance_path metadata on node if available."""
        parts = []
        if source_doc and source_doc.metadata.get("file_name"):
            parts.append(source_doc.metadata["file_name"])
        if node.metadata.get("section_header"):
            parts.append(node.metadata["section_header"])
        if node.metadata.get("page_label"):
            parts.append(f"p{node.metadata['page_label']}")
        if parts:
            node.metadata[PROVENANCE_PATH_KEY] = "/".join(parts)

    def _postprocess_parsed_nodes(
        self,
        nodes: List[BaseNode],
        parent_doc_map: Optional[dict] = None,
    ) -> List[BaseNode]:
        """Annotate parsed nodes with structural_role and optionally provenance_path."""
        annotated: List[BaseNode] = []
        for node in nodes:
            if isinstance(node, TextNode):
                role = self._classify_node(node)
                node.metadata[STRUCTURAL_ROLE_KEY] = role.value
                if self.include_provenance_path:
                    source_doc = (
                        parent_doc_map.get(node.source_node.node_id)
                        if parent_doc_map and node.source_node
                        else None
                    )
                    self._annotate_provenance(node, source_doc)
            annotated.append(node)
        return annotated

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse documents and annotate resulting nodes with structural roles."""
        doc_map = {doc.doc_id: doc for doc in documents}
        nodes = self.base_parser.get_nodes_from_documents(
            documents, show_progress=show_progress, **kwargs
        )
        return self._postprocess_parsed_nodes(nodes, parent_doc_map=doc_map)

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse nodes and annotate with structural roles."""
        parsed = self.base_parser._parse_nodes(
            nodes, show_progress=show_progress, **kwargs
        )
        return self._postprocess_parsed_nodes(parsed)
