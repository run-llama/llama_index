from typing import Any, Sequence, Set, Union

from llama_index.data_structs.data_structs import KeywordTable
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.keyword_table.base import (
    BaseKeywordTableIndex,
    KeywordTableRetrieverMode,
)
from llama_index.schema import BaseNode
from llama_index.utils import get_tqdm_iterable


class CodeHierarchyKeywordTableIndex(BaseKeywordTableIndex):
    """A keyword table made specifically to work with the code hierarchy node parser.

    Similar to SimpleKeywordTableIndex, but doesn't use GPT to extract keywords.
    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        raise NotImplementedError(
            "You should not be calling this method "
            "from within CodeHierarchyKeywordTableIndex."
        )

    def _extract_keywords_from_node(self, node: BaseNode) -> Set[str]:
        keywords = []
        keywords.append(node.node_id)
        file_path = node.metadata["filepath"]
        module_path = file_path.replace("/", ".").lstrip(".").rstrip(".py")
        keywords.append(module_path)
        # Add the last scope name and signature to the keywords
        if node.metadata["inclusive_scopes"]:
            keywords.append(node.metadata["inclusive_scopes"][-1]["name"])
            keywords.append(node.metadata["inclusive_scopes"][-1]["signature"])

        return {k.lower() for k in keywords}

    def _add_nodes_to_index(
        self,
        index_struct: KeywordTable,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        """Add document to index."""
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Extracting keywords from nodes"
        )
        for n in nodes_with_progress:
            index_struct.add_node(list(self._extract_keywords_from_node(n)), n)

    async def _async_add_nodes_to_index(
        self,
        index_struct: KeywordTable,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        """Add document to index."""
        return self._add_nodes_to_index(index_struct, nodes, show_progress)

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert nodes."""
        for n in nodes:
            self._index_struct.add_node(list(self._extract_keywords_from_node(n)), n)

    def as_retriever(
        self,
        retriever_mode: Union[str, KeywordTableRetrieverMode, None] = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        if retriever_mode is None:
            from llama_index.indices.keyword_table.base import KeywordTableRetrieverMode

            retriever_mode = KeywordTableRetrieverMode.SIMPLE
        return super().as_retriever(retriever_mode=retriever_mode, **kwargs)
