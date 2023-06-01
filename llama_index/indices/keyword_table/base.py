"""Keyword-table based index.

Similar to a "hash table" in concept. LlamaIndex first tries
to extract keywords from the source text, and stores the
keywords as keys per item. It similarly extracts keywords
from the query text. Then, it tries to match those keywords to
existing keywords in the table.

"""

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Set, Union

from llama_index.async_utils import run_async_tasks
from llama_index.data_structs.data_structs import KeywordTable
from llama_index.data_structs.node import Node
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.default_prompts import (
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
from llama_index.prompts.prompts import KeywordExtractPrompt
from llama_index.storage.docstore.types import RefDocInfo

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE


class KeywordTableRetrieverMode(str, Enum):
    DEFAULT = "default"
    SIMPLE = "simple"
    RAKE = "rake"


class BaseGPTKeywordTableIndex(BaseGPTIndex[KeywordTable]):
    """GPT Keyword Table Index.

    This index extracts keywords from the text, and maps each
    keyword to the node(s) that it corresponds to. In this sense it mimicks a
    "hash table". During index construction, the keyword table is constructed
    by extracting keywords from each node and creating an internal mapping.

    During query time, the keywords are extracted from the query text, and these
    keywords are used to index into the keyword table. The retrieved nodes
    are then used to answer the query.

    Args:
        keyword_extract_template (Optional[KeywordExtractPrompt]): A Keyword
            Extraction Prompt
            (see :ref:`Prompt-Templates`).
        use_async (bool): Whether to use asynchronous calls. Defaults to False.

    """

    index_struct_cls = KeywordTable

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[KeywordTable] = None,
        service_context: Optional[ServiceContext] = None,
        keyword_extract_template: Optional[KeywordExtractPrompt] = None,
        max_keywords_per_chunk: int = 10,
        use_async: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.max_keywords_per_chunk = max_keywords_per_chunk
        self.keyword_extract_template = (
            keyword_extract_template or DEFAULT_KEYWORD_EXTRACT_TEMPLATE
        )
        # NOTE: Partially format keyword extract template here.
        self.keyword_extract_template = self.keyword_extract_template.partial_format(
            max_keywords=self.max_keywords_per_chunk
        )
        self._use_async = use_async
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            **kwargs,
        )

    def as_retriever(
        self,
        retriever_mode: Union[
            str, KeywordTableRetrieverMode
        ] = KeywordTableRetrieverMode.DEFAULT,
        **kwargs: Any,
    ) -> BaseRetriever:
        # NOTE: lazy import
        from llama_index.indices.keyword_table.retrievers import (
            KeywordTableGPTRetriever,
            KeywordTableRAKERetriever,
            KeywordTableSimpleRetriever,
        )

        if retriever_mode == KeywordTableRetrieverMode.DEFAULT:
            return KeywordTableGPTRetriever(self, **kwargs)
        elif retriever_mode == KeywordTableRetrieverMode.SIMPLE:
            return KeywordTableSimpleRetriever(self, **kwargs)
        elif retriever_mode == KeywordTableRetrieverMode.RAKE:
            return KeywordTableRAKERetriever(self, **kwargs)
        else:
            raise ValueError(f"Unknown retriever mode: {retriever_mode}")

    @abstractmethod
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""

    async def _async_extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # by default just call sync version
        return self._extract_keywords(text)

    def _add_nodes_to_index(
        self, index_struct: KeywordTable, nodes: Sequence[Node]
    ) -> None:
        """Add document to index."""
        for n in nodes:
            keywords = self._extract_keywords(n.get_text())
            index_struct.add_node(list(keywords), n)

    async def _async_add_nodes_to_index(
        self, index_struct: KeywordTable, nodes: Sequence[Node]
    ) -> None:
        """Add document to index."""
        for n in nodes:
            keywords = await self._async_extract_keywords(n.get_text())
            index_struct.add_node(list(keywords), n)

    def _build_index_from_nodes(self, nodes: Sequence[Node]) -> KeywordTable:
        """Build the index from nodes."""
        # do simple concatenation
        index_struct = KeywordTable(table={})
        if self._use_async:
            tasks = [self._async_add_nodes_to_index(index_struct, nodes)]
            run_async_tasks(tasks)
        else:
            self._add_nodes_to_index(index_struct, nodes)

        return index_struct

    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert nodes."""
        for n in nodes:
            keywords = self._extract_keywords(n.get_text())
            self._index_struct.add_node(list(keywords), n)

    def _delete_node(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        # delete node from the keyword table
        keywords_to_delete = set()
        for keyword, node_ids in self._index_struct.table.items():
            if doc_id in node_ids:
                node_ids.remove(doc_id)
                if len(node_ids) == 0:
                    keywords_to_delete.add(keyword)

        # delete keywords that have zero nodes
        for keyword in keywords_to_delete:
            del self._index_struct.table[keyword]

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        node_doc_ids_sets = list(self._index_struct.table.values())
        node_doc_ids = list(set().union(*node_doc_ids_sets))
        nodes = self.docstore.get_nodes(node_doc_ids)

        all_ref_doc_info = {}
        for node in nodes:
            ref_doc_id = node.ref_doc_id
            if not ref_doc_id:
                continue

            ref_doc_info = self.docstore.get_ref_doc_info(ref_doc_id)
            if not ref_doc_info:
                continue

            all_ref_doc_info[ref_doc_id] = ref_doc_info
        return all_ref_doc_info


class GPTKeywordTableIndex(BaseGPTKeywordTableIndex):
    """GPT Keyword Table Index.

    This index uses a GPT model to extract keywords from the text.

    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        response, formatted_prompt = self._service_context.llm_predictor.predict(
            self.keyword_extract_template,
            text=text,
        )
        keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
        return keywords

    async def _async_extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        response, formatted_prompt = await self._service_context.llm_predictor.apredict(
            self.keyword_extract_template,
            text=text,
        )
        keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
        return keywords
