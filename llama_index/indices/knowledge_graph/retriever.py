"""KGTable Retriever."""
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.indices.query.schema import QueryBundle
from llama_index.prompts.default_prompts import DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE
from llama_index.prompts.prompts import QueryKeywordExtractPrompt
from llama_index.schema import BaseNode, MetadataMode, NodeWithScore, TextNode
from llama_index.utils import truncate_text

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE
DEFAULT_NODE_SCORE = 1000.0
GLOBAL_EXPLORE_NODE_LIMIT = 3
REL_TEXT_LIMIT = 30

logger = logging.getLogger(__name__)


class KGRetrieverMode(str, Enum):
    """Query mode enum for Knowledge Graphs.

    Can be passed as the enum struct, or as the underlying string.

    Attributes:
        KEYWORD ("keyword"): Default query mode, using keywords to find triplets.
        EMBEDDING ("embedding"): Embedding mode, using embeddings to find
            similar triplets.
        HYBRID ("hybrid"): Hyrbid mode, combining both keywords and embeddings
            to find relevant triplets.
    """

    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


class KGTableRetriever(BaseRetriever):
    """KG Table Retriever.

    Arguments are shared among subclasses.

    Args:
        query_keyword_extract_template (Optional[QueryKGExtractPrompt]): A Query
            KG Extraction
            Prompt (see :ref:`Prompt-Templates`).
        refine_template (Optional[RefinePrompt]): A Refinement Prompt
            (see :ref:`Prompt-Templates`).
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question Answering Prompt
            (see :ref:`Prompt-Templates`).
        max_keywords_per_query (int): Maximum number of keywords to extract from query.
        num_chunks_per_query (int): Maximum number of text chunks to query.
        include_text (bool): Use the document text source from each relevant triplet
            during queries.
        retriever_mode (KGRetrieverMode): Specifies whether to use keyowrds,
            embeddings, or both to find relevant triplets. Should be one of "keyword",
            "embedding", or "hybrid".
        similarity_top_k (int): The number of top embeddings to use
            (if embeddings are used).
        graph_store_query_depth (int): The depth of the graph store query.
        use_global_node_triplets (bool): Whether to get more keywords(entities) from
            text chunks matched by keywords. This helps introduce more global knowledge.
            While it's more expensive, thus to be turned off by default.
        max_knowledge_sequence (int): The maximum number of knowledge sequence to
            include in the response. By default, it's 30.
    """

    def __init__(
        self,
        index: KnowledgeGraphIndex,
        query_keyword_extract_template: Optional[QueryKeywordExtractPrompt] = None,
        max_keywords_per_query: int = 10,
        num_chunks_per_query: int = 10,
        include_text: bool = True,
        retriever_mode: Optional[KGRetrieverMode] = KGRetrieverMode.KEYWORD,
        similarity_top_k: int = 2,
        graph_store_query_depth: int = 2,
        use_global_node_triplets: bool = False,
        max_knowledge_sequence: int = REL_TEXT_LIMIT,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""

        assert isinstance(index, KnowledgeGraphIndex)
        self._index = index
        self._service_context = self._index.service_context
        self._index_struct = self._index.index_struct
        self._docstore = self._index.docstore

        self.max_keywords_per_query = max_keywords_per_query
        self.num_chunks_per_query = num_chunks_per_query
        self.query_keyword_extract_template = query_keyword_extract_template or DQKET
        self.similarity_top_k = similarity_top_k
        self._include_text = include_text
        self._retriever_mode = KGRetrieverMode(retriever_mode)

        self._graph_store = index.graph_store
        self.graph_store_query_depth = graph_store_query_depth
        self.use_global_node_triplets = use_global_node_triplets
        self.max_knowledge_sequence = max_knowledge_sequence

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        response = self._service_context.llm_predictor.predict(
            self.query_keyword_extract_template,
            max_keywords=self.max_keywords_per_query,
            question=query_str,
        )
        keywords = extract_keywords_given_response(
            response, start_token="KEYWORDS:", lowercase=False
        )
        return list(keywords)

    def _extract_rel_text_keywords(self, rel_texts: List[str]) -> List[str]:
        """Find the keywords for given rel text triplets."""
        keywords = []
        for rel_text in rel_texts:
            keyword = rel_text.split(",")[0]
            if keyword:
                keywords.append(keyword.strip("(\"'"))
        return keywords

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Get nodes for response."""
        logger.info("> Starting query: %s", query_bundle.query_str)
        node_visited = set()
        keywords = self._get_keywords(query_bundle.query_str)
        logger.info(f"> Query keywords: {keywords}")
        rel_texts = []
        cur_rel_map = {}
        chunk_indices_count: Dict[str, int] = defaultdict(int)
        if self._retriever_mode != KGRetrieverMode.EMBEDDING:
            for keyword in keywords:
                subjs = set((keyword,))
                node_ids = self._index_struct.search_node_by_keyword(keyword)
                for node_id in node_ids[:GLOBAL_EXPLORE_NODE_LIMIT]:
                    if node_id in node_visited:
                        continue

                    if self._include_text:
                        chunk_indices_count[node_id] += 1

                    node_visited.add(node_id)
                    if self.use_global_node_triplets:
                        # Get nodes from keyword search, and add them to the subjs
                        # set. This helps introduce more global knowledge into the
                        # query. While it's more expensive, thus to be turned off
                        # by default, it can be useful for some applications.

                        # TODO: we should a keyword-node_id map in IndexStruct, so that
                        # node-keywords extraction with LLM will be called only once
                        # during indexing.
                        extended_subjs = self._get_keywords(
                            self._docstore.get_node(node_id).get_content(
                                metadata_mode=MetadataMode.LLM
                            )
                        )
                        subjs.update(extended_subjs)

                rel_map = self._graph_store.get_rel_map(
                    list(subjs), self.graph_store_query_depth
                )
                logger.debug("rel_map: %s", rel_map)

                if not rel_map:
                    continue
                rel_texts.extend(
                    [
                        f"{sub} {rel_obj}"
                        for sub, rel_objs in rel_map.items()
                        for rel_obj in rel_objs
                    ]
                )
                cur_rel_map.update(rel_map)

        if (
            self._retriever_mode != KGRetrieverMode.KEYWORD
            and len(self._index_struct.embedding_dict) > 0
        ):
            query_embedding = self._service_context.embed_model.get_text_embedding(
                query_bundle.query_str
            )
            all_rel_texts = list(self._index_struct.embedding_dict.keys())

            rel_text_embeddings = [
                self._index_struct.embedding_dict[_id] for _id in all_rel_texts
            ]
            similarities, top_rel_texts = get_top_k_embeddings(
                query_embedding,
                rel_text_embeddings,
                similarity_top_k=self.similarity_top_k,
                embedding_ids=all_rel_texts,
            )
            logger.debug(
                "Found the following rel_texts+query similarites: %s", str(similarities)
            )
            logger.debug("Found the following top_k rel_texts: %s", str(rel_texts))
            rel_texts.extend(top_rel_texts)
            if self._include_text:
                keywords = self._extract_rel_text_keywords(top_rel_texts)
                nested_node_ids = [
                    self._index_struct.search_node_by_keyword(keyword)
                    for keyword in keywords
                ]
                node_ids = [_id for ids in nested_node_ids for _id in ids]
                for node_id in node_ids:
                    chunk_indices_count[node_id] += 1
        elif len(self._index_struct.embedding_dict) == 0:
            logger.error(
                "Index was not constructed with embeddings, skipping embedding usage..."
            )

        # remove any duplicates from keyword + embedding queries
        if self._retriever_mode == KGRetrieverMode.HYBRID:
            rel_texts = list(set(rel_texts))

            # remove shorter rel_texts that are substrings of longer rel_texts
            rel_texts.sort(key=len, reverse=True)
            for i in range(len(rel_texts)):
                for j in range(i + 1, len(rel_texts)):
                    if rel_texts[j] in rel_texts[i]:
                        rel_texts[j] = ""
            rel_texts = [rel_text for rel_text in rel_texts if rel_text != ""]

            # tuncate rel_texts
            rel_texts = rel_texts[: self.max_knowledge_sequence]

        sorted_chunk_indices = sorted(
            list(chunk_indices_count.keys()),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )
        sorted_chunk_indices = sorted_chunk_indices[: self.num_chunks_per_query]
        sorted_nodes = self._docstore.get_nodes(sorted_chunk_indices)

        # TMP/TODO: also filter rel_texts as nodes until we figure out better
        # abstraction
        # TODO(suo): figure out what this does
        # rel_text_nodes = [Node(text=rel_text) for rel_text in rel_texts]
        # for node_processor in self._node_postprocessors:
        #     rel_text_nodes = node_processor.postprocess_nodes(rel_text_nodes)
        # rel_texts = [node.get_content() for node in rel_text_nodes]

        sorted_nodes_with_scores = []
        for chunk_idx, node in zip(sorted_chunk_indices, sorted_nodes):
            # nodes are found with keyword mapping, give high conf to avoid cutoff
            sorted_nodes_with_scores.append(
                NodeWithScore(node=node, score=DEFAULT_NODE_SCORE)
            )
            logger.info(
                "> Querying with idx: %s: %s",
                chunk_idx,
                truncate_text(node.get_content(), 80),
            )
        # if no relationship is found, return the nodes found by keywords
        if not rel_texts:
            logger.info("> No relationships found, returning nodes found by keywords.")
            if len(sorted_nodes_with_scores) == 0:
                logger.info("> No nodes found by keywords, returning empty response.")
            return sorted_nodes_with_scores

        # add relationships as Node
        # TODO: make initial text customizable
        rel_initial_text = (
            f"The following are knowledge sequence in max depth"
            f" {self.graph_store_query_depth} "
            f"in the form of "
            f"`subject [predicate, object, predicate_next_hop, object_next_hop ...]`"
        )
        rel_info = [rel_initial_text] + rel_texts
        rel_node_info = {
            "kg_rel_texts": rel_texts,
            "kg_rel_map": cur_rel_map,
        }
        rel_text_node = TextNode(
            text="\n".join(rel_info),
            metadata=rel_node_info,
            excluded_embed_metadata_keys=["kg_rel_map", "kg_rel_texts"],
            excluded_llm_metadata_keys=["kg_rel_map", "kg_rel_texts"],
        )
        # this node is constructed from rel_texts, give high confidence to avoid cutoff
        sorted_nodes_with_scores.append(
            NodeWithScore(node=rel_text_node, score=DEFAULT_NODE_SCORE)
        )
        rel_info_text = "\n".join(rel_info)
        logger.info("> Extracted relationships: %s", rel_info_text)

        return sorted_nodes_with_scores

    def _get_metadata_for_response(
        self, nodes: List[BaseNode]
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for response."""
        for node in nodes:
            if node.metadata is None or "kg_rel_map" not in node.metadata:
                continue
            return node.metadata
        raise ValueError("kg_rel_map must be found in at least one Node.")
