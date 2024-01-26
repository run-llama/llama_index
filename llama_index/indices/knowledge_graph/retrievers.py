"""KG Retrievers."""
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from llama_index.callbacks.base import CallbackManager
from llama_index.core.base_retriever import BaseRetriever
from llama_index.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.prompts import BasePromptTemplate, PromptTemplate, PromptType
from llama_index.prompts.default_prompts import DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE
from llama_index.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.service_context import ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.utils import print_text, truncate_text

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
        refine_template (Optional[BasePromptTemplate]): A Refinement Prompt
            (see :ref:`Prompt-Templates`).
        text_qa_template (Optional[BasePromptTemplate]): A Question Answering Prompt
            (see :ref:`Prompt-Templates`).
        max_keywords_per_query (int): Maximum number of keywords to extract from query.
        num_chunks_per_query (int): Maximum number of text chunks to query.
        include_text (bool): Use the document text source from each relevant triplet
            during queries.
        retriever_mode (KGRetrieverMode): Specifies whether to use keywords,
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
        query_keyword_extract_template: Optional[BasePromptTemplate] = None,
        max_keywords_per_query: int = 10,
        num_chunks_per_query: int = 10,
        include_text: bool = True,
        retriever_mode: Optional[KGRetrieverMode] = KGRetrieverMode.KEYWORD,
        similarity_top_k: int = 2,
        graph_store_query_depth: int = 2,
        use_global_node_triplets: bool = False,
        max_knowledge_sequence: int = REL_TEXT_LIMIT,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
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
        self._verbose = kwargs.get("verbose", False)
        refresh_schema = kwargs.get("refresh_schema", False)
        try:
            self._graph_schema = self._graph_store.get_schema(refresh=refresh_schema)
        except NotImplementedError:
            self._graph_schema = ""
        except Exception as e:
            logger.warning(f"Failed to get graph schema: {e}")
            self._graph_schema = ""
        super().__init__(
            callback_manager=callback_manager, object_map=object_map, verbose=verbose
        )

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        response = self._service_context.llm.predict(
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
        node_visited = set()
        keywords = self._get_keywords(query_bundle.query_str)
        if self._verbose:
            print_text(f"Extracted keywords: {keywords}\n", color="green")
        rel_texts = []
        cur_rel_map = {}
        chunk_indices_count: Dict[str, int] = defaultdict(int)
        if self._retriever_mode != KGRetrieverMode.EMBEDDING:
            for keyword in keywords:
                subjs = {keyword}
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
                logger.debug(f"rel_map: {rel_map}")

                if not rel_map:
                    continue
                rel_texts.extend(
                    [
                        str(rel_obj)
                        for rel_objs in rel_map.values()
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
                f"Found the following rel_texts+query similarites: {similarities!s}"
            )
            logger.debug(f"Found the following top_k rel_texts: {rel_texts!s}")
            rel_texts.extend(top_rel_texts)

        elif len(self._index_struct.embedding_dict) == 0:
            logger.warning(
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

            # truncate rel_texts
            rel_texts = rel_texts[: self.max_knowledge_sequence]

        # When include_text = True just get the actual content of all the nodes
        # (Nodes with actual keyword match, Nodes which are found from the depth search and Nodes founnd from top_k similarity)
        if self._include_text:
            keywords = self._extract_rel_text_keywords(
                rel_texts
            )  # rel_texts will have all the Triplets retrieved with respect to the Query
            nested_node_ids = [
                self._index_struct.search_node_by_keyword(keyword)
                for keyword in keywords
            ]
            node_ids = [_id for ids in nested_node_ids for _id in ids]
            for node_id in node_ids:
                chunk_indices_count[node_id] += 1

        sorted_chunk_indices = sorted(
            chunk_indices_count.keys(),
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
                f"> Querying with idx: {chunk_idx}: "
                f"{truncate_text(node.get_content(), 80)}"
            )
        # if no relationship is found, return the nodes found by keywords
        if not rel_texts:
            logger.info("> No relationships found, returning nodes found by keywords.")
            if len(sorted_nodes_with_scores) == 0:
                logger.info("> No nodes found by keywords, returning empty response.")
                return [
                    NodeWithScore(
                        node=TextNode(text="No relationships found."), score=1.0
                    )
                ]
            # In else case the sorted_nodes_with_scores is not empty
            # thus returning the nodes found by keywords
            return sorted_nodes_with_scores

        # add relationships as Node
        # TODO: make initial text customizable
        rel_initial_text = (
            f"The following are knowledge sequence in max depth"
            f" {self.graph_store_query_depth} "
            f"in the form of directed graph like:\n"
            f"`subject -[predicate]->, object, <-[predicate_next_hop]-,"
            f" object_next_hop ...`"
        )
        rel_info = [rel_initial_text, *rel_texts]
        rel_node_info = {
            "kg_rel_texts": rel_texts,
            "kg_rel_map": cur_rel_map,
        }
        if self._graph_schema != "":
            rel_node_info["kg_schema"] = {"schema": self._graph_schema}
        rel_info_text = "\n".join(
            [
                str(item)
                for sublist in rel_info
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]
        )
        if self._verbose:
            print_text(f"KG context:\n{rel_info_text}\n", color="blue")
        rel_text_node = TextNode(
            text=rel_info_text,
            metadata=rel_node_info,
            excluded_embed_metadata_keys=["kg_rel_map", "kg_rel_texts"],
            excluded_llm_metadata_keys=["kg_rel_map", "kg_rel_texts"],
        )
        # this node is constructed from rel_texts, give high confidence to avoid cutoff
        sorted_nodes_with_scores.append(
            NodeWithScore(node=rel_text_node, score=DEFAULT_NODE_SCORE)
        )

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


DEFAULT_SYNONYM_EXPAND_TEMPLATE = """
Generate synonyms or possible form of keywords up to {max_keywords} in total,
considering possible cases of capitalization, pluralization, common expressions, etc.
Provide all synonyms of keywords in comma-separated format: 'SYNONYMS: <keywords>'
Note, result should be in one-line with only one 'SYNONYMS: ' prefix
----
KEYWORDS: {question}
----
"""

DEFAULT_SYNONYM_EXPAND_PROMPT = PromptTemplate(
    DEFAULT_SYNONYM_EXPAND_TEMPLATE,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)


class KnowledgeGraphRAGRetriever(BaseRetriever):
    """
    Knowledge Graph RAG retriever.

    Retriever that perform SubGraph RAG towards knowledge graph.

    Args:
        service_context (Optional[ServiceContext]): A service context to use.
        storage_context (Optional[StorageContext]): A storage context to use.
        entity_extract_fn (Optional[Callable]): A function to extract entities.
        entity_extract_template Optional[BasePromptTemplate]): A Query Key Entity
            Extraction Prompt (see :ref:`Prompt-Templates`).
        entity_extract_policy (Optional[str]): The entity extraction policy to use.
            default: "union"
            possible values: "union", "intersection"
        synonym_expand_fn (Optional[Callable]): A function to expand synonyms.
        synonym_expand_template (Optional[QueryKeywordExpandPrompt]): A Query Key Entity
            Expansion Prompt (see :ref:`Prompt-Templates`).
        synonym_expand_policy (Optional[str]): The synonym expansion policy to use.
            default: "union"
            possible values: "union", "intersection"
        max_entities (int): The maximum number of entities to extract.
            default: 5
        max_synonyms (int): The maximum number of synonyms to expand per entity.
            default: 5
        retriever_mode (Optional[str]): The retriever mode to use.
            default: "keyword"
            possible values: "keyword", "embedding", "keyword_embedding"
        with_nl2graphquery (bool): Whether to combine NL2GraphQuery in context.
            default: False
        graph_traversal_depth (int): The depth of graph traversal.
            default: 2
        max_knowledge_sequence (int): The maximum number of knowledge sequence to
            include in the response. By default, it's 30.
        verbose (bool): Whether to print out debug info.
    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        entity_extract_fn: Optional[Callable] = None,
        entity_extract_template: Optional[BasePromptTemplate] = None,
        entity_extract_policy: Optional[str] = "union",
        synonym_expand_fn: Optional[Callable] = None,
        synonym_expand_template: Optional[BasePromptTemplate] = None,
        synonym_expand_policy: Optional[str] = "union",
        max_entities: int = 5,
        max_synonyms: int = 5,
        retriever_mode: Optional[str] = "keyword",
        with_nl2graphquery: bool = False,
        graph_traversal_depth: int = 2,
        max_knowledge_sequence: int = REL_TEXT_LIMIT,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the retriever."""
        # Ensure that we have a graph store
        assert storage_context is not None, "Must provide a storage context."
        assert (
            storage_context.graph_store is not None
        ), "Must provide a graph store in the storage context."
        self._storage_context = storage_context
        self._graph_store = storage_context.graph_store

        self._service_context = service_context or ServiceContext.from_defaults()

        self._entity_extract_fn = entity_extract_fn
        self._entity_extract_template = (
            entity_extract_template or DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE
        )
        self._entity_extract_policy = entity_extract_policy

        self._synonym_expand_fn = synonym_expand_fn
        self._synonym_expand_template = (
            synonym_expand_template or DEFAULT_SYNONYM_EXPAND_PROMPT
        )
        self._synonym_expand_policy = synonym_expand_policy

        self._max_entities = max_entities
        self._max_synonyms = max_synonyms
        self._retriever_mode = retriever_mode
        self._with_nl2graphquery = with_nl2graphquery
        if self._with_nl2graphquery:
            from llama_index.query_engine.knowledge_graph_query_engine import (
                KnowledgeGraphQueryEngine,
            )

            graph_query_synthesis_prompt = kwargs.get(
                "graph_query_synthesis_prompt",
                None,
            )
            if graph_query_synthesis_prompt is not None:
                del kwargs["graph_query_synthesis_prompt"]

            graph_response_answer_prompt = kwargs.get(
                "graph_response_answer_prompt",
                None,
            )
            if graph_response_answer_prompt is not None:
                del kwargs["graph_response_answer_prompt"]

            refresh_schema = kwargs.get("refresh_schema", False)
            response_synthesizer = kwargs.get("response_synthesizer", None)
            self._kg_query_engine = KnowledgeGraphQueryEngine(
                service_context=self._service_context,
                storage_context=self._storage_context,
                graph_query_synthesis_prompt=graph_query_synthesis_prompt,
                graph_response_answer_prompt=graph_response_answer_prompt,
                refresh_schema=refresh_schema,
                verbose=verbose,
                response_synthesizer=response_synthesizer,
                **kwargs,
            )

        self._graph_traversal_depth = graph_traversal_depth
        self._max_knowledge_sequence = max_knowledge_sequence
        self._verbose = verbose
        refresh_schema = kwargs.get("refresh_schema", False)
        try:
            self._graph_schema = self._graph_store.get_schema(refresh=refresh_schema)
        except NotImplementedError:
            self._graph_schema = ""
        except Exception as e:
            logger.warning(f"Failed to get graph schema: {e}")
            self._graph_schema = ""
        super().__init__(callback_manager)

    def _process_entities(
        self,
        query_str: str,
        handle_fn: Optional[Callable],
        handle_llm_prompt_template: Optional[BasePromptTemplate],
        cross_handle_policy: Optional[str] = "union",
        max_items: Optional[int] = 5,
        result_start_token: str = "KEYWORDS:",
    ) -> List[str]:
        """Get entities from query string."""
        assert cross_handle_policy in [
            "union",
            "intersection",
        ], "Invalid entity extraction policy."
        if cross_handle_policy == "intersection":
            assert all(
                [
                    handle_fn is not None,
                    handle_llm_prompt_template is not None,
                ]
            ), "Must provide entity extract function and template."
        assert any(
            [
                handle_fn is not None,
                handle_llm_prompt_template is not None,
            ]
        ), "Must provide either entity extract function or template."
        enitities_fn: List[str] = []
        enitities_llm: Set[str] = set()

        if handle_fn is not None:
            enitities_fn = handle_fn(query_str)
        if handle_llm_prompt_template is not None:
            response = self._service_context.llm.predict(
                handle_llm_prompt_template,
                max_keywords=max_items,
                question=query_str,
            )
            enitities_llm = extract_keywords_given_response(
                response, start_token=result_start_token, lowercase=False
            )
        if cross_handle_policy == "union":
            entities = list(set(enitities_fn) | enitities_llm)
        elif cross_handle_policy == "intersection":
            entities = list(set(enitities_fn).intersection(set(enitities_llm)))
        if self._verbose:
            print_text(f"Entities processed: {entities}\n", color="green")

        return entities

    async def _aprocess_entities(
        self,
        query_str: str,
        handle_fn: Optional[Callable],
        handle_llm_prompt_template: Optional[BasePromptTemplate],
        cross_handle_policy: Optional[str] = "union",
        max_items: Optional[int] = 5,
        result_start_token: str = "KEYWORDS:",
    ) -> List[str]:
        """Get entities from query string."""
        assert cross_handle_policy in [
            "union",
            "intersection",
        ], "Invalid entity extraction policy."
        if cross_handle_policy == "intersection":
            assert all(
                [
                    handle_fn is not None,
                    handle_llm_prompt_template is not None,
                ]
            ), "Must provide entity extract function and template."
        assert any(
            [
                handle_fn is not None,
                handle_llm_prompt_template is not None,
            ]
        ), "Must provide either entity extract function or template."
        enitities_fn: List[str] = []
        enitities_llm: Set[str] = set()

        if handle_fn is not None:
            enitities_fn = handle_fn(query_str)
        if handle_llm_prompt_template is not None:
            response = await self._service_context.llm.apredict(
                handle_llm_prompt_template,
                max_keywords=max_items,
                question=query_str,
            )
            enitities_llm = extract_keywords_given_response(
                response, start_token=result_start_token, lowercase=False
            )
        if cross_handle_policy == "union":
            entities = list(set(enitities_fn) | enitities_llm)
        elif cross_handle_policy == "intersection":
            entities = list(set(enitities_fn).intersection(set(enitities_llm)))
        if self._verbose:
            print_text(f"Entities processed: {entities}\n", color="green")

        return entities

    def _get_entities(self, query_str: str) -> List[str]:
        """Get entities from query string."""
        entities = self._process_entities(
            query_str,
            self._entity_extract_fn,
            self._entity_extract_template,
            self._entity_extract_policy,
            self._max_entities,
            "KEYWORDS:",
        )
        expanded_entities = self._expand_synonyms(entities)
        return list(set(entities) | set(expanded_entities))

    async def _aget_entities(self, query_str: str) -> List[str]:
        """Get entities from query string."""
        entities = await self._aprocess_entities(
            query_str,
            self._entity_extract_fn,
            self._entity_extract_template,
            self._entity_extract_policy,
            self._max_entities,
            "KEYWORDS:",
        )
        expanded_entities = await self._aexpand_synonyms(entities)
        return list(set(entities) | set(expanded_entities))

    def _expand_synonyms(self, keywords: List[str]) -> List[str]:
        """Expand synonyms or similar expressions for keywords."""
        return self._process_entities(
            str(keywords),
            self._synonym_expand_fn,
            self._synonym_expand_template,
            self._synonym_expand_policy,
            self._max_synonyms,
            "SYNONYMS:",
        )

    async def _aexpand_synonyms(self, keywords: List[str]) -> List[str]:
        """Expand synonyms or similar expressions for keywords."""
        return await self._aprocess_entities(
            str(keywords),
            self._synonym_expand_fn,
            self._synonym_expand_template,
            self._synonym_expand_policy,
            self._max_synonyms,
            "SYNONYMS:",
        )

    def _get_knowledge_sequence(
        self, entities: List[str]
    ) -> Tuple[List[str], Optional[Dict[Any, Any]]]:
        """Get knowledge sequence from entities."""
        # Get SubGraph from Graph Store as Knowledge Sequence
        rel_map: Optional[Dict] = self._graph_store.get_rel_map(
            entities, self._graph_traversal_depth, limit=self._max_knowledge_sequence
        )
        logger.debug(f"rel_map: {rel_map}")

        # Build Knowledge Sequence
        knowledge_sequence = []
        if rel_map:
            knowledge_sequence.extend(
                [str(rel_obj) for rel_objs in rel_map.values() for rel_obj in rel_objs]
            )
        else:
            logger.info("> No knowledge sequence extracted from entities.")
            return [], None

        return knowledge_sequence, rel_map

    async def _aget_knowledge_sequence(
        self, entities: List[str]
    ) -> Tuple[List[str], Optional[Dict[Any, Any]]]:
        """Get knowledge sequence from entities."""
        # Get SubGraph from Graph Store as Knowledge Sequence
        # TBD: async in graph store
        rel_map: Optional[Dict] = self._graph_store.get_rel_map(
            entities, self._graph_traversal_depth, limit=self._max_knowledge_sequence
        )
        logger.debug(f"rel_map from GraphStore:\n{rel_map}")

        # Build Knowledge Sequence
        knowledge_sequence = []
        if rel_map:
            knowledge_sequence.extend(
                [str(rel_obj) for rel_objs in rel_map.values() for rel_obj in rel_objs]
            )
        else:
            logger.info("> No knowledge sequence extracted from entities.")
            return [], None

        return knowledge_sequence, rel_map

    def _build_nodes(
        self, knowledge_sequence: List[str], rel_map: Optional[Dict[Any, Any]] = None
    ) -> List[NodeWithScore]:
        """Build nodes from knowledge sequence."""
        if len(knowledge_sequence) == 0:
            logger.info("> No knowledge sequence extracted from entities.")
            return []
        _new_line_char = "\n"
        context_string = (
            f"The following are knowledge sequence in max depth"
            f" {self._graph_traversal_depth} "
            f"in the form of directed graph like:\n"
            f"`subject -[predicate]->, object, <-[predicate_next_hop]-,"
            f" object_next_hop ...`"
            f" extracted based on key entities as subject:\n"
            f"{_new_line_char.join(knowledge_sequence)}"
        )
        if self._verbose:
            print_text(f"Graph RAG context:\n{context_string}\n", color="blue")

        rel_node_info = {
            "kg_rel_map": rel_map,
            "kg_rel_text": knowledge_sequence,
        }
        metadata_keys = ["kg_rel_map", "kg_rel_text"]
        if self._graph_schema != "":
            rel_node_info["kg_schema"] = {"schema": self._graph_schema}
            metadata_keys.append("kg_schema")
        node = NodeWithScore(
            node=TextNode(
                text=context_string,
                score=1.0,
                metadata=rel_node_info,
                excluded_embed_metadata_keys=metadata_keys,
                excluded_llm_metadata_keys=metadata_keys,
            )
        )
        return [node]

    def _retrieve_keyword(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve in keyword mode."""
        if self._retriever_mode not in ["keyword", "keyword_embedding"]:
            return []
        # Get entities
        entities = self._get_entities(query_bundle.query_str)
        # Before we enable embedding/semantic search, we need to make sure
        # we don't miss any entities that's synoynm of the entities we extracted
        # in string matching based retrieval in following steps, thus we expand
        # synonyms here.
        if len(entities) == 0:
            logger.info("> No entities extracted from query string.")
            return []

        # Get SubGraph from Graph Store as Knowledge Sequence
        knowledge_sequence, rel_map = self._get_knowledge_sequence(entities)

        return self._build_nodes(knowledge_sequence, rel_map)

    async def _aretrieve_keyword(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """Retrieve in keyword mode."""
        if self._retriever_mode not in ["keyword", "keyword_embedding"]:
            return []
        # Get entities
        entities = await self._aget_entities(query_bundle.query_str)
        # Before we enable embedding/semantic search, we need to make sure
        # we don't miss any entities that's synoynm of the entities we extracted
        # in string matching based retrieval in following steps, thus we expand
        # synonyms here.
        if len(entities) == 0:
            logger.info("> No entities extracted from query string.")
            return []

        # Get SubGraph from Graph Store as Knowledge Sequence
        knowledge_sequence, rel_map = await self._aget_knowledge_sequence(entities)

        return self._build_nodes(knowledge_sequence, rel_map)

    def _retrieve_embedding(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve in embedding mode."""
        if self._retriever_mode not in ["embedding", "keyword_embedding"]:
            return []
        # TBD: will implement this later with vector store.
        raise NotImplementedError

    async def _aretrieve_embedding(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """Retrieve in embedding mode."""
        if self._retriever_mode not in ["embedding", "keyword_embedding"]:
            return []
        # TBD: will implement this later with vector store.
        raise NotImplementedError

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Build nodes for response."""
        nodes: List[NodeWithScore] = []
        if self._with_nl2graphquery:
            try:
                nodes_nl2graphquery = self._kg_query_engine._retrieve(query_bundle)
                nodes.extend(nodes_nl2graphquery)
            except Exception as e:
                logger.warning(f"Error in retrieving from nl2graphquery: {e}")

        nodes.extend(self._retrieve_keyword(query_bundle))
        nodes.extend(self._retrieve_embedding(query_bundle))

        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Build nodes for response."""
        nodes: List[NodeWithScore] = []
        if self._with_nl2graphquery:
            try:
                nodes_nl2graphquery = await self._kg_query_engine._aretrieve(
                    query_bundle
                )
                nodes.extend(nodes_nl2graphquery)
            except Exception as e:
                logger.warning(f"Error in retrieving from nl2graphquery: {e}")

        nodes.extend(await self._aretrieve_keyword(query_bundle))
        nodes.extend(await self._aretrieve_embedding(query_bundle))

        return nodes
