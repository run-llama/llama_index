""" Knowledge Graph Query Engine"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from llama_index.bridge.langchain import print_text
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.graph_stores.registery import GRAPH_STORE_CLASS_TO_GRAPH_STORE_TYPE
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.base import Prompt, PromptType
from llama_index.prompts.default_prompts import DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE
from llama_index.prompts.prompts import QueryKeywordExtractPrompt
from llama_index.query_engine import KnowledgeGraphQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.response_synthesizers import BaseSynthesizer, get_response_synthesizer
from llama_index.schema import NodeWithScore, TextNode
from llama_index.storage.storage_context import StorageContext

logger = logging.getLogger(__name__)

DEFAULT_SYNONYM_EXPAND_TEMPLATE = """
Generate synonyms or possible form of keywords,
considering possible cases of capitalization, pluralization, common expressions, etc.
Provide synonyms of keywords in comma-separated format: 'SYNONYMS: <keywords>'
----
KEYWORDS: {keywords}
----
"""
# Example of the prompt:
# ---------------------
# Generate synonyms or possible form of keywords,
# considering possible cases of capitalization, pluralization, common expressions, etc.
# Provide synonyms of keywords in comma-separated format: 'SYNONYMS: <keywords>'
# ----
# KEYWORDS: apple inc, book
# ----
# SYNONYMS: Apple Inc, Apple Incorporated, apple company, Book, books

DEFAULT_SYNONYM_EXPAND_PROMPT = Prompt(
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
        entity_extract_template Optional[QueryKeywordExtractPrompt]): A Query Key Entity
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
        retriever_mode (Optional[str]): The retriever mode to use.
            default: "keyword"
            possible values: "keyword", "embedding", "keyword_embedding"
        with_nl2graphquery (bool): Whether to combine NL2GraphQuery in context.
            default: False
        similarity_top_k (int): The number of top embeddings to use
            (if embeddings are used).
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
        entity_extract_template: Optional[QueryKeywordExtractPrompt] = None,
        entity_extract_policy: Optional[str] = "union",
        synonym_expand_fn: Optional[Callable] = None,
        synonym_expand_template: Optional[Prompt] = None,
        synonym_expand_policy: Optional[str] = "union",
        max_entities: int = 5,
        retriever_mode: Optional[str] = "keyword",
        with_nl2graphquery: bool = False,
        similarity_top_k: int = 5,
        graph_traversal_depth: int = 2,
        max_knowledge_sequence: int = 30,
        verbose: bool = False,
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

        # Get Graph Store Type
        self._graph_store_type = GRAPH_STORE_CLASS_TO_GRAPH_STORE_TYPE[
            self._graph_store.__class__
        ]

        # Get Graph schema
        self._graph_schema = self._graph_store.get_schema()

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
        self._retriever_mode = retriever_mode
        self._with_nl2graphquery = with_nl2graphquery
        if self._with_nl2graphquery:
            graph_query_synthesis_prompt = kwargs.get(
                "graph_query_synthesis_prompt",
                None,
            )
            graph_response_answer_prompt = kwargs.get(
                "graph_response_answer_prompt",
                None,
            )
            refresh_schema = kwargs.get("refresh_schema", False)
            verbose = kwargs.get("verbose", False)
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

        self._similarity_top_k = similarity_top_k
        self._graph_traversal_depth = graph_traversal_depth
        self._max_knowledge_sequence = max_knowledge_sequence
        self._verbose = verbose

    def _get_entities(self, query_str: str) -> List[str]:
        """Get entities from query string."""
        assert self._entity_extract_policy in [
            "union",
            "intersection",
        ], "Invalid entity extraction policy."
        if self._entity_extract_policy == "intersection":
            assert all(
                [
                    self._entity_extract_fn is not None,
                    self._entity_extract_template is not None,
                ]
            ), "Must provide entity extract function and template."
        assert any(
            [
                self._entity_extract_fn is not None,
                self._entity_extract_template is not None,
            ]
        ), "Must provide either entity extract function or template."
        keywords_fn: List[str] = []
        keywords_llm: Set[str] = set()

        if self._entity_extract_fn is not None:
            keywords_fn = self._entity_extract_fn(query_str)
        if self._entity_extract_template is not None:
            response = self._service_context.llm_predictor.predict(
                self._entity_extract_template,
                max_keywords=self._max_entities,
                question=query_str,
            )
            keywords_llm = extract_keywords_given_response(
                response, start_token="KEYWORDS:", lowercase=False
            )
        if self._entity_extract_policy == "union":
            keywords = list(set(keywords_fn) | keywords_llm)
        elif self._entity_extract_policy == "intersection":
            keywords = list(set(keywords_fn).intersection(set(keywords_llm)))
        if self._verbose:
            print_text(f"Entities extracted: {keywords}\n", color="green")

        return keywords

    def _expand_synonyms(self, keywords: List[str]) -> List[str]:
        """Expand synonyms."""
        # if no _synonym_expand_fn nor _synonym_expand_template is provided,
        # we don't expand synonyms
        if not any(
            [
                self._synonym_expand_fn is not None,
                self._synonym_expand_template is not None,
            ]
        ):
            return []

        assert self._synonym_expand_policy in [
            "union",
            "intersection",
        ], "Invalid synonym expansion policy."
        if self._synonym_expand_policy == "intersection":
            assert all(
                [
                    self._synonym_expand_fn is not None,
                    self._synonym_expand_template is not None,
                ]
            ), "Must provide synonym expand function and template."
        keywords_fn: List[str] = []
        keywords_llm: Set[str] = set()

        if self._synonym_expand_fn is not None:
            keywords_fn = self._synonym_expand_fn(keywords)
        if self._synonym_expand_template is not None:
            response = self._service_context.llm_predictor.predict(
                self._synonym_expand_template,
                keywords=str(keywords),
            )
            keywords_llm = extract_keywords_given_response(
                response, start_token="KEYWORDS:", lowercase=False
            )
        if self._synonym_expand_policy == "union":
            keywords = list(set(keywords_fn) | keywords_llm)
        elif self._synonym_expand_policy == "intersection":
            keywords = list(set(keywords_fn).intersection(set(keywords_llm)))
        if self._verbose:
            print_text(f"Entities expanded: {keywords}\n", color="green")
        return keywords

    def _get_knowledge_sequence(
        self, entities: List[str]
    ) -> Tuple[List[str], Optional[Dict[Any, Any]]]:
        """Get knowledge sequence from entities."""
        # Get SubGraph from Graph Store as Knowledge Sequence
        rel_map: Optional[Dict] = self._graph_store.get_rel_map(
            entities, self._graph_traversal_depth
        )
        logger.debug(f"rel_map: {rel_map}")
        if self._verbose:
            print_text(f"rel_map: {rel_map}\n", color="green")
        # Build Knowledge Sequence
        knowledge_sequence = []
        if rel_map:
            knowledge_sequence.extend(
                [
                    f"{sub} {rel_obj}"
                    for sub, rel_objs in rel_map.items()
                    for rel_obj in rel_objs
                ]
            )
        else:
            logger.info("> No knowledge sequence extracted from entities.")
            return [], None
        # truncate knowledge sequence
        if len(knowledge_sequence) > self._max_knowledge_sequence:
            knowledge_sequence = knowledge_sequence[: self._max_knowledge_sequence]
        if self._verbose:
            print_text(f"knowledge_sequence: {knowledge_sequence}\n", color="green")
        return knowledge_sequence, rel_map

    def _retrieve_keyword(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """retrieve in keyword mode."""
        # Get entities
        entities = self._get_entities(query_bundle.query_str)
        # Before we enable embedding/symantic search, we need to make sure
        # we don't miss any entities that's synoynm of the entities we extracted
        # in string matching based retrieval in following steps, thus we expand
        # synonyms here.
        if len(entities) == 0:
            logger.info("> No entities extracted from query string.")
            return []
        expanded_entities = self._expand_synonyms(entities)
        if expanded_entities:
            entities = list(set(entities + expanded_entities))

        # Get SubGraph from Graph Store as Knowledge Sequence
        knowledge_sequence, rel_map = self._get_knowledge_sequence(entities)
        if len(knowledge_sequence) == 0:
            logger.info("> No knowledge sequence extracted from entities.")
            return []

        context_string = (
            f"The following are knowledge sequence in max depth"
            f" {self._graph_traversal_depth} "
            f"in the form of "
            f"`subject [predicate, object, predicate_next_hop, object_next_hop ...]`"
            f" extracted from the query string:\n"
            f"'\n'.join(knowledge_sequence)"
        )

        node = NodeWithScore(
            node=TextNode(
                text=context_string,
                score=1.0,
                metadata={
                    "kg_rel_text": knowledge_sequence,
                    "kg_rel_map": rel_map,
                },
            )
        )
        return [node]

    def _retrieve_embedding(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """retrieve in embedding mode."""
        # TBD: will implement this later with vector store.
        raise NotImplementedError

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Build nodes for response."""
        nodes, nodes_keyword, nodes_embedding = [], [], []
        if self._with_nl2graphquery:
            nodes_nl2graphquery = self._kg_query_engine._retrieve(query_bundle)
            nodes.extend(nodes_nl2graphquery)

        if self._retriever_mode == "keyword":
            nodes_keyword = self._retrieve_keyword(query_bundle)
        elif self._retriever_mode == "embedding":
            nodes_embedding = self._retrieve_embedding(query_bundle)
        elif self._retriever_mode == "keyword_embedding":
            nodes_keyword = self._retrieve_keyword(query_bundle)
            nodes_embedding = self._retrieve_embedding(query_bundle)
        else:
            raise ValueError("Invalid retriever mode.")

        nodes.extend(nodes_keyword)
        nodes.extend(nodes_embedding)
        return nodes


class KnowledgeGraphRAGQueryEngine(BaseQueryEngine):
    """Knowledge Graph RAG query engine.

    Query engine that perform SubGraph RAG towards knowledge graph.

    Args:
        service_context (Optional[ServiceContext]): A service context to use.
        storage_context (Optional[StorageContext]): A storage context to use.
        entity_extract_fn (Optional[Callable]): A function to extract entities.
        entity_extract_template Optional[QueryKeywordExtractPrompt]): A Query Key Entity
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
        retriever_mode (Optional[str]): The retriever mode to use.
            default: "keyword"
            possible values: "keyword", "embedding", "keyword_embedding"
        with_nl2graphquery (bool): Whether to combine NL2GraphQuery in context.
            default: False
        similarity_top_k (int): The number of top embeddings to use
            (if embeddings are used).
        graph_traversal_depth (int): The depth of graph traversal.
            default: 2
        max_knowledge_sequence (int): The maximum number of knowledge sequence to
            include in the response. By default, it's 30.
        verbose (bool): Whether to print out debug info.
        response_synthesizer (Optional[BaseSynthesizer]): A response synthesizer to use.

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        entity_extract_fn: Optional[Callable] = None,
        entity_extract_template: Optional[Prompt] = None,
        entity_extract_policy: Optional[str] = "union",
        synonym_expand_fn: Optional[Callable] = None,
        synonym_expand_template: Optional[Prompt] = None,
        synonym_expand_policy: Optional[str] = "union",
        max_entities: int = 5,
        retriever_mode: Optional[str] = "keyword",
        with_nl2graphquery: bool = False,
        similarity_top_k: int = 5,
        graph_traversal_depth: int = 2,
        max_knowledge_sequence: int = 30,
        verbose: bool = False,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        **kwargs: Any,
    ):
        self._retriever = KnowledgeGraphRAGRetriever(
            service_context=service_context,
            storage_context=storage_context,
            entity_extract_fn=entity_extract_fn,
            entity_extract_template=entity_extract_template,
            entity_extract_policy=entity_extract_policy,
            synonym_expand_fn=synonym_expand_fn,
            synonym_expand_template=synonym_expand_template,
            synonym_expand_policy=synonym_expand_policy,
            max_entities=max_entities,
            retriever_mode=retriever_mode,
            with_nl2graphquery=with_nl2graphquery,
            similarity_top_k=similarity_top_k,
            graph_traversal_depth=graph_traversal_depth,
            max_knowledge_sequence=max_knowledge_sequence,
            **kwargs,
        )
        self._service_context = service_context or ServiceContext.from_defaults()
        self._storage_context = storage_context or StorageContext.from_defaults()
        self._graph_store = self._storage_context.graph_store

        self._verbose = verbose
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(
            callback_manager=self._service_context.callback_manager,
            service_context=self._service_context,
        )

        super().__init__(self._service_context.callback_manager)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query the graph store for RAG."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                # Get the graph store response
                nodes = self._retriever.retrieve(query_bundle)
                retrieve_event.on_end(payload={EventPayload.RESPONSE: nodes})

            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            if self._verbose:
                print_text(f"Final Response: {response}\n", color="green")

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query the graph store."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                # Get the graph store response
                # TBD: This is a blocking call. We need to make it async.
                nodes = self._retriever.retrieve(query_bundle)
                retrieve_event.on_end(payload={EventPayload.RESPONSE: nodes})

            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            if self._verbose:
                print_text(f"Final Response: {response}\n", color="green")

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Get the retriever."""
        return self._retriever
