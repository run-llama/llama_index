from typing import Any, Dict, List, Optional

import asyncio
from enum import Enum

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.base.response.schema import Response
from llama_index.core.base.base_retriever import BaseRetriever, QueryType
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import run_transformations
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import (
    BaseNode,
    NodeWithScore,
    QueryBundle,
    TextNode,
    TransformComponent,
)
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    BasePydanticVectorStore,
)
from llama_index.packs.raptor.clustering import get_clusters


DEFAULT_SUMMARY_PROMPT = (
    "Summarize the provided text, including as many key details as needed."
)


class QueryModes(str, Enum):
    """Query modes."""

    tree_traversal = "tree_traversal"
    collapsed = "collapsed"


class SummaryModule(BaseModel):
    response_synthesizer: BaseSynthesizer = Field(description="LLM")
    summary_prompt: str = Field(
        default=DEFAULT_SUMMARY_PROMPT,
        description="Summary prompt.",
    )
    num_workers: int = Field(
        default=4, description="Number of workers to generate summaries."
    )
    show_progress: bool = Field(default=True, description="Show progress.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        llm: Optional[LLM] = None,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        num_workers: int = 4,
    ) -> None:
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", use_async=True, llm=llm
        )
        super().__init__(
            response_synthesizer=response_synthesizer,
            summary_prompt=summary_prompt,
            num_workers=num_workers,
        )

    async def generate_summaries(
        self, documents_per_cluster: List[List[BaseNode]]
    ) -> List[str]:
        """Generate summaries of documents per cluster.

        Args:
            documents_per_cluster (List[List[BaseNode]]): List of documents per cluster

        Returns:
            List[str]: List of summary for each cluster
        """
        jobs = []
        for documents in documents_per_cluster:
            with_scores = [NodeWithScore(node=doc, score=1.0) for doc in documents]
            jobs.append(
                self.response_synthesizer.asynthesize(self.summary_prompt, with_scores)
            )

        lock = asyncio.Semaphore(self.num_workers)
        responses = []

        # run the jobs while limiting the number of concurrent jobs to num_workers
        for job in jobs:
            async with lock:
                responses.append(await job)

        return [str(response) for response in responses]


class RaptorRetriever(BaseRetriever):
    """Raptor indexing retriever."""

    def __init__(
        self,
        documents: List[BaseNode],
        tree_depth: int = 3,
        similarity_top_k: int = 2,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        transformations: Optional[List[TransformComponent]] = None,
        summary_module: Optional[SummaryModule] = None,
        existing_index: Optional[VectorStoreIndex] = None,
        mode: QueryModes = "collapsed",
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(
            **kwargs,
        )

        self.mode = mode
        self.summary_module = summary_module or SummaryModule(llm=llm)
        self.index = existing_index or VectorStoreIndex(
            nodes=[],
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=embed_model,
            transformations=transformations,
        )
        self.tree_depth = tree_depth
        self.similarity_top_k = similarity_top_k

        if len(documents) > 0:
            asyncio.run(self.insert(documents))

    def _get_embeddings_per_level(self, level: int = 0) -> List[float]:
        """Retrieve embeddings per level in the abstraction tree.

        Args:
            level (int, optional): Target level. Defaults to 0 which stands for leaf nodes.

        Returns:
            List[float]: List of embeddings
        """
        filters = MetadataFilters(filters=[MetadataFilter("level", level)])

        # kind of janky, but should work with any vector index
        source_nodes = self.index.as_retriever(
            similarity_top_k=10000, filters=filters
        ).retrieve("retrieve")

        return [x.node for x in source_nodes]

    async def insert(self, documents: List[BaseNode]) -> None:
        """Given a set of documents, this function inserts higher level of abstractions within the index.

        For later retrieval

        Args:
            documents (List[BaseNode]): List of Documents
        """
        embed_model = self.index._embed_model
        transformations = self.index._transformations

        cur_nodes = run_transformations(documents, transformations, in_place=False)
        for level in range(self.tree_depth):
            # get the embeddings for the current documents

            if self._verbose:
                print(f"Generating embeddings for level {level}.")

            embeddings = await embed_model.aget_text_embedding_batch(
                [node.get_content(metadata_mode="embed") for node in cur_nodes]
            )
            assert len(embeddings) == len(cur_nodes)
            id_to_embedding = {
                node.id_: embedding for node, embedding in zip(cur_nodes, embeddings)
            }

            if self._verbose:
                print(f"Performing clustering for level {level}.")

            # cluster the documents
            nodes_per_cluster = get_clusters(cur_nodes, id_to_embedding)

            if self._verbose:
                print(
                    f"Generating summaries for level {level} with {len(nodes_per_cluster)} clusters."
                )
            summaries_per_cluster = await self.summary_module.generate_summaries(
                nodes_per_cluster
            )

            if self._verbose:
                print(
                    f"Level {level} created summaries/clusters: {len(nodes_per_cluster)}"
                )

            # replace the current nodes with their summaries
            new_nodes = [
                TextNode(
                    text=summary,
                    metadata={"level": level},
                    excluded_embed_metadata_keys=["level"],
                    excluded_llm_metadata_keys=["level"],
                )
                for summary in summaries_per_cluster
            ]

            # insert the nodes with their embeddings and parent_id
            nodes_with_embeddings = []
            for cluster, summary_doc in zip(nodes_per_cluster, new_nodes):
                for node in cluster:
                    node.metadata["parent_id"] = summary_doc.id_
                    node.excluded_embed_metadata_keys.append("parent_id")
                    node.excluded_llm_metadata_keys.append("parent_id")
                    node.embedding = id_to_embedding[node.id_]
                    nodes_with_embeddings.append(node)

            self.index.insert_nodes(nodes_with_embeddings)

            # set the current nodes to the new nodes
            cur_nodes = new_nodes

        self.index.insert_nodes(cur_nodes)

    async def collapsed_retrieval(self, query_str: str) -> Response:
        """Query the index as a collapsed tree -- i.e. a single pool of nodes."""
        return await self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        ).aretrieve(query_str)

    async def tree_traversal_retrieval(self, query_str: str) -> Response:
        """Query the index as a tree, traversing the tree from the top down."""
        # get top k nodes for each level, starting with the top
        parent_ids = None
        selected_node_ids = set()
        selected_nodes = []
        level = self.tree_depth - 1
        while level >= 0:
            # retrieve nodes at the current level
            if parent_ids is None:
                nodes = await self.index.as_retriever(
                    similarity_top_k=self.similarity_top_k,
                    filters=MetadataFilters(
                        filters=[MetadataFilter(key="level", value=level)]
                    ),
                ).aretrieve(query_str)

                for node in nodes:
                    if node.id_ not in selected_node_ids:
                        selected_nodes.append(node)
                        selected_node_ids.add(node.id_)

                parent_ids = [node.id_ for node in nodes]
                if self._verbose:
                    print(f"Retrieved parent IDs from level {level}: {parent_ids!s}")
            # retrieve nodes that are children of the nodes at the previous level
            elif parent_ids is not None and len(parent_ids) > 0:
                nested_nodes = await asyncio.gather(
                    *[
                        self.index.as_retriever(
                            similarity_top_k=self.similarity_top_k,
                            filters=MetadataFilters(
                                filters=[MetadataFilter(key="parent_id", value=id_)]
                            ),
                        ).aretrieve(query_str)
                        for id_ in parent_ids
                    ]
                )

                nodes = [node for nested in nested_nodes for node in nested]
                for node in nodes:
                    if node.id_ not in selected_node_ids:
                        selected_nodes.append(node)
                        selected_node_ids.add(node.id_)

                if self._verbose:
                    print(f"Retrieved {len(nodes)} from parents at level {level}.")

                level -= 1
                parent_ids = None

        return selected_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query and mode."""
        # not used, needed for type checking

    def retrieve(
        self, query_str_or_bundle: QueryType, mode: Optional[QueryModes] = None
    ) -> List[NodeWithScore]:
        """Retrieve nodes given query and mode."""
        if isinstance(query_str_or_bundle, QueryBundle):
            query_str = query_str_or_bundle.query_str
        else:
            query_str = query_str_or_bundle

        return asyncio.run(self.aretrieve(query_str, mode or self.mode))

    async def aretrieve(
        self, query_str_or_bundle: QueryType, mode: Optional[QueryModes] = None
    ) -> List[NodeWithScore]:
        """Retrieve nodes given query and mode."""
        if isinstance(query_str_or_bundle, QueryBundle):
            query_str = query_str_or_bundle.query_str
        else:
            query_str = query_str_or_bundle

        mode = mode or self.mode
        if mode == "tree_traversal":
            return await self.tree_traversal_retrieval(query_str)
        elif mode == "collapsed":
            return await self.collapsed_retrieval(query_str)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def persist(self, persist_dir: str) -> None:
        self.index.storage_context.persist(persist_dir=persist_dir)

    @classmethod
    def from_persist_dir(
        cls: "RaptorRetriever",
        persist_dir: str,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> "RaptorRetriever":
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return cls(
            [],
            existing_index=load_index_from_storage(
                storage_context, embed_model=embed_model
            ),
            **kwargs,
        )


class RaptorPack(BaseLlamaPack):
    """Raptor pack."""

    def __init__(
        self,
        documents: List[BaseNode],
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        similarity_top_k: int = 2,
        mode: QueryModes = "collapsed",
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.retriever = RaptorRetriever(
            documents,
            embed_model=embed_model,
            llm=llm,
            similarity_top_k=similarity_top_k,
            vector_store=vector_store,
            mode=mode,
            verbose=verbose,
            **kwargs,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "retriever": self.retriever,
        }

    def run(
        self,
        query: str,
        mode: Optional[QueryModes] = None,
    ) -> Any:
        """Run the pipeline."""
        return self.retriever.retrieve(query, mode=mode)
