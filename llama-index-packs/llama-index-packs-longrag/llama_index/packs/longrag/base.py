import typing as t

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.async_utils import asyncio_run
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.workflow import (
    Event,
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Context,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import (
    QueryBundle,
    TextNode,
    BaseNode,
    NodeWithScore,
)
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    BasePydanticVectorStore,
)
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
)
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms import LLM

DEFAULT_CHUNK_SIZE = 4096
DEFAULT_MAX_GROUP_SIZE = 20
DEFAULT_SMALL_CHUNK_SIZE = 512
DEFAULT_TOP_K = 8


def split_doc(chunk_size: int, documents: t.List[BaseNode]) -> t.List[TextNode]:
    """
    Splits documents into smaller pieces.

    Args:
        chunk_size (int): Chunk size
        documents (t.List[BaseNode]): Documents

    Returns:
        t.List[TextNode]: Smaller chunks

    """
    # split docs into tokens
    text_parser = SentenceSplitter(chunk_size=chunk_size)
    return text_parser.get_nodes_from_documents(documents)


def group_docs(
    nodes: t.List[str],
    adj: t.Dict[str, t.List[str]],
    max_group_size: t.Optional[int] = DEFAULT_MAX_GROUP_SIZE,
) -> t.Set[t.FrozenSet[str]]:
    """
    Groups documents.

    Args:
        nodes (List[str]): documents IDs
        adj (Dict[str, List[str]]): related documents for each document; id -> list of doc strings
        max_group_size (Optional[int], optional): max group size, None if no max group size. Defaults to DEFAULT_MAX_GROUP_SIZE.

    """
    docs = sorted(nodes, key=lambda node: len(adj[node]))
    groups = set()  # set of set of IDs
    for d in docs:
        related_groups = set()
        for r in adj[d]:
            for g in groups:
                if r in g:
                    related_groups = related_groups.union(frozenset([g]))

        gnew = {d}
        related_groupsl = sorted(related_groups, key=lambda el: len(el))
        for g in related_groupsl:
            if max_group_size is None or len(gnew) + len(g) <= max_group_size:
                gnew = gnew.union(g)
                if g in groups:
                    groups.remove(g)

        groups.add(frozenset(gnew))

    return groups


def get_grouped_docs(
    nodes: t.List[TextNode], max_group_size: t.Optional[int] = DEFAULT_MAX_GROUP_SIZE
) -> t.List[TextNode]:
    """
    Gets list of documents that are grouped.

    Args:
        nodes (t.List[TextNode]): Input list
        max_group_size (Optional[int], optional): max group size, None if no max group size. Defaults to DEFAULT_MAX_GROUP_SIZE.

    Returns:
        t.List[TextNode]: Output list

    """
    # node IDs
    nodes_str = [node.id_ for node in nodes]
    # maps node ID -> related node IDs based on that node's relationships
    adj: t.Dict[str, t.List[str]] = {
        node.id_: [val.node_id for val in node.relationships.values()] for node in nodes
    }
    # node ID -> node
    nodes_dict = {node.id_: node for node in nodes}

    res = group_docs(nodes_str, adj, max_group_size)

    ret_nodes = []
    for g in res:
        cur_node = TextNode()

        for node_id in g:
            cur_node.text += nodes_dict[node_id].text + "\n\n"
            cur_node.metadata.update(nodes_dict[node_id].metadata)

        ret_nodes.append(cur_node)

    return ret_nodes


class LongRAGRetriever(BaseRetriever):
    """Long RAG Retriever."""

    def __init__(
        self,
        grouped_nodes: t.List[TextNode],
        small_toks: t.List[TextNode],
        vector_store: BasePydanticVectorStore,
        similarity_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """
        Constructor.

        Args:
            grouped_nodes (t.List[TextNode]): Long retrieval units, nodes with docs grouped together based on relationships
            small_toks (t.List[TextNode]): Smaller tokens
            embed_model (BaseEmbedding, optional): Embed model. Defaults to None.
            similarity_top_k (int, optional): Similarity top k. Defaults to 8.

        """
        self._grouped_nodes = grouped_nodes
        self._grouped_nodes_dict = {node.id_: node for node in grouped_nodes}
        self._small_toks = small_toks
        self._small_toks_dict = {node.id_: node for node in self._small_toks}

        self._similarity_top_k = similarity_top_k
        self._vec_store = vector_store
        self._embed_model = Settings.embed_model

    def _retrieve(self, query_bundle: QueryBundle) -> t.List[NodeWithScore]:
        """
        Retrieves.

        Args:
            query_bundle (QueryBundle): query bundle

        Returns:
            t.List[NodeWithScore]: nodes with scores

        """
        # make query
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=500
        )

        # query for answer
        query_res = self._vec_store.query(vector_store_query)

        # determine top parents of most similar children (these are long retrieval units)
        top_parents_set: t.Set[str] = set()
        top_parents: t.List[NodeWithScore] = []
        for id_, similarity in zip(query_res.ids, query_res.similarities):
            cur_node = self._small_toks_dict[id_]
            parent_id = cur_node.ref_doc_id
            if parent_id not in top_parents_set:
                top_parents_set.add(parent_id)

                parent_node = self._grouped_nodes_dict[parent_id]
                node_with_score = NodeWithScore(node=parent_node, score=similarity)
                top_parents.append(node_with_score)

                if len(top_parents_set) >= self._similarity_top_k:
                    break

        assert len(top_parents) == min(self._similarity_top_k, len(self._grouped_nodes))

        return top_parents


class LoadNodeEvent(Event):
    """Event for loading nodes."""

    small_nodes: t.Iterable[TextNode]
    grouped_nodes: t.List[TextNode]
    index: VectorStoreIndex
    similarity_top_k: int
    llm: LLM


class LongRAGWorkflow(Workflow):
    """Long RAG Workflow."""

    @step()
    async def ingest(self, ev: StartEvent) -> t.Optional[LoadNodeEvent]:
        """
        Ingestion step.

        Args:
            ctx (Context): Context
            ev (StartEvent): start event

        Returns:
            StopEvent | None: stop event with result

        """
        data_dir: str = ev.get("data_dir")
        llm: LLM = ev.get("llm")
        chunk_size: t.Optional[int] = ev.get("chunk_size")
        similarity_top_k: int = ev.get("similarity_top_k")
        small_chunk_size: int = ev.get("small_chunk_size")
        index: t.Optional[VectorStoreIndex] = ev.get("index")
        index_kwargs: t.Optional[t.Dict[str, t.Any]] = ev.get("index_kwargs")

        if any(i is None for i in [data_dir, llm, similarity_top_k, small_chunk_size]):
            return None

        if not index:
            docs = SimpleDirectoryReader(data_dir).load_data()
            if chunk_size is not None:
                nodes = split_doc(
                    chunk_size, docs
                )  # split documents into chunks of chunk_size
                grouped_nodes = get_grouped_docs(
                    nodes
                )  # get list of nodes after grouping (groups are combined into one node), these are long retrieval units
            else:
                grouped_nodes = docs

            # split large retrieval units into smaller nodes
            small_nodes = split_doc(small_chunk_size, grouped_nodes)

            index_kwargs = index_kwargs or {}
            index = VectorStoreIndex(small_nodes, **index_kwargs)
        else:
            # get smaller nodes from index and form large retrieval units from these nodes
            small_nodes = index.docstore.docs.values()
            grouped_nodes = get_grouped_docs(small_nodes, None)

        return LoadNodeEvent(
            small_nodes=small_nodes,
            grouped_nodes=grouped_nodes,
            index=index,
            similarity_top_k=similarity_top_k,
            llm=llm,
        )

    @step(pass_context=True)
    async def make_query_engine(self, ctx: Context, ev: LoadNodeEvent) -> StopEvent:
        """
        Query engine construction step.

        Args:
            ctx (Context): context
            ev (LoadNodeEvent): event

        Returns:
            StopEvent: stop event

        """
        # make retriever and query engine
        retriever = LongRAGRetriever(
            grouped_nodes=ev.grouped_nodes,
            small_toks=ev.small_nodes,
            similarity_top_k=ev.similarity_top_k,
            vector_store=ev.index.vector_store,
        )
        query_eng = RetrieverQueryEngine.from_args(retriever, ev.llm)
        ctx.data["query_eng"] = query_eng

        return StopEvent(
            result={
                "retriever": retriever,
                "query_engine": query_eng,
                "index": ev.index,
            }
        )

    @step(pass_context=True)
    async def query(self, ctx: Context, ev: StartEvent) -> t.Optional[StopEvent]:
        """
        Query step.

        Args:
            ctx (Context): context
            ev (StartEvent): start event

        Returns:
            StopEvent | None: stop event with result

        """
        query_str: t.Optional[str] = ev.get("query_str")

        if query_str is None:
            return None

        query_eng: RetrieverQueryEngine = ctx.data.get("query_eng")

        result = query_eng.query(query_str)
        return StopEvent(result=result)


class LongRAGPack(BaseLlamaPack):
    """
    Implements Long RAG.

    This implementation is based on the following paper: https://arxiv.org/pdf/2406.15319
    """

    def __init__(
        self,
        data_dir: str,
        llm: t.Optional[LLM] = None,
        chunk_size: t.Optional[int] = DEFAULT_CHUNK_SIZE,
        similarity_top_k: int = DEFAULT_TOP_K,
        small_chunk_size: int = DEFAULT_SMALL_CHUNK_SIZE,
        index: t.Optional[VectorStoreIndex] = None,
        index_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        verbose: bool = False,
    ):
        """
        Constructor.

        Args:
            data_dir (str): Data directory
            llm (t.Optional[LLM]): LLM
            chunk_size (Optional[int], optional): Splits each doc to chunk_size to demonstrate grouping. Set to None to disable splitting then grouping. Defaults to DEFAULT_CHUNK_SIZE.
            similarity_top_k (int, optional): Top k. Defaults to DEFAULT_TOP_K.
            small_chunk_size (int, optional): Small chunk size to split large documents into smaller embeddings of small_chunk_size. Defaults to DEFAULT_SMALL_CHUNK_SIZE.
            index (Optional[VectorStoreIndex], optional): Vector index to use (from persist dir). If None, creates a new vector index. Defaults to None
            index_kwargs (Optional[Dict[str, Any]], optional): Kwargs to use when constructing VectorStoreIndex. Defaults to None.
            verbose (bool, Optional): Verbose mode. Defaults to False

        """
        # initialize workflow
        self._wf = LongRAGWorkflow(verbose=verbose)

        # initialize vars
        self._data_dir = data_dir
        self._llm = llm or Settings.llm
        self._chunk_size = chunk_size
        self._similarity_top_k = similarity_top_k
        self._small_chunk_size = small_chunk_size

        # run wf initialization
        result = asyncio_run(
            self._wf.run(
                data_dir=self._data_dir,
                llm=self._llm,
                chunk_size=self._chunk_size,
                similarity_top_k=self._similarity_top_k,
                small_chunk_size=self._small_chunk_size,
                index=index,
                index_kwargs=index_kwargs,
            )
        )

        self._retriever = result["retriever"]
        self._query_eng = result["query_engine"]
        self._index = result["index"]

    def get_modules(self) -> t.Dict[str, t.Any]:
        """Get Modules."""
        return {
            "query_engine": self._query_eng,
            "llm": self._llm,
            "retriever": self._retriever,
            "index": self._index,
            "workflow": self._wf,
        }

    def run(self, query: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Runs pipeline."""
        return asyncio_run(self._wf.run(query_str=query))
