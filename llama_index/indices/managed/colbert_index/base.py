import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base_retriever import BaseRetriever
from llama_index.data_structs.data_structs import IndexDict
from llama_index.indices.base import BaseIndex, IndexNode
from llama_index.schema import BaseNode, NodeWithScore
from llama_index.service_context import ServiceContext
from llama_index.storage.docstore.types import RefDocInfo
from llama_index.storage.storage_context import StorageContext

# TODO(jon-chuang):
# 1. Add support for updating index (inserts/deletes)
# 2. Add proper support for storage (managing/loading from the index files)
# 3. Normalize scores (not sure what the best practice is here)


class ColbertIndex(BaseIndex[IndexDict]):
    """
    Store for ColBERT v2 with PLAID indexing.

    ColBERT is a neural retrieval method that tends to work
    well in a zero-shot setting on out of domain datasets, due
    to it's use of token-level encodings (rather than sentence or
    chunk level)

    Parameters:

    index_path: directory containing PLAID index files.
    model_name: ColBERT hugging face model name.
        Default: "colbert-ir/colbertv2.0".
    show_progress: whether to show progress bar when building index.
        Default: False. noop for ColBERT for now.
    nbits: number of bits to quantize the residual vectors. Default: 2.
    kmeans_niters: number of kmeans clustering iterations. Default: 1.
    gpus: number of GPUs to use for indexing. Default: 0.
    rank: number of ranks to use for indexing. Default: 1.
    doc_maxlen: max document length. Default: 120.
    query_maxlen: max query length. Default: 60.
    kmeans_niters: number of kmeans iterations. Default: 4.

    """

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[IndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        model_name: str = "colbert-ir/colbertv2.0",
        index_name: str = "",
        show_progress: bool = False,
        nbits: int = 2,
        gpus: int = 0,
        ranks: int = 1,
        doc_maxlen: int = 120,
        query_maxlen: int = 60,
        kmeans_niters: int = 4,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.index_path = "storage/colbert_index"
        self.index_name = index_name
        self.nbits = nbits
        self.gpus = gpus
        self.ranks = ranks
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.kmeans_niters = kmeans_niters
        self._docs_pos_to_node_id: Dict[int, str] = {}
        try:
            pass
        except ImportError as exc:
            raise ImportError(
                "Please install colbert to use this feature from the repo:",
                "https://github.com/stanford-futuredata/ColBERT",
            ) from exc
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            index_name=index_name,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            objects=objects,
            **kwargs,
        )

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        raise NotImplementedError("ColbertStoreIndex does not support insertion yet.")

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError("ColbertStoreIndex does not support deletion yet.")

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        from .retriever import ColbertRetriever

        return ColbertRetriever(index=self, object_map=self._object_map, **kwargs)

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        raise NotImplementedError("ColbertStoreIndex does not support ref_doc_info.")

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        """Generate a PLAID index from the ColBERT checkpoint via its hugging face
        model_name.
        """
        from colbert import Indexer, Searcher
        from colbert.infra import ColBERTConfig, Run, RunConfig

        index_struct = IndexDict()

        docs_list = []
        for i, node in enumerate(nodes):
            docs_list.append(node.get_content())
            self._docs_pos_to_node_id[i] = node.node_id
            index_struct.add_node(node, text_id=str(i))

        with Run().context(
            RunConfig(index_root=self.index_path, nranks=self.ranks, gpus=self.gpus)
        ):
            config = ColBERTConfig(
                doc_maxlen=self.doc_maxlen,
                query_maxlen=self.query_maxlen,
                nbits=self.nbits,
                kmeans_niters=self.kmeans_niters,
            )
            indexer = Indexer(checkpoint=self.model_name, config=config)
            indexer.index(name=self.index_name, collection=docs_list, overwrite=True)
            self.store = Searcher(
                index=self.index_name, collection=docs_list, checkpoint=self.model_name
            )
        return index_struct

    # @staticmethod
    # def _normalize_scores(docs: List[Document]) -> None:
    #     "Normalizing the MaxSim scores using softmax."
    #     Z = sum(math.exp(doc.score) for doc in docs)
    #     for doc in docs:
    #         doc.score = math.exp(doc.score) / Z

    def persist(self, persist_dir: str) -> None:
        # Check if the destination directory exists
        if os.path.exists(persist_dir):
            # Remove the existing destination directory
            shutil.rmtree(persist_dir)

        # Copy PLAID vectors
        shutil.copytree(
            Path(self.index_path) / self.index_name, Path(persist_dir) / self.index_name
        )
        self._storage_context.persist(persist_dir=persist_dir)

    @classmethod
    def load_from_disk(cls, persist_dir: str, index_name: str = "") -> "ColbertIndex":
        from colbert import Searcher
        from colbert.infra import ColBERTConfig

        colbert_config = ColBERTConfig.load_from_index(Path(persist_dir) / index_name)
        searcher = Searcher(
            index=index_name, index_root=persist_dir, config=colbert_config
        )
        sc = StorageContext.from_defaults(persist_dir=persist_dir)
        colbert_index = ColbertIndex(
            index_struct=sc.index_store.index_structs()[0], storage_context=sc
        )
        docs_pos_to_node_id = {
            int(k): v for k, v in colbert_index.index_struct.nodes_dict.items()
        }
        colbert_index._docs_pos_to_node_id = docs_pos_to_node_id
        colbert_index.store = searcher
        return colbert_index

    def query(self, query_str: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Query the Colbert v2 + Plaid store.

        Returns: list of NodeWithScore.
        """
        doc_ids, _, scores = self.store.search(text=query_str, k=top_k)

        node_doc_ids = [self._docs_pos_to_node_id[id] for id in doc_ids]
        nodes = self.docstore.get_nodes(node_doc_ids)

        nodes_with_score = []

        for node, score in zip(nodes, scores):
            nodes_with_score.append(NodeWithScore(node=node, score=score))

        return nodes_with_score
