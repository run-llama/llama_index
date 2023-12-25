"""Pathway Retriever."""

import logging
from typing import Any, Callable, List, Optional, Union

from llama_index.callbacks.base import CallbackManager
from llama_index.core import BaseRetriever
from llama_index.embeddings import BaseEmbedding
from llama_index.indices.query.schema import QueryBundle
from llama_index.ingestion.pipeline import run_transformations
from llama_index.schema import (
    NodeWithScore,
    QueryBundle,
    TextNode,
    TransformComponent,
)

logger = logging.getLogger(__name__)


def node_transformer(x):
    return [TextNode(text=x)]


def node_to_pathway(x):
    return [(node.text, node.extra_info) for node in x]


class PathwayVectorServer:
    """
    Build an autoupdating document indexing pipeline
    for approximate nearest neighbor search.

    Args:
        docs (list): Pathway tables, may be pw.io connectors or custom tables.

        transformations (List[TransformComponent]): list of transformation steps, has to
            include embedding as last step, optionally splitter and other
            TransformComponent in the middle

        parser (Callable[[bytes], list[tuple[str, dict]]]): optional, callable that
            parses file contents into a list of documents. If None, defaults to `uft-8` decoding of the file contents. Defaults to None.
    """

    def __init__(
        self,
        *docs,
        transformations: List[TransformComponent],
        parser: Optional[Callable[[bytes], list[tuple[str, dict]]]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            from pathway.xpacks.llm import vector_store
        except ImportError:
            raise ImportError(
                "Could not import pathway python package. "
                "Please install it with `pip install pathway`."
            )

        if transformations is None or not transformations:
            raise ValueError("Transformations list cannot be None or empty.")

        if not isinstance(transformations[-1], BaseEmbedding):
            raise ValueError(
                f"Last step of transformations should be an instance of {BaseEmbedding.__name__}, "
                f"found {type(transformations[-1])}."
            )

        embedder = transformations.pop()

        def embedding_callable(x):
            return embedder.get_text_embedding(x)

        transformations.insert(0, node_transformer)
        transformations.append(node_to_pathway)  # TextNode -> (str, dict)

        def generic_transformer(x):
            return run_transformations(x, transformations)

        self.vector_store_server = vector_store.VectorStoreServer(
            *docs,
            embedder=embedding_callable,
            parser=parser,
            splitter=generic_transformer,
            **kwargs,
        )

    def run_server(
        self,
        host,
        port,
        threaded=False,
        with_cache=True,
        cache_backend=None,
    ):
        """
        Run the server and start answering queries.

        Args:
            host (str): host to bind the HTTP listener
            port (str | int): port to bind the HTTP listener
            threaded (bool): if True, run in a thread. Else block computation
            with_cache (bool): if True, embedding requests for the same contents are cached
            cache_backend: the backend to use for caching if it is enabled. The
              default is the disk cache, hosted locally in the folder ``./Cache``. You
              can use ``Backend`` class of the [`persistence API`]
              (/developers/api-docs/persistence-api/#pathway.persistence.Backend)
              to override it.

        Returns:
            If threaded, return the Thread object. Else, does not return.
        """
        try:
            import pathway as pw
        except ImportError:
            raise ImportError(
                "Could not import pathway python package. "
                "Please install it with `pip install pathway`."
            )
        if with_cache and cache_backend is None:
            cache_backend = pw.persistence.Backend.filesystem("./Cache")
        return self.vector_store_server.run_server(
            host,
            port,
            threaded=threaded,
            with_cache=with_cache,
            cache_backend=cache_backend,
        )


class PathwayRetriever(BaseRetriever):
    """Pathway retriever.
    Pathway is an open data processing framework.
    It allows you to easily develop data transformation pipelines
    that work with live data sources and changing data.

    This is the client that implements Retriever API for PathwayVectorServer.
    """

    def __init__(
        self,
        host: str,
        port: Union[str, int],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Initializing the Pathway retriever client."""
        import_err_msg = "`pathway` package not found, please run `pip install pathway`"
        try:
            from pathway.xpacks.llm.vector_store import VectorStoreClient
        except ImportError:
            raise ImportError(import_err_msg)
        self.client = VectorStoreClient(host, port)
        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        rets = self.client(query=query_bundle.query_str, k=3)

        return [
            NodeWithScore(
                node=TextNode(text=ret["text"], extra_info=ret["metadata"]),
                score=ret["dist"],
            )
            for ret in rets
        ]
