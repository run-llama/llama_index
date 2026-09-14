"""A LlamaIndex retriever backed by Whoosh BM25 (pure-Python lexical search).

LlamaIndex pipelines usually reach for a *vector* index, but dense retrieval has
a well-known blind spot: it can quietly miss the *exact* tokens that matter most
(product SKUs, function names, error codes like ``ERR_2043``, gene symbols,
ticket IDs). A lexical BM25 retriever is the classic complement -- and Whoosh
gives you one in pure Python, with no server, no native wheels, and an index
that is just a folder on disk.

``WhooshRetriever`` is a drop-in ``llama_index.core.retrievers.BaseRetriever``
you can wire into any query engine or a ``QueryFusionRetriever`` (for hybrid
search) exactly like any other retriever::

    from llama_index.retrievers.whoosh import WhooshRetriever

    retriever = WhooshRetriever.from_texts(
        texts=["Whoosh is a pure-Python search library.",
               "BM25 ranks by term rarity."],
        ids=["a", "b"],
        metadatas=[{"src": "readme"}, {"src": "docs"}],
        k=4,
    )
    nodes = retriever.retrieve("pure python search")   # -> list[NodeWithScore]

For true *hybrid* search, drop this retriever and your vector retriever into
LlamaIndex's ``QueryFusionRetriever``; it does Reciprocal Rank Fusion for you.
"""

from __future__ import annotations

from collections.abc import Sequence

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from whoosh.retrieval import WhooshSearch

__all__ = ["WhooshRetriever"]


class WhooshRetriever(BaseRetriever):
    """A LlamaIndex retriever that ranks nodes with Whoosh BM25.

    Parameters
    ----------
    core:
        A :class:`whoosh.retrieval.WhooshSearch` instance. Build one directly
        (``WhooshSearch.from_texts(...)`` / ``WhooshSearch.open_dir(path)``) for
        full control, or use :meth:`from_texts` / :meth:`from_index` below.
    k:
        The maximum number of nodes to return per query (default ``4``).
    """

    def __init__(self, core: WhooshSearch, k: int = 4) -> None:
        self._core = core
        self._k = k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return [
            NodeWithScore(
                node=TextNode(
                    text=hit.text,
                    id_=hit.id,
                    metadata={"id": hit.id, **hit.metadata},
                ),
                score=hit.score,
            )
            for hit in self._core.search(query_bundle.query_str, self._k)
        ]

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        *,
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict] | None = None,
        path: str | None = None,
        k: int = 4,
    ) -> WhooshRetriever:
        """Build an in-memory (or on-disk) index from parallel lists.

        Pass ``path`` to persist the index to a directory; omit it to keep the
        whole thing in memory (handy for tests and notebooks).
        """
        core = WhooshSearch.from_texts(
            texts=texts, ids=ids, metadatas=metadatas, path=path
        )
        return cls(core=core, k=k)

    @classmethod
    def from_index(cls, path: str, *, k: int = 4) -> WhooshRetriever:
        """Open an index previously built with ``from_texts(..., path=...)``."""
        return cls(core=WhooshSearch.open_dir(path), k=k)
