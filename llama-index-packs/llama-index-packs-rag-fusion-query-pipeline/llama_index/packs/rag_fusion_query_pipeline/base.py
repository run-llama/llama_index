"""RAG Fusion Pipeline."""

from typing import Any, Dict, List, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_pipeline.components.argpacks import ArgPackComponent
from llama_index.core.query_pipeline.components.function import FnComponent
from llama_index.core.query_pipeline.components.input import InputComponent
from llama_index.core.query_pipeline.query import QueryPipeline
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

DEFAULT_CHUNK_SIZES = [128, 256, 512, 1024]


def reciprocal_rank_fusion(
    results: List[NodeWithScore],
) -> List[NodeWithScore]:
    """Apply reciprocal rank fusion.

    The original paper uses k=60 for best results:
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
    """
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for rank, node_with_score in enumerate(
        sorted(results, key=lambda x: x.score or 0.0, reverse=True)
    ):
        text = node_with_score.node.get_content()
        text_to_node[text] = node_with_score
        if text not in fused_scores:
            fused_scores[text] = 0.0
        fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes


class RAGFusionPipelinePack(BaseLlamaPack):
    """RAG Fusion pipeline.

    Create a bunch of vector indexes of different chunk sizes.

    """

    def __init__(
        self,
        documents: List[Document],
        llm: Optional[LLM] = None,
        embed_model: Optional[Any] = "default",
        chunk_sizes: Optional[List[int]] = None,
    ) -> None:
        """Init params."""
        self.documents = documents
        self.chunk_sizes = chunk_sizes or DEFAULT_CHUNK_SIZES

        # construct index
        self.llm = llm or OpenAI(model="gpt-3.5-turbo")

        self.query_engines = []
        self.retrievers = {}
        for chunk_size in self.chunk_sizes:
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)
            nodes = splitter.get_nodes_from_documents(documents)

            vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
            self.query_engines.append(vector_index.as_query_engine(llm=self.llm))

            self.retrievers[str(chunk_size)] = vector_index.as_retriever()

        # define rerank component
        rerank_component = FnComponent(fn=reciprocal_rank_fusion)

        # construct query pipeline
        p = QueryPipeline()
        module_dict = {
            **self.retrievers,
            "input": InputComponent(),
            "summarizer": TreeSummarize(llm=llm),
            # NOTE: Join args
            "join": ArgPackComponent(),
            "reranker": rerank_component,
        }
        p.add_modules(module_dict)
        # add links from input to retriever (id'ed by chunk_size)
        for chunk_size in self.chunk_sizes:
            p.add_link("input", str(chunk_size))
            p.add_link(str(chunk_size), "join", dest_key=str(chunk_size))
        p.add_link("join", "reranker")
        p.add_link("input", "summarizer", dest_key="query_str")
        p.add_link("reranker", "summarizer", dest_key="nodes")

        self.query_pipeline = p

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "llm": self.llm,
            "retrievers": self.retrievers,
            "query_engines": self.query_engines,
            "query_pipeline": self.query_pipeline,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_pipeline.run(*args, **kwargs)
