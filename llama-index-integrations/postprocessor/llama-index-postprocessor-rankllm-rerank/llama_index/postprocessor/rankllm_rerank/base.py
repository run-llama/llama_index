from typing import Any, List, Optional
from enum import Enum

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

dispatcher = get_dispatcher(__name__)


class RankLLMRerank(BaseNodePostprocessor):
    """RankLLM-based reranker."""

    top_n: int = Field(default=5, description="Top N nodes to return from reranking.")
    model: str = Field(default="zephyr", description="Reranker model name.")
    with_retrieval: bool = Field(
        default=False, description="Perform retrieval before reranking."
    )
    step_size: int = Field(
        default=10, description="Step size for moving sliding window."
    )
    gpt_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name.")
    _model: Any = PrivateAttr()
    _result: Any = PrivateAttr()
    _retriever: Any = PrivateAttr()

    def __init__(
        self,
        model,
        top_n: int = 10,
        with_retrieval: Optional[bool] = False,
        step_size: Optional[int] = 10,
        gpt_model: Optional[str] = "gpt-3.5-turbo",
    ):
        try:
            model_enum = ModelType(model.lower())
        except ValueError:
            raise ValueError(
                "Unsupported model type. Please use 'vicuna', 'zephyr', or 'gpt'."
            )

        from rank_llm.result import Result

        super().__init__(
            model=model,
            top_n=top_n,
            with_retrieval=with_retrieval,
            step_size=step_size,
            gpt_model=gpt_model,
        )

        self._result = Result

        if model_enum == ModelType.VICUNA:
            from rank_llm.rerank.vicuna_reranker import VicunaReranker

            self._model = VicunaReranker()
        elif model_enum == ModelType.ZEPHYR:
            from rank_llm.rerank.zephyr_reranker import ZephyrReranker

            self._model = ZephyrReranker()
        elif model_enum == ModelType.GPT:
            from rank_llm.rerank.rank_gpt import SafeOpenai
            from rank_llm.rerank.reranker import Reranker
            from llama_index.llms.openai import OpenAI

            llm = OpenAI(
                model=gpt_model,
                temperature=0.0,
            )

            llm.metadata

            agent = SafeOpenai(model=gpt_model, context_size=4096, keys=llm.api_key)
            self._model = Reranker(agent)
        if with_retrieval:
            from rank_llm.retrieve.retriever import Retriever

            self._retriever = Retriever

    @classmethod
    def class_name(cls) -> str:
        return "RankLLMRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model,
            )
        )

        docs = [
            (node.get_content(metadata_mode=MetadataMode.EMBED), node.get_score())
            for node in nodes
        ]

        if self.with_retrieval:
            hits = [
                {
                    "content": doc[0],
                    "qid": 1,
                    "docid": str(index),
                    "rank": index,
                    "score": doc[1],
                }
                for index, doc in enumerate(docs)
            ]
            retrieved_results = self._retriever.from_inline_hits(
                query=query_bundle.query_str, hits=hits
            )
        else:
            retrieved_results = [
                self._result(
                    query=query_bundle.query_str,
                    hits=[
                        {
                            "content": doc[0],
                            "qid": 1,
                            "docid": str(index),
                            "rank": index,
                            "score": doc[1],
                        }
                        for index, doc in enumerate(docs)
                    ],
                )
            ]

        permutation = self._model.rerank(
            retrieved_results=retrieved_results,
            rank_end=len(docs),
            window_size=min(20, len(docs)),
            step=self.step_size,
        )

        new_nodes: List[NodeWithScore] = []
        for hit in permutation[0].hits:
            idx: int = int(hit["docid"])
            new_nodes.append(
                NodeWithScore(node=nodes[idx].node, score=nodes[idx].score)
            )

        dispatcher.event(ReRankEndEvent(nodes=new_nodes[: self.top_n]))
        return new_nodes[: self.top_n]


class ModelType(Enum):
    VICUNA = "vicuna"
    ZEPHYR = "zephyr"
    GPT = "gpt"
