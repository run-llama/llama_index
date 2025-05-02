from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

dispatcher = get_dispatcher(__name__)

try:
    from rank_llm.rerank.reranker import Reranker
    from rank_llm.data import Request, Query, Candidate
except ImportError:
    raise ImportError("RankLLM requires `pip install rank-llm`")


class RankLLMRerank(BaseNodePostprocessor):
    """
    RankLLM reranking suite. This class allows access to several reranking models supported by RankLLM. To use a model offered by the RankLLM suite, pass the desired model's hugging face path, found at https://huggingface.co/castorini. e.g., to access LiT5-Distill-base, pass 'castorini/LiT5-Distill-base' as the model name (https://huggingface.co/castorini/LiT5-Distill-base).

    Below are all the rerankers supported with the model name to be passed as an argument to the constructor. Some model have convenience names for ease of use:
        Listwise:
            - RankZephyr. model='rank_zephyr' or 'castorini/rank_zephyr_7b_v1_full'
            - RankVicuna. model='rank_zephyr' or 'castorini/rank_vicuna_7b_v1'
            - RankGPT. Takes in a valid gpt model. e.g., 'gpt-3.5-turbo', 'gpt-4','gpt-3'
            - LiT5 Distill. model='castorini/LiT5-Distill-base'
            - LiT5 Score. model='castorini/LiT5-Score-base'
        Pointwise:
            - MonoT5. model='monot5'

    """

    model: str = Field(description="Model name.")
    top_n: Optional[int] = Field(
        description="Number of nodes to return sorted by reranking score."
    )
    window_size: Optional[int] = Field(
        description="Reranking window size. Applicable only for listwise and pairwise models."
    )
    batch_size: Optional[int] = Field(
        description="Reranking batch size. Applicable only for pointwise models."
    )

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model: str,
        top_n: Optional[int] = None,
        window_size: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__(
            model=model,
            top_n=top_n,
            window_size=window_size,
            batch_size=batch_size,
        )

        self._model = Reranker.create_agent(
            model.lower(),
            default_agent=None,
            interactive=False,
            window_size=window_size,
            batch_size=batch_size,
        )

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

        request: List[Request] = [
            Request(
                query=Query(
                    text=query_bundle.query_str,
                    qid=1,
                ),
                candidates=[
                    Candidate(
                        docid=index,
                        score=doc[1],
                        doc={
                            "body": doc[0],
                            "headings": "",
                            "title": "",
                            "url": "",
                        },
                    )
                    for index, doc in enumerate(docs)
                ],
            )
        ]

        # scores are maintained the same as generated from the retriever
        permutation = self._model.rerank_batch(
            request,
            rank_end=len(request[0].candidates),
            rank_start=0,
            shuffle_candidates=False,
            logging=False,
            top_k_retrieve=len(request[0].candidates),
        )

        new_nodes: List[NodeWithScore] = []
        for candidate in permutation[0].candidates:
            id: int = int(candidate.docid)
            new_nodes.append(NodeWithScore(node=nodes[id].node, score=nodes[id].score))

        if self.top_n is None:
            dispatcher.event(ReRankEndEvent(nodes=new_nodes))
            return new_nodes
        else:
            dispatcher.event(ReRankEndEvent(nodes=new_nodes[: self.top_n]))
            return new_nodes[: self.top_n]
