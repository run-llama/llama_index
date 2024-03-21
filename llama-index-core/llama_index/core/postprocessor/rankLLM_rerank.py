from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class RankLLMRerank(BaseNodePostprocessor):
    """RankLLM-based reranker."""
    top_n: int = Field(default=5, description="Top N nodes to return from reranking.")
    model: str = Field(default="zephyr", description="Reranker model name.")
    _model: Any = PrivateAttr()
    _result: Any = PrivateAttr()
    
    def __init__(
        self,
        top_n: int = 10,
        model: str = "zephyr",
    ):
        try: 
            from rank_llm.result import Result 
            self._result = Result
            
            from rank_llm.rerank.zephyr_reranker import ZephyrReranker
            self._model = ZephyrReranker()

        except ImportError:
            raise ImportError(
                "Cannot import rank_llm",
                "please `pip install rank_llm`",
            )

        super().__init__(
            top_n=top_n,
            model=model,
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "RankLLMRerank"
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")

        docs = [{"content": node.get_content()} for node in nodes]

        items = self._result(query=query_bundle.query_str, hits=[{"content": doc, 'qid': '1', 'docid': str(index), 'rank': str(index), 'score':str(index)} for index,doc in enumerate(docs,start=1)])

        permutation = self._model.rerank([items])[0]

        new_nodes: List[NodeWithScore] = []

        for hit in permutation.hits:
            idx: int = int(hit['docid'])
            new_nodes.append(
                NodeWithScore(node=nodes[idx-1].node, score=nodes[idx-1].score)
            )
        return new_nodes[: self.top_n]