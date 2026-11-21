from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank


def test_class():
    names_of_base_classes = [b.__name__ for b in RankGPTRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


class _PermutingLLM(MockLLM):
    """Stands in for the lazily created default LLM."""

    def chat(self, messages, **kwargs):
        return ChatResponse(message=ChatMessage(role="assistant", content="[3] [1]"))


def test_default_llm_is_resolved_before_reranking(monkeypatch):
    """Omitting `llm` must use the lazily created default instead of crashing."""

    def _ensure_llm(self) -> None:
        self.llm = _PermutingLLM()

    monkeypatch.setattr(RankGPTRerank, "_ensure_llm", _ensure_llm)

    nodes = [
        NodeWithScore(node=TextNode(text=f"passage {i}", id_=f"node-{i}"), score=i / 10)
        for i in range(4)
    ]

    reranker = RankGPTRerank(top_n=2)
    results = reranker.postprocess_nodes(
        nodes, query_bundle=QueryBundle(query_str="which passage is first")
    )

    assert [result.node.node_id for result in results] == ["node-2", "node-0"]
