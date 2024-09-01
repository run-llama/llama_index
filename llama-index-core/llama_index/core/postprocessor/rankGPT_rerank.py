import logging
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field, SerializeAsAny
from llama_index.core.llms import LLM, ChatMessage, ChatResponse
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import RANKGPT_RERANK_PROMPT
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.utils import print_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def get_default_llm() -> LLM:
    from llama_index.llms.openai import OpenAI  # pants: no-infer-dep

    return OpenAI(model="gpt-3.5-turbo-16k")


class RankGPTRerank(BaseNodePostprocessor):
    """RankGPT-based reranker."""

    top_n: int = Field(default=5, description="Top N nodes to return from reranking.")
    llm: LLM = Field(
        default_factory=get_default_llm,
        description="LLM to use for rankGPT",
    )
    verbose: bool = Field(
        default=False, description="Whether to print intermediate steps."
    )
    rankgpt_rerank_prompt: SerializeAsAny[BasePromptTemplate] = Field(
        description="rankGPT rerank prompt."
    )

    def __init__(
        self,
        top_n: int = 5,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        rankgpt_rerank_prompt: Optional[BasePromptTemplate] = None,
    ):
        rankgpt_rerank_prompt = rankgpt_rerank_prompt or RANKGPT_RERANK_PROMPT
        super().__init__(
            verbose=verbose,
            llm=llm,
            top_n=top_n,
            rankgpt_rerank_prompt=rankgpt_rerank_prompt,
        )

    @classmethod
    def class_name(cls) -> str:
        return "RankGPTRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")

        items = {
            "query": query_bundle.query_str,
            "hits": [{"content": node.get_content()} for node in nodes],
        }

        messages = self.create_permutation_instruction(item=items)
        permutation = self.run_llm(messages=messages)
        return self._llm_result_to_nodes(permutation, nodes, items)

    async def _apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        items = {
            "query": query_bundle.query_str,
            "hits": [{"content": node.get_content()} for node in nodes],
        }

        messages = self.create_permutation_instruction(item=items)
        permutation = await self.arun_llm(messages=messages)
        return self._llm_result_to_nodes(permutation, nodes, items)

    async def apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes asynchronously."""
        if query_str is not None and query_bundle is not None:
            raise ValueError("Cannot specify both query_str and query_bundle")
        elif query_str is not None:
            query_bundle = QueryBundle(query_str)
        else:
            pass
        return await self._apostprocess_nodes(nodes, query_bundle)

    def _llm_result_to_nodes(
        self, permutation: ChatResponse, nodes: List[NodeWithScore], items: Dict
    ) -> List[NodeWithScore]:
        if permutation.message is not None and permutation.message.content is not None:
            rerank_ranks = self._receive_permutation(
                items, str(permutation.message.content)
            )
            if self.verbose:
                print_text(f"After Reranking, new rank list for nodes: {rerank_ranks}")

            initial_results: List[NodeWithScore] = []

            for idx in rerank_ranks:
                initial_results.append(
                    NodeWithScore(node=nodes[idx].node, score=nodes[idx].score)
                )
            return initial_results[: self.top_n]
        else:
            return nodes[: self.top_n]

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"rankgpt_rerank_prompt": self.rankgpt_rerank_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "rankgpt_rerank_prompt" in prompts:
            self.rankgpt_rerank_prompt = prompts["rankgpt_rerank_prompt"]

    def _get_prefix_prompt(self, query: str, num: int) -> List[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content="You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
            ),
            ChatMessage(
                role="user",
                content=f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
            ),
            ChatMessage(role="assistant", content="Okay, please provide the passages."),
        ]

    def _get_post_prompt(self, query: str, num: int) -> str:
        return self.rankgpt_rerank_prompt.format(query=query, num=num)

    def create_permutation_instruction(self, item: Dict[str, Any]) -> List[ChatMessage]:
        query = item["query"]
        num = len(item["hits"])

        messages = self._get_prefix_prompt(query, num)
        rank = 0
        for hit in item["hits"]:
            rank += 1
            content = hit["content"]
            content = content.replace("Title: Content: ", "")
            content = content.strip()
            # For Japanese should cut by character: content = content[:int(max_length)]
            content = " ".join(content.split()[:300])
            messages.append(ChatMessage(role="user", content=f"[{rank}] {content}"))
            messages.append(
                ChatMessage(role="assistant", content=f"Received passage [{rank}].")
            )
        messages.append(
            ChatMessage(role="user", content=self._get_post_prompt(query, num))
        )
        return messages

    def run_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        return self.llm.chat(messages)

    async def arun_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        return await self.llm.achat(messages)

    def _clean_response(self, response: str) -> str:
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        return new_response.strip()

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def _receive_permutation(self, item: Dict[str, Any], permutation: str) -> List[int]:
        rank_end = len(item["hits"])

        response = self._clean_response(permutation)
        response_list = [int(x) - 1 for x in response.split()]
        response_list = self._remove_duplicate(response_list)
        response_list = [ss for ss in response_list if ss in range(rank_end)]
        return response_list + [
            tt for tt in range(rank_end) if tt not in response_list
        ]  # add the rest of the rank
