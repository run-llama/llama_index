import logging
from typing import Any, Dict, List, Optional, Sequence

from llama_index.bridge.pydantic import Field
from llama_index.llms import ChatMessage, ChatResponse, OpenAI
from llama_index.llms.openai_utils import (
    openai_modelname_to_contextsize,
    resolve_openai_credentials,
)
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.utilities.token_counting import TokenCounter
from llama_index.utils import print_text

DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo-0301"

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RankGPTRerank(BaseNodePostprocessor):
    """RankGPT-based reranker."""

    top_n: int = Field(description="Top N nodes to return.")
    model: str = Field(
        default=DEFAULT_OPENAI_MODEL, description="The OpenAI model to use."
    )
    temperature: float = Field(
        default=0.0,
        description="The temperature to use during generation.",
        gte=0.0,
        lte=1.0,
    )
    api_key: str = Field(default=None, description="The OpenAI API key.", exclude=True)
    api_base: str = Field(description="The base URL for OpenAI API.")
    api_version: str = Field(description="The API version for OpenAI API.")
    token_counter: TokenCounter = Field(description="token counter.")
    verbose: bool = Field(
        default=False, description="Whether to print intermediate steps."
    )

    def __init__(
        self,
        top_n: int = 10,
        model: Optional[str] = "gpt-3.5-turbo",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        token_counter: Optional[TokenCounter] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        api_key, api_base, api_version = resolve_openai_credentials(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )
        token_counter = TokenCounter()
        super().__init__(
            top_n=top_n,
            model=model,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            token_counter=token_counter,
            verbose=verbose,
            **kwargs,
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
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."

    def create_permutation_instruction(self, item: Dict[str, Any]) -> List[ChatMessage]:
        query = item["query"]
        num = len(item["hits"])

        max_length = 300
        while True:
            messages = self._get_prefix_prompt(query, num)
            rank = 0
            for hit in item["hits"]:
                rank += 1
                content = hit["content"]
                content = content.replace("Title: Content: ", "")
                content = content.strip()
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = " ".join(content.split()[: int(max_length)])
                messages.append(ChatMessage(role="user", content=f"[{rank}] {content}"))
                messages.append(
                    ChatMessage(role="assistant", content=f"Received passage [{rank}].")
                )
            messages.append(
                ChatMessage(role="user", content=self._get_post_prompt(query, num))
            )
            if (
                self.token_counter.estimate_tokens_in_messages(messages)
                <= openai_modelname_to_contextsize(self.model) - 200
            ):
                break
            else:
                max_length -= 1
        return messages

    def run_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        return OpenAI(temperature=0, model=self.model, api_key=self.api_key).chat(
            messages=messages
        )

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
