import copy
import logging
from typing import Any, Dict, List, Optional, Sequence

import tiktoken

from llama_index.bridge.pydantic import Field
from llama_index.llms import ChatMessage, ChatResponse, OpenAI
from llama_index.llms.openai_utils import (
    openai_modelname_to_contextsize,
    resolve_openai_credentials,
)
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.service_context import ServiceContext

DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo-0301"

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RankGPTRerank(BaseNodePostprocessor):
    """LLM-based reranker."""

    top_n: int = Field(description="Top N nodes to return.")
    service_context: ServiceContext = Field(
        description="Service context.", exclude=True
    )
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

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        top_n: int = 10,
        model: Optional[str] = "gpt-3.5-turbo",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        service_context = service_context or ServiceContext.from_defaults()

        api_key, api_base, api_version = resolve_openai_credentials(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )
        super().__init__(
            service_context=service_context,
            top_n=top_n,
            model=model or service_context.llm.metadata.model_name,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
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

    def _num_tokens_from_messages(
        self, messages: Sequence[ChatMessage], model: str = "gpt-3.5-turbo-0301"
    ) -> int:
        """Returns the number of tokens used by a list of messages."""
        if model == "gpt-3.5-turbo":
            return self._num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
        elif model == "gpt-4":
            return self._num_tokens_from_messages(messages, model="gpt-4-0314")
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            # tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            # tokens_per_name = 1
        else:
            tokens_per_message, tokens_per_name = 0, 0

        try:
            encoding = tiktoken.get_encoding(model)
        except Exception as e:
            logger.warning(
                f"Encountered exception when loading encoding for given model: {e}"
            )
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            if message.content is not None:
                num_tokens += len(encoding.encode(str(message.content)))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

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
            for m in messages:
                print(m)
            if (
                self._num_tokens_from_messages(messages, self.model)
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
        rank_start = 0
        rank_end = len(item["hits"])

        response = self._clean_response(permutation)
        response_list = [int(x) - 1 for x in response.split()]
        response_list = self._remove_duplicate(response_list)
        original_rank = copy.deepcopy(item["hits"][rank_start:rank_end])
        # original_rank = [tt for tt in range(len(cut_range))]
        response_list = [ss for ss in response_list if ss in original_rank]
        return response_list + [
            tt for tt in original_rank if tt not in response_list
        ]  # add the rest of the rank
