"""Query processor"""

from typing import Optional

from gpt_index.indices.query.schema import QueryBundle
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import DEFAULT_HYDE_PROMPT


class BaseQueryProcessor:
    def __init__(self) -> None:
        pass

    def __call__(self, query_str: str) -> QueryBundle:
        return QueryBundle(query_str=query_str, embedding_strs=[query_str])


class HyDEQueryProcessor(BaseQueryProcessor):
    def __init__(
        self,
        llm_predictor: Optional[LLMPredictor] = None,
        hyde_prompt: Optional[Prompt] = None,
        include_original: bool = True,
    ) -> None:
        super().__init__()

        self._llm_predictor = llm_predictor or LLMPredictor()
        self._hyde_prompt = hyde_prompt or DEFAULT_HYDE_PROMPT
        self._include_original = include_original

    def __call__(self, query_str: str) -> QueryBundle:
        """Override QueryProcessor.process_query"""
        # TODO: support generating multiple hypothetical docs
        hypothetical_doc, _ = self._llm_predictor.predict(
            self._hyde_prompt, context_str=query_str
        )
        embedding_strs = [hypothetical_doc]
        if self._include_original:
            embedding_strs.append(query_str)
        return QueryBundle(
            query_str=query_str,
            embedding_strs=embedding_strs,
        )
        
