"""Query processor"""

from typing import Optional

from gpt_index.indices.query.schema import QueryBundle
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import DEFAULT_HYDE_PROMPT


class BaseQueryProcessor:
    def __init__(self) -> None:
        pass

    def __call__(self, query: str | QueryBundle) -> QueryBundle:
        if isinstance(query, str):
            return QueryBundle(query_str=query, embedding_strs=[query])
        elif isinstance(query, QueryBundle):
            return query
        else:
            raise ValueError("Unknown query type")


class HyDEQueryProcessor(BaseQueryProcessor):
    def __init__(
        self,
        llm_predictor: Optional[LLMPredictor] = None,
        hyde_prompt: Optional[Prompt] = None,
    ) -> None:
        super().__init__()

        self._llm_predictor = llm_predictor or LLMPredictor()
        self._hyde_prompt = hyde_prompt or DEFAULT_HYDE_PROMPT

    def __call__(self, query: str | QueryBundle) -> QueryBundle:
        """Override QueryProcessor.process_query"""

        if isinstance(query, str):
            query_str = query
        elif isinstance(query, QueryBundle):
            query_str = query.query_str
        else:
            raise ValueError('Unknown query type')

        hypothetical_doc, _ = self._llm_predictor.predict(
            self._hyde_prompt, context_str=query_str
        )
        return QueryBundle(
            query_str=query_str,
            embedding_strs=[hypothetical_doc]
        )
