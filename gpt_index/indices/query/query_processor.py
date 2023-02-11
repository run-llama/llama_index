"""Query processor"""

from abc import abstractmethod
from typing import Optional, List
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.default_prompts import DEFAULT_HYDE_PROMPT


class QueryProcessor:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, query_str: str) -> List[str]:
        ...


class HyDEQueryProcessor(QueryProcessor):
    def __init__(self, llm_predictor: Optional[LLMPredictor] = None, hyde_prompt: Optional[str] = None) -> None:
        super().__init__()

        self._llm_predictor = llm_predictor or LLMPredictor()
        self._hyde_prompt = hyde_prompt or DEFAULT_HYDE_PROMPT
    
    def __call__(self, query_str: str) -> List[str]:
        """Override QueryProcessor.process_query"""

        hypothetical_doc, _ = self._llm_predictor.predict(self._hyde_prompt, context_str=query_str)
        return hypothetical_doc