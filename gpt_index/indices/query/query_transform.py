"""Query transform."""

from typing import Optional

from gpt_index.indices.query.schema import QueryBundle
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import DEFAULT_HYDE_PROMPT


class BaseQueryTransform:
    """Base class for query transform.

    A query transform augments a raw query string with associated transformations
    to improve index querying.
    """

    def __call__(self, query_str: str) -> QueryBundle:
        """Run query processor."""
        return QueryBundle(query_str=query_str, custom_embedding_strs=[query_str])


class HyDEQueryTransform(BaseQueryTransform):
    """Hypothetical Document Embeddings (HyDE) query transform.

    It uses an LLM to generate hypothetical answer(s) to a given query,
    and use the resulting documents as embedding strings.

    As described in `[Precise Zero-Shot Dense Retrieval without Relevance Labels]
        (https://arxiv.org/abs/2212.10496)`
    """

    def __init__(
        self,
        llm_predictor: Optional[LLMPredictor] = None,
        hyde_prompt: Optional[Prompt] = None,
        include_original: bool = True,
    ) -> None:
        """Initialize HyDEQueryTransform.

        Args:
            llm_predictor (Optional[LLMPredictor]): LLM for generating
                hypothetical documents
            hyde_prompt (Optional[Prompt]): Custom prompt for HyDE
            include_original (bool): Whether to include original query
                string as one of the embedding strings
        """
        super().__init__()

        self._llm_predictor = llm_predictor or LLMPredictor()
        self._hyde_prompt = hyde_prompt or DEFAULT_HYDE_PROMPT
        self._include_original = include_original

    def __call__(self, query_str: str) -> QueryBundle:
        """Run query transform."""
        # TODO: support generating multiple hypothetical docs
        hypothetical_doc, _ = self._llm_predictor.predict(
            self._hyde_prompt, context_str=query_str
        )
        embedding_strs = [hypothetical_doc]
        if self._include_original:
            embedding_strs.append(query_str)
        return QueryBundle(
            query_str=query_str,
            custom_embedding_strs=embedding_strs,
        )
