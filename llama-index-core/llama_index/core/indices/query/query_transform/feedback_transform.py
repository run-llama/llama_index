import logging
from typing import Dict, Optional

from llama_index.core.evaluation.base import Evaluation
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle
from llama_index.core.service_context_elements.llm_predictor import (
    LLMPredictorType,
)
from llama_index.core.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_RESYNTHESIS_PROMPT_TMPL = (
    "Here is the original query:\n"
    "{query_str}\n"
    "Here is the response given:\n"
    "{response}\n"
    "Here is some feedback from evaluator about the response given.\n"
    "{feedback}\n"
    "If you want to resynthesize the query, please return the modified query below.\n"
    "Otherwise, please return the original query.\n"
)

DEFAULT_RESYNTHESIS_PROMPT = PromptTemplate(DEFAULT_RESYNTHESIS_PROMPT_TMPL)


class FeedbackQueryTransformation(BaseQueryTransform):
    """Transform the query given the evaluation feedback.

    Args:
        eval(Evaluation): An evaluation object.
        llm(LLM): An LLM.
        resynthesize_query(bool): Whether to resynthesize the query.
        resynthesis_prompt(BasePromptTemplate): A prompt for resynthesizing the query.

    """

    def __init__(
        self,
        llm: Optional[LLMPredictorType] = None,
        resynthesize_query: bool = False,
        resynthesis_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        super().__init__()
        self.llm = llm or Settings.llm
        self.should_resynthesize_query = resynthesize_query
        self.resynthesis_prompt = resynthesis_prompt or DEFAULT_RESYNTHESIS_PROMPT

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"resynthesis_prompt": self.resynthesis_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "resynthesis_prompt" in prompts:
            self.resynthesis_prompt = prompts["resynthesis_prompt"]

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        orig_query_str = query_bundle.query_str
        if metadata.get("evaluation") and isinstance(
            metadata.get("evaluation"), Evaluation
        ):
            self.evaluation = metadata.get("evaluation")
        if self.evaluation is None or not isinstance(self.evaluation, Evaluation):
            raise ValueError("Evaluation is not set.")
        if self.evaluation.response is None or self.evaluation.feedback is None:
            raise ValueError("Evaluation result must contain response and feedback.")

        if self.evaluation.feedback == "YES" or self.evaluation.feedback == "NO":
            new_query = (
                orig_query_str
                + "\n----------------\n"
                + self._construct_feedback(response=self.evaluation.response)
            )
        else:
            if self.should_resynthesize_query:
                new_query_str = self._resynthesize_query(
                    orig_query_str, self.evaluation.response, self.evaluation.feedback
                )
            else:
                new_query_str = orig_query_str
            new_query = (
                self._construct_feedback(response=self.evaluation.response)
                + "\n"
                + "Here is some feedback from the evaluator about the response given.\n"
                + self.evaluation.feedback
                + "\n"
                + "Now answer the question.\n"
                + new_query_str
            )
        return QueryBundle(new_query, custom_embedding_strs=[orig_query_str])

    @staticmethod
    def _construct_feedback(response: Optional[str]) -> str:
        """Construct feedback from response."""
        if response is None:
            return ""
        else:
            return "Here is a previous bad answer.\n" + response

    def _resynthesize_query(
        self, query_str: str, response: str, feedback: Optional[str]
    ) -> str:
        """Resynthesize query given feedback."""
        if feedback is None:
            return query_str
        else:
            new_query_str = self.llm.predict(
                self.resynthesis_prompt,
                query_str=query_str,
                response=response,
                feedback=feedback,
            )
            logger.debug("Resynthesized query: %s", new_query_str)
            return new_query_str
