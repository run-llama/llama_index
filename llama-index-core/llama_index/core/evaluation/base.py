"""Evaluator."""
from abc import abstractmethod
from typing import Any, Optional, Sequence

from llama_index.core.async_utils import asyncio_run
from llama_index.core.base.response.schema import Response
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.prompts.mixin import PromptMixin, PromptMixinType


class EvaluationResult(BaseModel):
    """
    Evaluation result.

    Output of an BaseEvaluator.
    """

    query: Optional[str] = Field(default=None, description="Query string")
    contexts: Optional[Sequence[str]] = Field(
        default=None, description="Context strings"
    )
    response: Optional[str] = Field(default=None, description="Response string")
    passing: Optional[bool] = Field(
        default=None, description="Binary evaluation result (passing or not)"
    )
    feedback: Optional[str] = Field(
        default=None, description="Feedback or reasoning for the response"
    )
    score: Optional[float] = Field(default=None, description="Score for the response")
    pairwise_source: Optional[str] = Field(
        default=None,
        description=(
            "Used only for pairwise and specifies whether it is from original order of"
            " presented answers or flipped order"
        ),
    )
    invalid_result: bool = Field(
        default=False, description="Whether the evaluation result is an invalid one."
    )
    invalid_reason: Optional[str] = Field(
        default=None, description="Reason for invalid evaluation."
    )


class BaseEvaluator(PromptMixin):
    """Base Evaluator class."""

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def evaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Run evaluation with query string, retrieved contexts,
        and generated response string.

        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        return asyncio_run(
            self.aevaluate(
                query=query,
                response=response,
                contexts=contexts,
                **kwargs,
            )
        )

    @abstractmethod
    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Run evaluation with query string, retrieved contexts,
        and generated response string.

        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        raise NotImplementedError

    def evaluate_response(
        self,
        query: Optional[str] = None,
        response: Optional[Response] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Run evaluation with query string and generated Response object.

        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        response_str: Optional[str] = None
        contexts: Optional[Sequence[str]] = None
        if response is not None:
            response_str = response.response
            contexts = [node.get_content() for node in response.source_nodes]

        return self.evaluate(
            query=query, response=response_str, contexts=contexts, **kwargs
        )

    async def aevaluate_response(
        self,
        query: Optional[str] = None,
        response: Optional[Response] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Run evaluation with query string and generated Response object.

        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        response_str: Optional[str] = None
        contexts: Optional[Sequence[str]] = None
        if response is not None:
            response_str = response.response
            contexts = [node.get_content() for node in response.source_nodes]

        return await self.aevaluate(
            query=query, response=response_str, contexts=contexts, **kwargs
        )


# legacy: backward compatibility
Evaluation = EvaluationResult
