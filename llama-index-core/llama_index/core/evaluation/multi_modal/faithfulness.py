"""Faithfulness evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.llms import ImageBlock, LLM, TextBlock, ChatMessage
from llama_index.core.prompts.mixin import PromptDictType

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    "Please tell if a given piece of information "
    "is supported by the visual as well as textual context information.\n"
    "You need to answer with either YES or NO.\n"
    "Answer YES if any of the image(s) and textual context supports the information, even "
    "if most of the context is unrelated. "
    "Some examples are provided below with only text context, but please do use\n"
    "any images for context if they are provided.\n\n"
    "Information: Apple pie is generally double-crusted.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: YES\n"
    "Information: Apple pies tastes bad.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: NO\n"
    "Information: {query_str}\n"
    "Context: {context_str}\n"
    "Answer: "
)

DEFAULT_REFINE_TEMPLATE = PromptTemplate(
    "We want to understand if the following information is present "
    "in the context information: {query_str}\n"
    "We have provided an existing YES/NO answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "If the existing answer was already YES, still answer YES. "
    "If the information is present in the new context, answer YES. "
    "Otherwise answer NO.\n"
)


class MultiModalFaithfulnessEvaluator(BaseEvaluator):
    """
    Multi-Modal Faithfulness evaluator.

    Evaluates whether a response is faithful to the contexts
    (i.e. whether the response is supported by the contexts or hallucinated.)

    This evaluator only considers the response string and the list of context strings.

    Args:
        multi_modal_llm(Optional[LLM]):
            The LLM Judge to use for evaluations.
        raise_error(bool): Whether to raise an error when the response is invalid.
            Defaults to False.
        eval_template(Optional[Union[str, BasePromptTemplate]]):
            The template to use for evaluation.
        refine_template(Optional[Union[str, BasePromptTemplate]]):
            The template to use for refining the evaluation.

    """

    def __init__(
        self,
        multi_modal_llm: Optional[LLM] = None,
        raise_error: bool = False,
        eval_template: Union[str, BasePromptTemplate, None] = None,
        refine_template: Union[str, BasePromptTemplate, None] = None,
    ) -> None:
        """Init params."""
        if multi_modal_llm is None:
            try:
                from llama_index.llms.openai import (
                    OpenAIResponses,
                )  # pants: no-infer-dep
            except ImportError:
                raise ImportError(
                    "llama-index-llms-openai is not installed. "
                    "Please install it using `pip install llama-index-multi-modal-llms-openai`"
                )

            self._multi_modal_llm: LLM = OpenAIResponses(
                model="gpt-4.1", max_output_tokens=1000
            )
        else:
            self._multi_modal_llm = multi_modal_llm

        self._raise_error = raise_error

        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

        self._refine_template: BasePromptTemplate
        if isinstance(refine_template, str):
            self._refine_template = PromptTemplate(refine_template)
        else:
            self._refine_template = refine_template or DEFAULT_REFINE_TEMPLATE

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "eval_template": self._eval_template,
            "refine_template": self._refine_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "eval_template" in prompts:
            self._eval_template = prompts["eval_template"]
        if "refine_template" in prompts:
            self._refine_template = prompts["refine_template"]

    def evaluate(
        self,
        query: Union[str, None] = None,
        response: Union[str, None] = None,
        contexts: Union[Sequence[str], None] = None,
        image_paths: Union[List[str], None] = None,
        image_urls: Union[List[str], None] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate whether the response is faithful to the multi-modal contexts."""
        del query  # Unused
        del kwargs  # Unused
        if contexts is None or response is None:
            raise ValueError("contexts and response must be provided")

        context_str = "\n\n".join(contexts)
        fmt_prompt = self._eval_template.format(
            context_str=context_str, query_str=response
        )

        image_nodes: List[Union[ImageBlock, TextBlock]] = []

        if image_paths:
            image_nodes.extend(
                [ImageBlock(path=Path(image_path)) for image_path in image_paths]
            )
        if image_urls:
            image_nodes.extend([ImageBlock(url=image_url) for image_url in image_urls])

        image_nodes.append(TextBlock(text=fmt_prompt))

        response_obj = self._multi_modal_llm.chat(
            messages=[ChatMessage(role="user", blocks=image_nodes)],
        )

        raw_response_txt: str = response_obj.message.content or ""

        if "yes" in raw_response_txt.lower():
            passing = True
        else:
            passing = False
            if self._raise_error:
                raise ValueError("The response is invalid")

        return EvaluationResult(
            response=response,
            contexts=contexts,
            passing=passing,
            score=1.0 if passing else 0.0,
            feedback=raw_response_txt,
        )

    async def aevaluate(
        self,
        query: Union[str, None] = None,
        response: Union[str, None] = None,
        contexts: Union[Sequence[str], None] = None,
        image_paths: Union[List[str], None] = None,
        image_urls: Union[List[str], None] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Async evaluate whether the response is faithful to the multi-modal contexts."""
        del query  # Unused
        del kwargs  # Unused
        if contexts is None or response is None:
            raise ValueError("contexts and response must be provided")

        context_str = "\n\n".join(contexts)
        fmt_prompt = self._eval_template.format(
            context_str=context_str, query_str=response
        )

        blocks: List[Union[ImageBlock, TextBlock]] = []

        if image_paths:
            blocks.extend(
                [ImageBlock(path=Path(image_path)) for image_path in image_paths]
            )
        if image_urls:
            blocks.extend([ImageBlock(url=image_url) for image_url in image_urls])

        blocks.append(TextBlock(text=fmt_prompt))

        response_obj = await self._multi_modal_llm.achat(
            messages=[ChatMessage(role="user", blocks=blocks)],
        )

        raw_response_txt: str = response_obj.message.content or ""

        if "yes" in raw_response_txt.lower():
            passing = True
        else:
            passing = False
            if self._raise_error:
                raise ValueError("The response is invalid")

        return EvaluationResult(
            response=response,
            contexts=contexts,
            passing=passing,
            score=1.0 if passing else 0.0,
            feedback=raw_response_txt,
        )
