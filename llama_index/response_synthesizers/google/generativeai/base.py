"""Google GenerativeAI Attributed Question and Answering (AQA) service.

The GenAI Semantic AQA API is a managed end to end service that allows
developers to create responses grounded on specified passages based on
a user query. For more information visit:
https://developers.generativeai.google/guide
"""

import logging
from typing import Any, List, Optional, Sequence, cast

from llama_index.bridge.pydantic import BaseModel  # type: ignore
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.indices.query.schema import QueryBundle
from llama_index.prompts.mixin import PromptDictType
from llama_index.response.schema import Response
from llama_index.response_synthesizers.base import BaseSynthesizer, QueryTextType
from llama_index.schema import MetadataMode, NodeWithScore, TextNode
from llama_index.types import RESPONSE_TEXT_TYPE
from llama_index.vector_stores.google.generativeai import google_service_context

_logger = logging.getLogger(__name__)
_import_err_msg = "`google.generativeai` package not found, please run `pip install google-generativeai`"
_separator = "\n\n"


class SynthesizedResponse(BaseModel):
    """Response of `GoogleTextSynthesizer.get_response`."""

    answer: str
    """The grounded response to the user's question."""

    attributed_passages: List[str]
    """The list of passages the AQA model used for its response."""

    answerable_probability: float
    """The probability of the question being answered from the provided passages."""


class GoogleTextSynthesizer(BaseSynthesizer):
    """Google's Attributed Question and Answering service.

    Given a user's query and a list of passages, Google's server will return
    a response that is grounded to the provided list of passages. It will not
    base the response on parametric memory.
    """

    _client: Any
    _answer_style: int

    def __init__(self, answer_style: int = 1, **kwargs: Any):
        """Create a new Google AQA.

        Args:
          answer_style: See `google.ai.generativelanguage.AnswerStyle`
        """
        try:
            import google.ai.generativelanguage as genai

            import llama_index.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        super().__init__(
            service_context=google_service_context,
            output_cls=SynthesizedResponse,
            **kwargs,
        )
        self._client = genaix.build_text_service()
        self._answer_style = genai.AnswerStyle(answer_style)

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> SynthesizedResponse:
        """Generate a grounded response on provided passages.

        Args:
            query_str: The user's question.
            text_chunks: A list of passages that should be used to answer the
                question.

        Returns:
            A `SynthesizedResponse` object.
        """
        try:
            import google.ai.generativelanguage as genai

            import llama_index.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        _logger.debug(
            f"""\n
GoogleTextSynthesizer.get_response(
    query_str="{query_str}",
    text_chunks=[
{_separator.join(text_chunks)}
    ],
    {response_kwargs})"""
        )

        client = cast(genai.TextServiceClient, self._client)
        response = genaix.generate_text_answer(
            prompt=query_str,
            passages=list(text_chunks),
            answer_style=genai.AnswerStyle(self._answer_style),
            client=client,
        )

        return SynthesizedResponse(
            answer=response.answer,
            attributed_passages=[
                passage.text for passage in response.attributed_passages
            ],
            answerable_probability=response.answerable_probability,
        )

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        # TODO: Implement a true async version.
        return self.get_response(query_str, text_chunks, **response_kwargs)

    def synthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> Response:
        if len(nodes) == 0:
            return Response("Empty Response")

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE, payload={EventPayload.QUERY_STR: query.query_str}
        ) as event:
            internal_response = self.get_response(
                query_str=query.query_str,
                text_chunks=[
                    n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes
                ],
                **response_kwargs,
            )

            additional_source_nodes = list(additional_source_nodes or [])

            external_response = self._prepare_external_response(
                internal_response, additional_source_nodes
            )

            event.on_end(payload={EventPayload.RESPONSE: external_response})

        return external_response

    async def asynthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> Response:
        # TODO: Implement a true async version.
        return self.synthesize(query, nodes, additional_source_nodes, **response_kwargs)

    def _prepare_external_response(
        self,
        response: SynthesizedResponse,
        source_nodes: List[NodeWithScore],
    ) -> Response:
        return Response(
            response=response.answer,
            source_nodes=[
                NodeWithScore(
                    node=TextNode(text="\n".join(response.attributed_passages)),
                    score=response.answerable_probability,
                ),
                *source_nodes,
            ],
        )

    def _get_prompts(self) -> PromptDictType:
        # Not used.
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        # Not used.
        ...
