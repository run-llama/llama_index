"""Google GenerativeAI Attributed Question and Answering (AQA) service.

The GenAI Semantic AQA API is a managed end to end service that allows
developers to create responses grounded on specified passages based on
a user query. For more information visit:
https://developers.generativeai.google/guide
"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, cast

from llama_index.legacy.bridge.pydantic import BaseModel  # type: ignore
from llama_index.legacy.callbacks.schema import CBEventType, EventPayload
from llama_index.legacy.core.response.schema import Response
from llama_index.legacy.indices.query.schema import QueryBundle
from llama_index.legacy.prompts.mixin import PromptDictType
from llama_index.legacy.response_synthesizers.base import BaseSynthesizer, QueryTextType
from llama_index.legacy.schema import MetadataMode, NodeWithScore, TextNode
from llama_index.legacy.types import RESPONSE_TEXT_TYPE
from llama_index.legacy.vector_stores.google.generativeai import google_service_context

if TYPE_CHECKING:
    import google.ai.generativelanguage as genai


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
    """The model's estimate of the probability that its answer is correct and grounded in the input passages."""


class GoogleTextSynthesizer(BaseSynthesizer):
    """Google's Attributed Question and Answering service.

    Given a user's query and a list of passages, Google's server will return
    a response that is grounded to the provided list of passages. It will not
    base the response on parametric memory.
    """

    _client: Any
    _temperature: float
    _answer_style: Any
    _safety_setting: List[Any]

    def __init__(
        self,
        *,
        temperature: float,
        answer_style: Any,
        safety_setting: List[Any],
        **kwargs: Any,
    ):
        """Create a new Google AQA.

        Prefer to use the factory `from_defaults` instead for type safety.
        See `from_defaults` for more documentation.
        """
        try:
            import llama_index.legacy.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        super().__init__(
            service_context=google_service_context,
            output_cls=SynthesizedResponse,
            **kwargs,
        )

        self._client = genaix.build_generative_service()
        self._temperature = temperature
        self._answer_style = answer_style
        self._safety_setting = safety_setting

    # Type safe factory that is only available if Google is installed.
    @classmethod
    def from_defaults(
        cls,
        temperature: float = 0.7,
        answer_style: int = 1,
        safety_setting: List["genai.SafetySetting"] = [],
    ) -> "GoogleTextSynthesizer":
        """Create a new Google AQA.

        Example:
          responder = GoogleTextSynthesizer.create(
              temperature=0.7,
              answer_style=AnswerStyle.ABSTRACTIVE,
              safety_setting=[
                  SafetySetting(
                      category=HARM_CATEGORY_SEXUALLY_EXPLICIT,
                      threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                  ),
              ]
          )

        Args:
          temperature: 0.0 to 1.0.
          answer_style: See `google.ai.generativelanguage.GenerateAnswerRequest.AnswerStyle`
            The default is ABSTRACTIVE (1).
          safety_setting: See `google.ai.generativelanguage.SafetySetting`.

        Returns:
          an instance of GoogleTextSynthesizer.
        """
        return cls(
            temperature=temperature,
            answer_style=answer_style,
            safety_setting=safety_setting,
        )

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

            import llama_index.legacy.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        client = cast(genai.GenerativeServiceClient, self._client)
        response = genaix.generate_answer(
            prompt=query_str,
            passages=list(text_chunks),
            answer_style=self._answer_style,
            safety_settings=self._safety_setting,
            temperature=self._temperature,
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
        """Returns a grounded response based on provided passages.

        Returns:
            Response's `source_nodes` will begin with a list of attributed
            passages. These passages are the ones that were used to construct
            the grounded response. These passages will always have no score,
            the only way to mark them as attributed passages. Then, the list
            will follow with the originally provided passages, which will have
            a score from the retrieval.

            Response's `metadata` may also have have an entry with key
            `answerable_probability`, which is the model's estimate of the
            probability that its answer is correct and grounded in the input
            passages.
        """
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
                internal_response, nodes + additional_source_nodes
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
                NodeWithScore(node=TextNode(text=passage))
                for passage in response.attributed_passages
            ]
            + source_nodes,
            metadata={
                "answerable_probability": response.answerable_probability,
            },
        )

    def _get_prompts(self) -> PromptDictType:
        # Not used.
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        # Not used.
        ...
