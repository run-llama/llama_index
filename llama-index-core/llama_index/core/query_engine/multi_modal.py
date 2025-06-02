from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.indices.query.base import BaseQueryEngine
from llama_index.core.indices.query.schema import QueryBundle, QueryType
from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode

if TYPE_CHECKING:
    from llama_index.core.indices.multi_modal import MultiModalVectorIndexRetriever


def _get_image_and_text_nodes(
    nodes: List[NodeWithScore],
) -> Tuple[List[NodeWithScore], List[NodeWithScore]]:
    image_nodes = []
    text_nodes = []
    for res_node in nodes:
        if isinstance(res_node.node, ImageNode):
            image_nodes.append(res_node)
        else:
            text_nodes.append(res_node)
    return image_nodes, text_nodes


class SimpleMultiModalQueryEngine(BaseQueryEngine):
    """
    Simple Multi Modal Retriever query engine.

    Assumes that retrieved text context fits within context window of LLM, along with images.

    Args:
        retriever (MultiModalVectorIndexRetriever): A retriever object.
        multi_modal_llm (Optional[MultiModalLLM]): MultiModalLLM Models.
        text_qa_template (Optional[BasePromptTemplate]): Text QA Prompt Template.
        image_qa_template (Optional[BasePromptTemplate]): Image QA Prompt Template.
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): Node Postprocessors.
        callback_manager (Optional[CallbackManager]): A callback manager.

    """

    def __init__(
        self,
        retriever: "MultiModalVectorIndexRetriever",
        multi_modal_llm: Optional[MultiModalLLM] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        image_qa_template: Optional[BasePromptTemplate] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        self._retriever = retriever
        if multi_modal_llm:
            self._multi_modal_llm = multi_modal_llm
        else:
            try:
                from llama_index.multi_modal_llms.openai import (
                    OpenAIMultiModal,
                )  # pants: no-infer-dep

                self._multi_modal_llm = OpenAIMultiModal(
                    model="gpt-4-vision-preview", max_new_tokens=1000
                )
            except ImportError as e:
                raise ImportError(
                    "`llama-index-multi-modal-llms-openai` package cannot be found. "
                    "Please install it by using `pip install `llama-index-multi-modal-llms-openai`"
                )
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self._image_qa_template = image_qa_template or DEFAULT_TEXT_QA_PROMPT

        self._node_postprocessors = node_postprocessors or []
        callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = callback_manager

        super().__init__(callback_manager)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {"text_qa_template": self._text_qa_template}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def _apply_node_postprocessors(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        image_nodes, text_nodes = _get_image_and_text_nodes(nodes)
        context_str = "\n\n".join(
            [r.get_content(metadata_mode=MetadataMode.LLM) for r in text_nodes]
        )
        fmt_prompt = self._text_qa_template.format(
            context_str=context_str, query_str=query_bundle.query_str
        )

        llm_response = self._multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=[
                image_node.node
                for image_node in image_nodes
                if isinstance(image_node.node, ImageNode)
            ],
        )
        return Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
        )

    def _get_response_with_images(
        self,
        prompt_str: str,
        image_nodes: List[NodeWithScore],
    ) -> RESPONSE_TYPE:
        assert all(isinstance(node.node, ImageNode) for node in image_nodes)

        fmt_prompt = self._image_qa_template.format(
            query_str=prompt_str,
        )

        llm_response = self._multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=[
                node.node for node in image_nodes if isinstance(node.node, ImageNode)
            ],
        )
        return Response(
            response=str(llm_response),
            source_nodes=image_nodes,
            metadata={"image_nodes": image_nodes},
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        image_nodes, text_nodes = _get_image_and_text_nodes(nodes)
        context_str = "\n\n".join(
            [r.get_content(metadata_mode=MetadataMode.LLM) for r in text_nodes]
        )
        fmt_prompt = self._text_qa_template.format(
            context_str=context_str, query_str=query_bundle.query_str
        )

        llm_response = await self._multi_modal_llm.acomplete(
            prompt=fmt_prompt,
            image_documents=[
                image_node.node
                for image_node in image_nodes
                if isinstance(image_node.node, ImageNode)
            ],
        )
        return Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self.retrieve(query_bundle)

                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )

            response = self.synthesize(
                query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    def image_query(self, image_path: QueryType, prompt_str: str) -> RESPONSE_TYPE:
        """Answer a image query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: str(image_path)}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: str(image_path)},
            ) as retrieve_event:
                nodes = self._retriever.image_to_image_retrieve(image_path)

                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )

            image_nodes, _ = _get_image_and_text_nodes(nodes)
            response = self._get_response_with_images(
                prompt_str=prompt_str,
                image_nodes=image_nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self.aretrieve(query_bundle)

                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )

            response = await self.asynthesize(
                query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @property
    def retriever(self) -> "MultiModalVectorIndexRetriever":
        """Get the retriever object."""
        return self._retriever
