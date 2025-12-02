from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks import trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.base.response.schema import (
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.indices.query.schema import QueryBundle, QueryType
from llama_index.core.llms import LLM, TextBlock, ChatMessage, ImageBlock
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
from llama_index.core.base.llms.generic_utils import image_node_to_image_block
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer

# from llama_index.core.query_engine.multi_modal import _get_image_and_text_nodes
from llama_index.core.llms.llm import (
    astream_chat_response_to_tokens,
    stream_chat_response_to_tokens,
)
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.settings import Settings
from llama_index.core.base.base_retriever import BaseRetriever

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


def _ensure_query_bundle(str_or_query_bundle: QueryType) -> QueryBundle:
    if isinstance(str_or_query_bundle, str):
        return QueryBundle(str_or_query_bundle)
    return str_or_query_bundle


class MultiModalContextChatEngine(BaseChatEngine):
    """
    Multimodal Context Chat Engine.

    Assumes that retrieved text context fits within context window of LLM, along with images.
    This class closely relates to the non-multimodal version, ContextChatEngine.

    Args:
        retriever (MultiModalVectorIndexRetriever): A retriever object.
        multi_modal_llm (LLM): A multimodal LLM model.
        memory (BaseMemory): Chat memory buffer to store the history.
        system_prompt (str): System prompt.
        context_template (Optional[Union[str, PromptTemplate]]): Prompt Template to embed query and context.
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): Node Postprocessors.
        callback_manager (Optional[CallbackManager]): A callback manager.

    """

    def __init__(
        self,
        retriever: "MultiModalVectorIndexRetriever",
        multi_modal_llm: LLM,
        memory: BaseMemory,
        system_prompt: str,
        context_template: Optional[Union[str, PromptTemplate]] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        self._retriever = retriever
        self._multi_modal_llm = multi_modal_llm
        context_template = context_template or DEFAULT_TEXT_QA_PROMPT
        if isinstance(context_template, str):
            context_template = PromptTemplate(context_template)
        self._context_template = context_template

        self._memory = memory
        self._system_prompt = system_prompt

        self._node_postprocessors = node_postprocessors or []
        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[Union[str, PromptTemplate]] = None,
        multi_modal_llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> "MultiModalContextChatEngine":
        """Initialize a MultiModalContextChatEngine from default parameters."""
        multi_modal_llm = multi_modal_llm or Settings.llm

        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history,
            token_limit=multi_modal_llm.metadata.context_window - 256,
        )

        system_prompt = system_prompt or ""
        node_postprocessors = node_postprocessors or []

        return cls(
            retriever,
            multi_modal_llm=multi_modal_llm,
            memory=memory,
            system_prompt=system_prompt,
            node_postprocessors=node_postprocessors,
            callback_manager=Settings.callback_manager,
            context_template=context_template,
        )

    def _apply_node_postprocessors(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes

    def _get_nodes(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    async def _aget_nodes(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        streaming: bool = False,
    ) -> RESPONSE_TYPE:
        image_nodes, text_nodes = _get_image_and_text_nodes(nodes)
        context_str = "\n\n".join(
            [r.get_content(metadata_mode=MetadataMode.LLM) for r in text_nodes]
        )
        fmt_prompt = self._context_template.format(
            context_str=context_str, query_str=query_bundle.query_str
        )

        blocks: List[Union[ImageBlock, TextBlock]] = [
            image_node_to_image_block(image_node.node)
            for image_node in image_nodes
            if isinstance(image_node.node, ImageNode)
        ]

        blocks.append(TextBlock(text=fmt_prompt))

        chat_history = self._memory.get(
            input=str(query_bundle),
        )

        if streaming:
            llm_stream = self._multi_modal_llm.stream_chat(
                [
                    ChatMessage(role="system", content=self._system_prompt),
                    *chat_history,
                    ChatMessage(role="user", blocks=blocks),
                ]
            )
            stream_tokens = stream_chat_response_to_tokens(llm_stream)
            return StreamingResponse(
                response_gen=stream_tokens,
                source_nodes=nodes,
                metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
            )
        else:
            llm_response = self._multi_modal_llm.chat(
                [
                    ChatMessage(role="system", content=self._system_prompt),
                    *chat_history,
                    ChatMessage(role="user", blocks=blocks),
                ]
            )
            output = llm_response.message.content or ""
            return Response(
                response=output,
                source_nodes=nodes,
                metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
            )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        streaming: bool = False,
    ) -> RESPONSE_TYPE:
        image_nodes, text_nodes = _get_image_and_text_nodes(nodes)
        context_str = "\n\n".join(
            [r.get_content(metadata_mode=MetadataMode.LLM) for r in text_nodes]
        )
        fmt_prompt = self._context_template.format(
            context_str=context_str, query_str=query_bundle.query_str
        )

        blocks: List[Union[ImageBlock, TextBlock]] = [
            image_node_to_image_block(image_node.node)
            for image_node in image_nodes
            if isinstance(image_node.node, ImageNode)
        ]

        blocks.append(TextBlock(text=fmt_prompt))

        chat_history = await self._memory.aget(
            input=str(query_bundle),
        )

        if streaming:
            llm_stream = await self._multi_modal_llm.astream_chat(
                [
                    ChatMessage(role="system", content=self._system_prompt),
                    *chat_history,
                    ChatMessage(role="user", blocks=blocks),
                ]
            )
            stream_tokens = await astream_chat_response_to_tokens(llm_stream)
            return AsyncStreamingResponse(
                response_gen=stream_tokens,
                source_nodes=nodes,
                metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
            )
        else:
            llm_response = await self._multi_modal_llm.achat(
                [
                    ChatMessage(role="system", content=self._system_prompt),
                    *chat_history,
                    ChatMessage(role="user", blocks=blocks),
                ]
            )
            output = llm_response.message.content or ""
            return Response(
                response=output,
                source_nodes=nodes,
                metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
            )

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        # get nodes and postprocess them
        nodes = self._get_nodes(_ensure_query_bundle(message))
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        response = self.synthesize(
            _ensure_query_bundle(message), nodes=nodes, streaming=False
        )

        user_message = ChatMessage(content=str(message), role=MessageRole.USER)
        ai_message = ChatMessage(content=str(response), role=MessageRole.ASSISTANT)

        self._memory.put(user_message)
        self._memory.put(ai_message)

        return AgentChatResponse(
            response=str(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=response.metadata,
                )
            ],
            source_nodes=response.source_nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        # get nodes and postprocess them
        nodes = self._get_nodes(_ensure_query_bundle(message))
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        response = self.synthesize(
            _ensure_query_bundle(message), nodes=nodes, streaming=True
        )
        assert isinstance(response, StreamingResponse)

        def wrapped_gen(response: StreamingResponse) -> ChatResponseGen:
            full_response = ""
            for token in response.response_gen:
                full_response += token
                yield ChatResponse(
                    message=ChatMessage(
                        content=full_response, role=MessageRole.ASSISTANT
                    ),
                    delta=token,
                )

            user_message = ChatMessage(content=str(message), role=MessageRole.USER)
            ai_message = ChatMessage(content=full_response, role=MessageRole.ASSISTANT)
            self._memory.put(user_message)
            self._memory.put(ai_message)

        return StreamingAgentChatResponse(
            chat_stream=wrapped_gen(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=response.metadata,
                )
            ],
            source_nodes=response.source_nodes,
            is_writing_to_memory=False,
        )

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            await self._memory.aset(chat_history)

        # get nodes and postprocess them
        nodes = await self._aget_nodes(_ensure_query_bundle(message))
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        response = await self.asynthesize(
            _ensure_query_bundle(message), nodes=nodes, streaming=False
        )

        user_message = ChatMessage(content=str(message), role=MessageRole.USER)
        ai_message = ChatMessage(content=str(response), role=MessageRole.ASSISTANT)

        await self._memory.aput(user_message)
        await self._memory.aput(ai_message)

        return AgentChatResponse(
            response=str(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=response.metadata,
                )
            ],
            source_nodes=response.source_nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            await self._memory.aset(chat_history)

        # get nodes and postprocess them
        nodes = await self._aget_nodes(_ensure_query_bundle(message))
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        response = await self.asynthesize(
            _ensure_query_bundle(message), nodes=nodes, streaming=True
        )
        assert isinstance(response, AsyncStreamingResponse)

        async def wrapped_gen(response: AsyncStreamingResponse) -> ChatResponseAsyncGen:
            full_response = ""
            async for token in response.async_response_gen():
                full_response += token
                yield ChatResponse(
                    message=ChatMessage(
                        content=full_response, role=MessageRole.ASSISTANT
                    ),
                    delta=token,
                )

            user_message = ChatMessage(content=str(message), role=MessageRole.USER)
            ai_message = ChatMessage(content=full_response, role=MessageRole.ASSISTANT)

            await self._memory.aput(user_message)
            await self._memory.aput(ai_message)

        return StreamingAgentChatResponse(
            achat_stream=wrapped_gen(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=response.metadata,
                )
            ],
            source_nodes=response.source_nodes,
            is_writing_to_memory=False,
        )

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
