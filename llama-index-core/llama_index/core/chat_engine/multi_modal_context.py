"""Multi-Modal Context Chat Engine."""

import base64
import logging
from typing import Any, List, Optional, Tuple, Union

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import ImageNode, NodeWithScore, QueryBundle
from llama_index.core.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TEMPLATE = (
    "Use the context information below to assist the user."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)


class MultiModalContextChatEngine(BaseChatEngine):
    """
    Multi-Modal Context Chat Engine.

    Retrieves both text nodes and image nodes from a MultiModal index via a
    ``MultiModalVectorIndexRetriever``, then builds a multi-modal chat message
    containing:

    * A ``TextBlock`` with the retrieved text context and the user's question.
    * One ``ImageBlock`` per retrieved ``ImageNode``.

    The assembled message list (prefix messages + memory history + user message)
    is sent to a ``MultiModalLLM``.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        multi_modal_llm: MultiModalLLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: str = DEFAULT_CONTEXT_TEMPLATE,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._multi_modal_llm = multi_modal_llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._node_postprocessors = node_postprocessors or []
        self._context_template = context_template
        self.callback_manager = callback_manager or CallbackManager([])
        for postprocessor in self._node_postprocessors:
            postprocessor.callback_manager = self.callback_manager

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        multi_modal_llm: Optional[MultiModalLLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        **kwargs: Any,
    ) -> "MultiModalContextChatEngine":
        """Create a ``MultiModalContextChatEngine`` from default parameters."""
        mm_llm = multi_modal_llm or Settings.llm
        if not isinstance(mm_llm, MultiModalLLM):
            raise ValueError(
                f"MultiModalContextChatEngine requires a MultiModalLLM, "
                f"but got {type(mm_llm).__name__}. "
                "Pass multi_modal_llm=<your MultiModalLLM instance>."
            )

        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history,
            token_limit=(mm_llm.metadata.context_window or 4096) - 256,
        )

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [
                ChatMessage(content=system_prompt, role=MessageRole.SYSTEM)
            ]

        return cls(
            retriever=retriever,
            multi_modal_llm=mm_llm,
            memory=memory,
            prefix_messages=prefix_messages or [],
            node_postprocessors=node_postprocessors or [],
            context_template=context_template or DEFAULT_CONTEXT_TEMPLATE,
            callback_manager=Settings.callback_manager,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_nodes(
        self, message: str
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore]]:
        """Retrieve nodes and split into text and image lists."""
        nodes = self._retriever.retrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )
        text_nodes = [n for n in nodes if not isinstance(n.node, ImageNode)]
        image_nodes = [n for n in nodes if isinstance(n.node, ImageNode)]
        return text_nodes, image_nodes

    async def _aget_nodes(
        self, message: str
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore]]:
        """Asynchronously retrieve nodes and split into text and image lists."""
        nodes = await self._retriever.aretrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )
        text_nodes = [n for n in nodes if not isinstance(n.node, ImageNode)]
        image_nodes = [n for n in nodes if isinstance(n.node, ImageNode)]
        return text_nodes, image_nodes

    def _build_for_llm_messages(
        self,
        message: str,
        text_nodes: List[NodeWithScore],
        image_nodes: List[NodeWithScore],
    ) -> List[ChatMessage]:
        """Build the complete message list to pass to the MultiModalLLM.

        Layout:
            prefix_messages
            + memory chat history
            + user message (TextBlock with context + question, ImageBlocks)
        """
        context_str = "\n\n".join(node.get_content() for node in text_nodes)
        context_text = self._context_template.format(context_str=context_str)

        blocks: List[Union[TextBlock, ImageBlock]] = [
            TextBlock(text=f"{context_text}\n{message}")
        ]

        for node_with_score in image_nodes:
            node = node_with_score.node
            if isinstance(node, ImageNode):
                if node.image_url:
                    blocks.append(ImageBlock(url=node.image_url))
                elif node.image_path:
                    blocks.append(ImageBlock(path=node.image_path))
                elif node.image:
                    blocks.append(
                        ImageBlock(
                            image=base64.b64decode(node.image),
                            image_mimetype=node.image_mimetype,
                        )
                    )

        user_message = ChatMessage(role=MessageRole.USER, blocks=blocks)
        chat_history = self._memory.get(input=message)
        return self._prefix_messages + chat_history + [user_message]

    # ------------------------------------------------------------------
    # BaseChatEngine interface
    # ------------------------------------------------------------------

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        text_nodes, image_nodes = self._get_nodes(message)
        all_nodes = text_nodes + image_nodes

        messages = self._build_for_llm_messages(message, text_nodes, image_nodes)
        response: ChatResponse = self._multi_modal_llm.chat(messages)
        response_text = str(response.message.content or "")

        self._memory.put(ChatMessage(content=message, role=MessageRole.USER))
        self._memory.put(
            ChatMessage(content=response_text, role=MessageRole.ASSISTANT)
        )

        return AgentChatResponse(
            response=response_text,
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(all_nodes),
                    raw_input={"message": message},
                    raw_output=all_nodes,
                )
            ],
            source_nodes=all_nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        text_nodes, image_nodes = self._get_nodes(message)
        all_nodes = text_nodes + image_nodes

        messages = self._build_for_llm_messages(message, text_nodes, image_nodes)
        response_stream = self._multi_modal_llm.stream_chat(messages)

        def gen() -> ChatResponseGen:
            full_response = ""
            for chunk in response_stream:
                delta = chunk.delta or ""
                full_response += delta
                yield chunk
            self._memory.put(ChatMessage(content=message, role=MessageRole.USER))
            self._memory.put(
                ChatMessage(content=full_response, role=MessageRole.ASSISTANT)
            )

        return StreamingAgentChatResponse(
            chat_stream=gen(),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(all_nodes),
                    raw_input={"message": message},
                    raw_output=all_nodes,
                )
            ],
            source_nodes=all_nodes,
            is_writing_to_memory=False,
        )

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            await self._memory.aset(chat_history)

        text_nodes, image_nodes = await self._aget_nodes(message)
        all_nodes = text_nodes + image_nodes

        messages = self._build_for_llm_messages(message, text_nodes, image_nodes)
        response: ChatResponse = await self._multi_modal_llm.achat(messages)
        response_text = str(response.message.content or "")

        await self._memory.aput(ChatMessage(content=message, role=MessageRole.USER))
        await self._memory.aput(
            ChatMessage(content=response_text, role=MessageRole.ASSISTANT)
        )

        return AgentChatResponse(
            response=response_text,
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(all_nodes),
                    raw_input={"message": message},
                    raw_output=all_nodes,
                )
            ],
            source_nodes=all_nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            await self._memory.aset(chat_history)

        text_nodes, image_nodes = await self._aget_nodes(message)
        all_nodes = text_nodes + image_nodes

        messages = self._build_for_llm_messages(message, text_nodes, image_nodes)
        response_stream = await self._multi_modal_llm.astream_chat(messages)

        async def async_gen() -> ChatResponseAsyncGen:
            full_response = ""
            async for chunk in response_stream:
                delta = chunk.delta or ""
                full_response += delta
                yield chunk
            await self._memory.aput(
                ChatMessage(content=message, role=MessageRole.USER)
            )
            await self._memory.aput(
                ChatMessage(content=full_response, role=MessageRole.ASSISTANT)
            )

        return StreamingAgentChatResponse(
            achat_stream=async_gen(),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(all_nodes),
                    raw_input={"message": message},
                    raw_output=all_nodes,
                )
            ],
            source_nodes=all_nodes,
            is_writing_to_memory=False,
        )

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self._memory.get_all()
