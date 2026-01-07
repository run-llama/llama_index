import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

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
from llama_index.core.base.llms.generic_utils import messages_to_history_str
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.llms import LLM, TextBlock, ChatMessage, ImageBlock
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
from llama_index.core.base.llms.generic_utils import image_node_to_image_block
from llama_index.core.memory import BaseMemory, Memory

from llama_index.core.chat_engine.multi_modal_context import _get_image_and_text_nodes
from llama_index.core.llms.llm import (
    astream_chat_response_to_tokens,
    stream_chat_response_to_tokens,
)
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.chat_engine.condense_plus_context import (
    DEFAULT_CONDENSE_PROMPT_TEMPLATE,
)
from llama_index.core.settings import Settings
from llama_index.core.base.base_retriever import BaseRetriever


logger = logging.getLogger(__name__)


class MultiModalCondensePlusContextChatEngine(BaseChatEngine):
    """
    Multi-Modal Condensed Conversation & Context Chat Engine.

    First condense a conversation and latest user message to a standalone question
    Then build a context for the standalone question from a retriever,
    Then pass the context along with prompt and user message to LLM to generate a response.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        multi_modal_llm: LLM,
        memory: BaseMemory,
        context_prompt: Optional[Union[str, PromptTemplate]] = None,
        condense_prompt: Optional[Union[str, PromptTemplate]] = None,
        system_prompt: Optional[str] = None,
        skip_condense: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ):
        self._retriever = retriever
        self._multi_modal_llm = multi_modal_llm
        self._memory = memory

        context_prompt = context_prompt or DEFAULT_TEXT_QA_PROMPT
        if isinstance(context_prompt, str):
            context_prompt = PromptTemplate(context_prompt)
        self._context_prompt_template = context_prompt

        condense_prompt = condense_prompt or DEFAULT_CONDENSE_PROMPT_TEMPLATE
        if isinstance(condense_prompt, str):
            condense_prompt = PromptTemplate(condense_prompt)
        self._condense_prompt_template = condense_prompt

        self._system_prompt = system_prompt
        self._skip_condense = skip_condense
        self._node_postprocessors = node_postprocessors or []
        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager

        self._verbose = verbose

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        multi_modal_llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        context_prompt: Optional[Union[str, PromptTemplate]] = None,
        condense_prompt: Optional[Union[str, PromptTemplate]] = None,
        skip_condense: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "MultiModalCondensePlusContextChatEngine":
        """Initialize a MultiModalCondensePlusContextChatEngine from default parameters."""
        multi_modal_llm = multi_modal_llm or Settings.llm

        chat_history = chat_history or []
        memory = memory or Memory.from_defaults(
            chat_history=chat_history,
            token_limit=multi_modal_llm.metadata.context_window - 256,
        )

        return cls(
            retriever=retriever,
            multi_modal_llm=multi_modal_llm,
            memory=memory,
            context_prompt=context_prompt,
            condense_prompt=condense_prompt,
            skip_condense=skip_condense,
            callback_manager=Settings.callback_manager,
            node_postprocessors=node_postprocessors,
            system_prompt=system_prompt,
            verbose=verbose,
        )

    def _condense_question(
        self, chat_history: List[ChatMessage], latest_message: str
    ) -> str:
        """Condense a conversation history and latest user message to a standalone question."""
        if self._skip_condense or len(chat_history) == 0:
            return latest_message

        chat_history_str = messages_to_history_str(chat_history)
        logger.debug(chat_history_str)

        llm_input = self._condense_prompt_template.format(
            chat_history=chat_history_str, question=latest_message
        )

        return str(self._multi_modal_llm.complete(llm_input))

    async def _acondense_question(
        self, chat_history: List[ChatMessage], latest_message: str
    ) -> str:
        """Condense a conversation history and latest user message to a standalone question."""
        if self._skip_condense or len(chat_history) == 0:
            return latest_message

        chat_history_str = messages_to_history_str(chat_history)
        logger.debug(chat_history_str)

        llm_input = self._condense_prompt_template.format(
            chat_history=chat_history_str, question=latest_message
        )

        return str(await self._multi_modal_llm.acomplete(llm_input))

    def _get_nodes(self, message: str) -> List[NodeWithScore]:
        """Generate context information from a message."""
        nodes = self._retriever.retrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        return nodes

    async def _aget_nodes(self, message: str) -> List[NodeWithScore]:
        """Generate context information from a message."""
        nodes = await self._retriever.aretrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        return nodes

    def _run_c2(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> Tuple[ToolOutput, List[NodeWithScore]]:
        if chat_history is not None:
            self._memory.set(chat_history)

        chat_history = self._memory.get(input=message)

        # Condense conversation history and latest message to a standalone question
        condensed_question = self._condense_question(chat_history, message)  # type: ignore
        logger.info(f"Condensed question: {condensed_question}")
        if self._verbose:
            print(f"Condensed question: {condensed_question}")

        # get the context nodes using the condensed question
        context_nodes = self._get_nodes(condensed_question)
        context_source = ToolOutput(
            tool_name="retriever",
            content=str(context_nodes),
            raw_input={"message": condensed_question},
            raw_output=context_nodes,
        )

        return context_source, context_nodes

    async def _arun_c2(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> Tuple[ToolOutput, List[NodeWithScore]]:
        if chat_history is not None:
            await self._memory.aset(chat_history)

        chat_history = await self._memory.aget(input=message)

        # Condense conversation history and latest message to a standalone question
        condensed_question = await self._acondense_question(chat_history, message)  # type: ignore
        logger.info(f"Condensed question: {condensed_question}")
        if self._verbose:
            print(f"Condensed question: {condensed_question}")

        # get the context nodes using the condensed question
        context_nodes = await self._aget_nodes(condensed_question)
        context_source = ToolOutput(
            tool_name="retriever",
            content=str(context_nodes),
            raw_input={"message": condensed_question},
            raw_output=context_nodes,
        )

        return context_source, context_nodes

    def synthesize(
        self,
        query_str: str,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        streaming: bool = False,
    ) -> RESPONSE_TYPE:
        image_nodes, text_nodes = _get_image_and_text_nodes(nodes)
        context_str = "\n\n".join(
            [r.get_content(metadata_mode=MetadataMode.LLM) for r in text_nodes]
        )
        fmt_prompt = self._context_prompt_template.format(
            context_str=context_str, query_str=query_str
        )

        blocks: List[Union[ImageBlock, TextBlock]] = [
            image_node_to_image_block(image_node.node)
            for image_node in image_nodes
            if isinstance(image_node.node, ImageNode)
        ]

        blocks.append(TextBlock(text=fmt_prompt))

        chat_history = self._memory.get(
            input=query_str,
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
        query_str: str,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        streaming: bool = False,
    ) -> RESPONSE_TYPE:
        image_nodes, text_nodes = _get_image_and_text_nodes(nodes)
        context_str = "\n\n".join(
            [r.get_content(metadata_mode=MetadataMode.LLM) for r in text_nodes]
        )
        fmt_prompt = self._context_prompt_template.format(
            context_str=context_str, query_str=query_str
        )

        blocks: List[Union[ImageBlock, TextBlock]] = [
            image_node_to_image_block(image_node.node)
            for image_node in image_nodes
            if isinstance(image_node.node, ImageNode)
        ]

        blocks.append(TextBlock(text=fmt_prompt))

        chat_history = await self._memory.aget(
            input=query_str,
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
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        context_source, context_nodes = self._run_c2(message, chat_history)

        response = self.synthesize(message, nodes=context_nodes, streaming=False)

        user_message = ChatMessage(content=str(message), role=MessageRole.USER)
        assistant_message = ChatMessage(
            content=str(response), role=MessageRole.ASSISTANT
        )
        self._memory.put(user_message)
        self._memory.put(assistant_message)

        assert context_source.tool_name == "retriever"
        # re-package raw_outputs here to provide image nodes and text nodes separately
        context_source.raw_output = response.metadata

        return AgentChatResponse(
            response=str(response),
            sources=[context_source],
            source_nodes=context_nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        context_source, context_nodes = self._run_c2(message, chat_history)

        response = self.synthesize(message, nodes=context_nodes, streaming=True)
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
            assistant_message = ChatMessage(
                content=full_response, role=MessageRole.ASSISTANT
            )
            self._memory.put(user_message)
            self._memory.put(assistant_message)

        assert context_source.tool_name == "retriever"
        # re-package raw_outputs here to provide image nodes and text nodes separately
        context_source.raw_output = response.metadata

        return StreamingAgentChatResponse(
            chat_stream=wrapped_gen(response),
            sources=[context_source],
            source_nodes=context_nodes,
            is_writing_to_memory=False,
        )

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        context_source, context_nodes = await self._arun_c2(message, chat_history)

        response = await self.asynthesize(message, nodes=context_nodes, streaming=False)

        user_message = ChatMessage(content=str(message), role=MessageRole.USER)
        assistant_message = ChatMessage(
            content=str(response), role=MessageRole.ASSISTANT
        )
        await self._memory.aput(user_message)
        await self._memory.aput(assistant_message)

        assert context_source.tool_name == "retriever"
        # re-package raw_outputs here to provide image nodes and text nodes separately
        context_source.raw_output = response.metadata

        return AgentChatResponse(
            response=str(response),
            sources=[context_source],
            source_nodes=context_nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        context_source, context_nodes = await self._arun_c2(message, chat_history)

        response = await self.asynthesize(message, nodes=context_nodes, streaming=True)
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

            user_message = ChatMessage(content=message, role=MessageRole.USER)
            assistant_message = ChatMessage(
                content=full_response, role=MessageRole.ASSISTANT
            )
            await self._memory.aput(user_message)
            await self._memory.aput(assistant_message)

        assert context_source.tool_name == "retriever"
        # re-package raw_outputs here to provide image nodes and text nodes separately
        context_source.raw_output = response.metadata

        return StreamingAgentChatResponse(
            achat_stream=wrapped_gen(response),
            sources=[context_source],
            source_nodes=context_nodes,
            is_writing_to_memory=False,
        )

    def reset(self) -> None:
        # Clear chat history
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
