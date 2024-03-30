import asyncio
from threading import Thread
from typing import Any, List, Optional, Tuple

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)


class FlowChatEngine(BaseChatEngine):
    """Flow Chat Engine.

    First retrieves text from the index using the user message, then augments the context
    in the latest user message prompt (text_qa_template) to generate a response from the LLM.
    Finally, it condenses previous user messages to simple queries (excluding the full prompt from history).
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        text_qa_template: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._node_postprocessors = node_postprocessors or []
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_TMPL

        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        service_context: Optional[ServiceContext] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        text_qa_template: Optional[str] = None,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> "FlowChatEngine":
        """Initialize a FlowChatEngine from default parameters."""
        llm = llm or llm_from_settings_or_context(Settings, service_context)

        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=llm.metadata.context_window - 256
        )

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [
                ChatMessage(content=system_prompt, role=llm.metadata.system_role)
            ]

        prefix_messages = prefix_messages or []
        node_postprocessors = node_postprocessors or []

        return cls(
            retriever,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            node_postprocessors=node_postprocessors,
            callback_manager=callback_manager_from_settings_or_context(
                Settings, service_context
            ),
            text_qa_template=text_qa_template,
        )

    def _generate_qa_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Generate context information from a message."""
        nodes = self._retriever.retrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes]
        )

        return (
            self._text_qa_template.format(context_str=context_str, query_str=message),
            nodes,
        )

    async def _agenerate_qa_context(
        self, message: str
    ) -> Tuple[str, List[NodeWithScore]]:
        """Generate context information from a message."""
        nodes = await self._retriever.aretrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )
        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes]
        )

        return (
            self._context_template.format(context_str=context_str, query_str=message),
            nodes,
        )

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        qa_context_str_template, nodes = self._generate_qa_context(message)

        user_message = ChatMessage(
            role=MessageRole.USER, content=qa_context_str_template
        )
        self._memory.put(user_message)

        prefix_messages_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in self._prefix_messages])
            )
        )
        all_messages = self._prefix_messages + self._memory.get(
            initial_token_count=prefix_messages_token_count
        )
        chat_response = self._llm.chat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        self._memory.get_all()[-2].content = message

        return AgentChatResponse(
            response=str(chat_response.message.content),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(user_message),
                    raw_input={"message": message},
                    raw_output=user_message,
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        qa_context_str_template, nodes = self._generate_qa_context(message)

        user_message = ChatMessage(content=qa_context_str_template, role="user")
        self._memory.put(user_message)

        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in self._prefix_messages])
            )
        )
        all_messages = self._prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(all_messages),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(user_message),
                    raw_input={"message": message},
                    raw_output=user_message,
                )
            ],
            source_nodes=nodes,
        )

        def modify_last_message():
            self._memory.get_all()[-2].content = message

        thread = Thread(
            target=chat_response.write_response_to_history,
            args=(self._memory,),
            kwargs={"on_stream_end_fn": modify_last_message},
        )
        thread.start()

        return chat_response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        qa_context_str_template, nodes = self._agenerate_qa_context(message)

        user_message = ChatMessage(
            role=MessageRole.USER, content=qa_context_str_template
        )
        self._memory.put(user_message)

        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in self._prefix_messages])
            )
        )
        all_messages = self._prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = await self._llm.achat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        self._memory.get_all()[-2].content = message

        return AgentChatResponse(
            response=str(chat_response.message.content),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(user_message),
                    raw_input={"message": message},
                    raw_output=user_message,
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        qa_context_str_template, nodes = self._agenerate_qa_context(message)

        user_message = ChatMessage(content=qa_context_str_template, role="user")
        self._memory.put(user_message)

        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in self._prefix_messages])
            )
        )
        all_messages = self._prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(all_messages),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(user_message),
                    raw_input={"message": message},
                    raw_output=user_message,
                )
            ],
            source_nodes=nodes,
        )

        def modify_last_message():
            self._memory.get_all()[-2].content = message

        thread = Thread(
            target=lambda x: asyncio.run(chat_response.awrite_response_to_history(x)),
            args=(self._memory,),
            kwargs={"on_stream_end_fn": modify_last_message},
        )
        thread.start()

        return chat_response

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
