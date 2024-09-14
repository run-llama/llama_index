import asyncio
from typing import Any, List, Optional, Sequence, Tuple

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
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.settings import (
    Settings,
)
from llama_index.core.types import Thread

DEFAULT_CONTEXT_TEMPLATE = (
    "Context information is below."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)


class ContextChatEngine(BaseChatEngine):
    """
    Context Chat Engine.

    Uses a retriever to retrieve a context, set the context in the system prompt,
    and then uses an LLM to generate a response, for a fluid chat experience.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._node_postprocessors = node_postprocessors or []
        self._context_template = context_template or DEFAULT_CONTEXT_TEMPLATE

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
        prefix_messages: Optional[List[ChatMessage]] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> "ContextChatEngine":
        """Initialize a ContextChatEngine from default parameters."""
        llm = llm or Settings.llm

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
            callback_manager=Settings.callback_manager,
            context_template=context_template,
        )

    def _generate_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Generate context information from a message."""
        nodes = self._retriever.retrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes]
        )

        return self._context_template.format(context_str=context_str), nodes

    async def _agenerate_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Generate context information from a message."""
        nodes = await self._retriever.aretrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )
        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes]
        )

        return self._context_template.format(context_str=context_str), nodes

    def _get_prefix_messages_with_context(self, context_str: str) -> List[ChatMessage]:
        """Get the prefix messages with context."""
        # ensure we grab the user-configured system prompt
        system_prompt = ""
        prefix_messages = self._prefix_messages
        if (
            len(self._prefix_messages) != 0
            and self._prefix_messages[0].role == MessageRole.SYSTEM
        ):
            system_prompt = str(self._prefix_messages[0].content)
            prefix_messages = self._prefix_messages[1:]

        context_str_w_sys_prompt = system_prompt.strip() + "\n" + context_str
        return [
            ChatMessage(
                content=context_str_w_sys_prompt, role=self._llm.metadata.system_role
            ),
            *prefix_messages,
        ]

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = self._generate_context(message)

        # If the fetched context is completely empty
        if len(nodes) == 0 and prev_chunks is not None:
            context_str = "\n\n".join(
                [
                    n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                    for n in prev_chunks
                ]
            )

            # Create a new context string template by using previous nodes
            context_str_template = self._context_template.format(
                context_str=context_str
            )

        prefix_messages = self._get_prefix_messages_with_context(context_str_template)

        if hasattr(self._memory, "tokenizer_fn"):
            prefix_messages_token_count = len(
                self._memory.tokenizer_fn(
                    " ".join([(m.content or "") for m in prefix_messages])
                )
            )
        else:
            prefix_messages_token_count = 0

        all_messages = prefix_messages + self._memory.get(
            initial_token_count=prefix_messages_token_count
        )
        chat_response = self._llm.chat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return AgentChatResponse(
            response=str(chat_response.message.content),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
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

        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = self._generate_context(message)

        # If the fetched context is completely empty
        if len(nodes) == 0 and prev_chunks is not None:
            context_str = "\n\n".join(
                [
                    n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                    for n in prev_chunks
                ]
            )

            # Create a new context string template by using previous nodes
            context_str_template = self._context_template.format(
                context_str=context_str
            )

        prefix_messages = self._get_prefix_messages_with_context(context_str_template)

        if hasattr(self._memory, "tokenizer_fn"):
            initial_token_count = len(
                self._memory.tokenizer_fn(
                    " ".join([(m.content or "") for m in prefix_messages])
                )
            )
        else:
            initial_token_count = 0

        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(all_messages),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return chat_response

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[Sequence[NodeWithScore]] = None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = await self._agenerate_context(message)

        # If the fetched context is completely empty
        if len(nodes) == 0 and prev_chunks is not None:
            context_str = "\n\n".join(
                [
                    n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                    for n in prev_chunks
                ]
            )

            # Create a new context string template by using previous nodes
            context_str_template = self._context_template.format(
                context_str=context_str
            )

        prefix_messages = self._get_prefix_messages_with_context(context_str_template)

        if hasattr(self._memory, "tokenizer_fn"):
            initial_token_count = len(
                self._memory.tokenizer_fn(
                    " ".join([(m.content or "") for m in prefix_messages])
                )
            )
        else:
            initial_token_count = 0

        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = await self._llm.achat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return AgentChatResponse(
            response=str(chat_response.message.content),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[Sequence[NodeWithScore]] = None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = await self._agenerate_context(message)

        # If the fetched context is completely empty
        if len(nodes) == 0 and prev_chunks is not None:
            context_str = "\n\n".join(
                [
                    n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                    for n in prev_chunks
                ]
            )

            # Create a new context string template by using previous nodes
            context_str_template = self._context_template.format(
                context_str=context_str
            )

        prefix_messages = self._get_prefix_messages_with_context(context_str_template)

        if hasattr(self._memory, "tokenizer_fn"):
            initial_token_count = len(
                self._memory.tokenizer_fn(
                    " ".join([(m.content or "") for m in prefix_messages])
                )
            )
        else:
            initial_token_count = 0

        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(all_messages),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )
        asyncio.create_task(chat_response.awrite_response_to_history(self._memory))

        return chat_response

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
