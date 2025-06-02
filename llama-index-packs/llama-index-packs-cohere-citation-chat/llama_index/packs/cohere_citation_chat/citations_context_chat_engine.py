import asyncio
from typing import List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import logging

from llama_index.core.callbacks import trace_method
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import ToolOutput
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
    is_function,
)
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.core.memory import BaseMemory
from llama_index.core.types import Thread

from .types import (
    Document,
    Citation,
    CitationsSettings,
)

from .utils import (
    convert_nodes_to_documents_list,
    convert_chat_response_to_citations,
    convert_chat_response_to_documents,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass
class AgentCitationsChatResponse(AgentChatResponse):
    """Cohere Agent chat response. Adds citations and documents to the response."""

    citations: List[Citation] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)


@dataclass
class StreamingAgentCitationsChatResponse(StreamingAgentChatResponse):
    """Streaming chat response to user and writing to chat history."""

    citations: List[Citation] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    citations_settings: CitationsSettings = field(
        default_factory=lambda: CitationsSettings(
            documents_response_field="documents",
            documents_request_param="documents",
            documents_stream_event_type="search-results",
            citations_response_field="citations",
            citations_stream_event_type="citation-generation",
        )
    )

    def write_response_to_history(
        self, memory: BaseMemory, raise_error: bool = False
    ) -> None:
        if self.chat_stream is None:
            raise ValueError(
                "chat_stream is None. Cannot write to history without chat_stream."
            )
        # try/except to prevent hanging on error
        try:
            final_text = ""
            for chat in self.chat_stream:
                # LLM response queue
                self.is_function = is_function(chat.message)
                self.put_in_queue(chat.delta)
                final_text += chat.delta or ""
                if chat.raw is not None:
                    # Citations stream event
                    if (
                        chat.raw.get("event_type", "")
                        == self.citations_settings.citations_stream_event_type
                    ):
                        self.citations += convert_chat_response_to_citations(
                            chat, self.citations_settings
                        )
                    # Documents stream event
                    if (
                        chat.raw.get("event_type", "")
                        == self.citations_settings.documents_stream_event_type
                    ):
                        self.documents += convert_chat_response_to_documents(
                            chat, self.citations_settings
                        )
            if self.is_function is not None:  # if loop has gone through iteration
                # NOTE: this is to handle the special case where we consume some of the
                # chat stream, but not all of it (e.g. in react agent)
                chat.message.content = final_text.strip()  # final message
                memory.put(chat.message)
        except Exception as e:
            if not raise_error:
                logger.warning(
                    f"Encountered exception writing response to history: {e}"
                )
            else:
                raise

        self.is_done = True

        # This act as is_done events for any consumers waiting
        self.is_function_not_none_thread_event.set()

    async def awrite_response_to_history(
        self,
        memory: BaseMemory,
    ) -> None:
        if self.achat_stream is None:
            raise ValueError(
                "achat_stream is None. Cannot asynchronously write to "
                "history without achat_stream."
            )
        # try/except to prevent hanging on error
        try:
            final_text = ""
            async for chat in self.achat_stream:
                # Chat response queue
                self.is_function = is_function(chat.message)
                self.aput_in_queue(chat.delta)
                final_text += chat.delta or ""
                if self.is_function is False:
                    self.is_function_false_event.set()
                if chat.raw is not None:
                    # Citations stream event
                    if (
                        chat.raw.get("event_type", "")
                        == self.citations_settings.citations_stream_event_type
                    ):
                        self.citations += convert_chat_response_to_citations(
                            chat, self.citations_settings
                        )
                    # Documents stream event
                    if (
                        chat.raw.get("event_type", "")
                        == self.citations_settings.documents_stream_event_type
                    ):
                        self.documents += convert_chat_response_to_documents(
                            chat, self.citations_settings
                        )
                self.new_item_event.set()
            if self.is_function is not None:  # if loop has gone through iteration
                # NOTE: this is to handle the special case where we consume some of the
                # chat stream, but not all of it (e.g. in react agent)
                chat.message.content = final_text.strip()  # final message
                memory.put(chat.message)
        except Exception as e:
            logger.warning(f"Encountered exception writing response to history: {e}")
        self.is_done = True


class ChatModeCitations(str, Enum):
    """Chat Engine Modes."""

    SIMPLE = "simple"
    """Corresponds to `SimpleChatEngine`.

    Chat with LLM, without making use of a knowledge base.
    """

    CONDENSE_QUESTION = "condense_question"
    """Corresponds to `CondenseQuestionChatEngine`.

    First generate a standalone question from conversation context and last message,
    then query the query engine for a response.
    """

    CONTEXT = "context"
    """Corresponds to `ContextChatEngine`.

    First retrieve text from the index using the user's message, then use the context
    in the system prompt to generate a response.
    """

    CONDENSE_PLUS_CONTEXT = "condense_plus_context"
    """Corresponds to `CondensePlusContextChatEngine`.

    First condense a conversation and latest user message to a standalone question.
    Then build a context for the standalone question from a retriever,
    Then pass the context along with prompt and user message to LLM to generate a response.
    """

    REACT = "react"
    """Corresponds to `ReActAgent`.

    Use a ReAct agent loop with query engine tools.
    """

    OPENAI = "openai"
    """Corresponds to `OpenAIAgent`.

    Use an OpenAI function calling agent loop.

    NOTE: only works with OpenAI models that support function calling API.
    """

    BEST = "best"
    """Select the best chat engine based on the current LLM.

    Corresponds to `OpenAIAgent` if using an OpenAI model that supports
    function calling API, otherwise, corresponds to `ReActAgent`.
    """

    CITATIONS_CONTEXT = "citations_context"
    """Corresponds to `CitationsContextChatEngine`.

    First retrieve text from the index using the user's message, then convert the context to
    the Citation's documents list. Then pass the context along with prompt and user message to LLM to generate
    a response with citations and related documents
    """

    COHERE_CITATIONS_CONTEXT = "cohere_citations_context"
    """Corresponds to `CitationsContextChatEngine`.

    First retrieve text from the index using the user's message, then convert the context to
    the Citation's documents list. Then pass the context along with prompt and user message to LLM to generate
    a response with citations and related documents
    """


class VectorStoreIndexWithCitationsChat(VectorStoreIndex):
    """Vector Store Index with Citations Chat."""

    def set_embed_model_input_type(self, input_type: str) -> None:
        try:
            from llama_index.embeddings.cohere import CohereEmbedding
        except ImportError:
            raise ImportError(
                "Please run `pip install llama-index-embeddings-cohere` "
                "to use the Cohere."
            )
        # Set the embed model input type. We need to change the Cohere embed input type to the 'search_query' value.
        if isinstance(self._embed_model, CohereEmbedding):
            self._embed_model.input_type = input_type

    def as_chat_engine(
        self,
        chat_mode: ChatModeCitations = ChatModeCitations.COHERE_CITATIONS_CONTEXT,
        llm: Optional[LLMType] = None,
        **kwargs: Any,
    ) -> BaseChatEngine:
        if chat_mode not in [ChatModeCitations.COHERE_CITATIONS_CONTEXT]:
            return super().as_chat_engine(chat_mode=chat_mode, llm=llm, **kwargs)
        else:
            llm = (
                resolve_llm(llm, callback_manager=self._callback_manager)
                if llm
                else Settings.llm
            )

            return CitationsContextChatEngine.from_defaults(
                retriever=self.as_retriever(**kwargs),
                llm=llm,
                **kwargs,
            )


class CitationsContextChatEngine(ContextChatEngine):
    """
    Cohere Context + Citations Chat Engine.

    Uses a retriever to retrieve a context, set the context in the Cohere
    documents param(https://docs.cohere.com/docs/retrieval-augmented-generation-rag),
    and then uses an LLM to generate a response.

    NOTE: this is made to be compatible with Cohere's chat model Document mode
    """

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentCitationsChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = self._generate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)

        all_messages = self._memory.get_all()
        # transform nodes to documents list
        documents_list = convert_nodes_to_documents_list(nodes)
        # prepare request kwargs
        citations_settings = CitationsSettings()
        kwargs = {}
        if citations_settings and citations_settings.documents_request_param:
            kwargs[citations_settings.documents_request_param] = documents_list
        # and then uses an LLM to generate a response
        chat_response = self._llm.chat(all_messages, **kwargs)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return AgentCitationsChatResponse(
            response=str(chat_response.message.content),
            citations=convert_chat_response_to_citations(
                chat_response, citations_settings
            ),
            documents=convert_chat_response_to_documents(
                chat_response, citations_settings
            ),
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
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentCitationsChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = self._generate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        all_messages = self._memory.get_all()
        documents_list = convert_nodes_to_documents_list(nodes)
        # prepare request kwargs
        citations_settings = CitationsSettings()
        kwargs = {}
        if citations_settings and citations_settings.documents_request_param:
            kwargs[citations_settings.documents_request_param] = documents_list
        # and then uses an LLM to generate a response
        chat_response = StreamingAgentCitationsChatResponse(
            chat_stream=self._llm.stream_chat(all_messages, **kwargs),
            citations_settings=citations_settings,
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
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentCitationsChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = await self._agenerate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        all_messages = self._memory.get_all()
        documents_list = convert_nodes_to_documents_list(nodes)
        # prepare request kwargs
        citations_settings = CitationsSettings()
        kwargs = {}
        if citations_settings and citations_settings.documents_request_param:
            kwargs[citations_settings.documents_request_param] = documents_list
        # and then uses an LLM to generate a response
        chat_response = await self._llm.achat(all_messages, **kwargs)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return AgentCitationsChatResponse(
            response=str(chat_response.message.content),
            citations=convert_chat_response_to_citations(
                chat_response, citations_settings
            ),
            documents=convert_chat_response_to_documents(
                chat_response, citations_settings
            ),
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
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentCitationsChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = await self._agenerate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)

        all_messages = self._memory.get_all()
        documents_list = convert_nodes_to_documents_list(nodes)
        # prepare request kwargs
        citations_settings = CitationsSettings()
        kwargs = {}
        if citations_settings and citations_settings.documents_request_param:
            kwargs[citations_settings.documents_request_param] = documents_list

        chat_response = StreamingAgentCitationsChatResponse(
            achat_stream=await self._llm.astream_chat(all_messages, **kwargs),
            citations_settings=citations_settings,
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
        loop = asyncio.get_event_loop()
        thread = Thread(
            target=lambda x: loop.create_task(
                chat_response.awrite_response_to_history(x)
            ),
            args=(self._memory,),
        )
        thread.start()

        return chat_response
