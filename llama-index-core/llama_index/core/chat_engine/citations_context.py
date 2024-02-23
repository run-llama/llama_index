import asyncio
from threading import Thread
from typing import List, Optional

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import trace_method
from llama_index.core.chat_engine.types import (
    AgentCitationsChatResponse,
    StreamingAgentCitationsChatResponse,
    ToolOutput,
)
from llama_index.core.chat_engine.utils import (
    convert_nodes_to_documents_list,
    convert_chat_response_to_citations,
    convert_chat_response_to_documents,
)

from llama_index.core.chat_engine.context import ContextChatEngine


class CitationsContextChatEngine(ContextChatEngine):
    """Cohere Context + Citations Chat Engine.

    Uses a retriever to retrieve a context, set the context in the Cohere
    documents param(https://docs.cohere.com/docs/retrieval-augmented-generation-rag),
    and then uses an LLM to generate a response.

    NOTE: this is made to be compatible with Cohere's chat model Document mode
    """

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentCitationsChatResponse:
        if not self._llm.metadata.has_chat_citation_mode:
            raise ValueError("LLM does not support citations mode!")
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = self._generate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)

        all_messages = self._memory.get_all()
        # transform nodes to documents list
        documents_list = convert_nodes_to_documents_list(nodes)
        # prepare request kwargs
        citations_settings = self._llm.metadata.citations_settings
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
        if not self._llm.metadata.has_chat_citation_mode:
            raise ValueError("LLM does not support citations mode!")
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = self._generate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        all_messages = self._memory.get_all()
        documents_list = convert_nodes_to_documents_list(nodes)
        # prepare request kwargs
        citations_settings = self._llm.metadata.citations_settings
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
        if not self._llm.metadata.has_chat_citation_mode:
            raise ValueError("LLM does not support citations mode!")
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = await self._agenerate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        all_messages = self._memory.get_all()
        documents_list = convert_nodes_to_documents_list(nodes)
        # prepare request kwargs
        citations_settings = self._llm.metadata.citations_settings
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
        if not self._llm.metadata.has_chat_citation_mode:
            raise ValueError("LLM does not support citations mode!")
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = await self._agenerate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)

        all_messages = self._memory.get_all()
        documents_list = convert_nodes_to_documents_list(nodes)
        # prepare request kwargs
        citations_settings = self._llm.metadata.citations_settings
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

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
