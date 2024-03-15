import asyncio
import logging
from threading import Thread
from typing import Any, List, Optional, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.indices.base_retriever import BaseRetriever
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.base.llms.generic_utils import messages_to_history_str
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.utilities.token_counting import TokenCounter

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_PROMPT_TEMPLATE = """
  The following is a friendly conversation between a user and an AI assistant.
  The assistant is talkative and provides lots of specific details from its context.
  If the assistant does not know the answer to a question, it truthfully says it
  does not know.

  Here are the relevant documents for the context:

  {context_str}

  Instruction: Based on the above documents, provide a detailed answer for the user question below.
  Answer "don't know" if not present in the document.
  """

DEFAULT_CONDENSE_PROMPT_TEMPLATE = """
  Given the following conversation between a user and an AI assistant and a follow up question from user,
  rephrase the follow up question to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:"""


class CondensePlusContextChatEngine(BaseChatEngine):
    """Condensed Conversation & Context Chat Engine.

    First condense a conversation and latest user message to a standalone question
    Then build a context for the standalone question from a retriever,
    Then pass the context along with prompt and user message to LLM to generate a response.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        context_prompt: Optional[str] = None,
        condense_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        skip_condense: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ):
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._context_prompt_template = (
            context_prompt or DEFAULT_CONTEXT_PROMPT_TEMPLATE
        )
        condense_prompt_str = condense_prompt or DEFAULT_CONDENSE_PROMPT_TEMPLATE
        self._condense_prompt_template = PromptTemplate(condense_prompt_str)
        self._system_prompt = system_prompt
        self._skip_condense = skip_condense
        self._node_postprocessors = node_postprocessors or []
        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager

        self._token_counter = TokenCounter()
        self._verbose = verbose

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        service_context: Optional[ServiceContext] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        context_prompt: Optional[str] = None,
        condense_prompt: Optional[str] = None,
        skip_condense: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "CondensePlusContextChatEngine":
        """Initialize a CondensePlusContextChatEngine from default parameters."""
        llm = llm or llm_from_settings_or_context(Settings, service_context)

        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=llm.metadata.context_window - 256
        )

        return cls(
            retriever=retriever,
            llm=llm,
            memory=memory,
            context_prompt=context_prompt,
            condense_prompt=condense_prompt,
            skip_condense=skip_condense,
            callback_manager=callback_manager_from_settings_or_context(
                Settings, service_context
            ),
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

        return self._llm.predict(
            self._condense_prompt_template,
            question=latest_message,
            chat_history=chat_history_str,
        )

    async def _acondense_question(
        self, chat_history: List[ChatMessage], latest_message: str
    ) -> str:
        """Condense a conversation history and latest user message to a standalone question."""
        if self._skip_condense or len(chat_history) == 0:
            return latest_message

        chat_history_str = messages_to_history_str(chat_history)
        logger.debug(chat_history_str)

        return await self._llm.apredict(
            self._condense_prompt_template,
            question=latest_message,
            chat_history=chat_history_str,
        )

    def _retrieve_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Build context for a message from retriever."""
        nodes = self._retriever.retrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes]
        )
        return context_str, nodes

    async def _aretrieve_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Build context for a message from retriever."""
        nodes = await self._retriever.aretrieve(message)
        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes]
        )
        return context_str, nodes

    def _run_c3(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Tuple[List[ChatMessage], ToolOutput, List[NodeWithScore]]:
        if chat_history is not None:
            self._memory.set(chat_history)

        chat_history = self._memory.get()

        # Condense conversation history and latest message to a standalone question
        condensed_question = self._condense_question(chat_history, message)  # type: ignore
        logger.info(f"Condensed question: {condensed_question}")
        if self._verbose:
            print(f"Condensed question: {condensed_question}")

        # Build context for the standalone question from a retriever
        context_str, context_nodes = self._retrieve_context(condensed_question)
        context_source = ToolOutput(
            tool_name="retriever",
            content=context_str,
            raw_input={"message": condensed_question},
            raw_output=context_str,
        )
        logger.debug(f"Context: {context_str}")
        if self._verbose:
            print(f"Context: {context_str}")

        system_message_content = self._context_prompt_template.format(
            context_str=context_str
        )
        if self._system_prompt:
            system_message_content = self._system_prompt + "\n" + system_message_content

        system_message = ChatMessage(
            content=system_message_content, role=self._llm.metadata.system_role
        )

        initial_token_count = self._token_counter.estimate_tokens_in_messages(
            [system_message]
        )

        self._memory.put(ChatMessage(content=message, role=MessageRole.USER))
        chat_messages = [
            system_message,
            *self._memory.get(initial_token_count=initial_token_count),
        ]
        return chat_messages, context_source, context_nodes

    async def _arun_c3(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Tuple[List[ChatMessage], ToolOutput, List[NodeWithScore]]:
        if chat_history is not None:
            self._memory.set(chat_history)

        chat_history = self._memory.get()

        # Condense conversation history and latest message to a standalone question
        condensed_question = await self._acondense_question(chat_history, message)  # type: ignore
        logger.info(f"Condensed question: {condensed_question}")
        if self._verbose:
            print(f"Condensed question: {condensed_question}")

        # Build context for the standalone question from a retriever
        context_str, context_nodes = await self._aretrieve_context(condensed_question)
        context_source = ToolOutput(
            tool_name="retriever",
            content=context_str,
            raw_input={"message": condensed_question},
            raw_output=context_str,
        )
        logger.debug(f"Context: {context_str}")
        if self._verbose:
            print(f"Context: {context_str}")

        system_message_content = self._context_prompt_template.format(
            context_str=context_str
        )
        if self._system_prompt:
            system_message_content = self._system_prompt + "\n" + system_message_content

        system_message = ChatMessage(
            content=system_message_content, role=self._llm.metadata.system_role
        )

        initial_token_count = self._token_counter.estimate_tokens_in_messages(
            [system_message]
        )

        self._memory.put(ChatMessage(content=message, role=MessageRole.USER))
        chat_messages = [
            system_message,
            *self._memory.get(initial_token_count=initial_token_count),
        ]

        return chat_messages, context_source, context_nodes

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        chat_messages, context_source, context_nodes = self._run_c3(
            message, chat_history
        )

        # pass the context, system prompt and user message as chat to LLM to generate a response
        chat_response = self._llm.chat(chat_messages)
        assistant_message = chat_response.message
        self._memory.put(assistant_message)

        return AgentChatResponse(
            response=str(assistant_message.content),
            sources=[context_source],
            source_nodes=context_nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_messages, context_source, context_nodes = self._run_c3(
            message, chat_history
        )

        # pass the context, system prompt and user message as chat to LLM to generate a response
        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(chat_messages),
            sources=[context_source],
            source_nodes=context_nodes,
        )
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return chat_response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        chat_messages, context_source, context_nodes = await self._arun_c3(
            message, chat_history
        )

        # pass the context, system prompt and user message as chat to LLM to generate a response
        chat_response = await self._llm.achat(chat_messages)
        assistant_message = chat_response.message
        self._memory.put(assistant_message)

        return AgentChatResponse(
            response=str(assistant_message.content),
            sources=[context_source],
            source_nodes=context_nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_messages, context_source, context_nodes = await self._arun_c3(
            message, chat_history
        )

        # pass the context, system prompt and user message as chat to LLM to generate a response
        chat_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(chat_messages),
            sources=[context_source],
            source_nodes=context_nodes,
        )
        thread = Thread(
            target=lambda x: asyncio.run(chat_response.awrite_response_to_history(x)),
            args=(self._memory,),
        )
        thread.start()

        return chat_response

    def reset(self) -> None:
        # Clear chat history
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
