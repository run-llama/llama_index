import logging
from typing import Any, List, Optional, Tuple, Union

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.base.response.schema import (
    AsyncStreamingResponse,
    StreamingResponse,
)
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.indices.base_retriever import BaseRetriever
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.base.llms.generic_utils import messages_to_history_str
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore
from llama_index.core.settings import Settings
from llama_index.core.utilities.token_counting import TokenCounter
from llama_index.core.chat_engine.utils import (
    get_prefix_messages_with_context,
    get_response_synthesizer,
)

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

DEFAULT_CONTEXT_REFINE_PROMPT_TEMPLATE = """
  The following is a friendly conversation between a user and an AI assistant.
  The assistant is talkative and provides lots of specific details from its context.
  If the assistant does not know the answer to a question, it truthfully says it
  does not know.

  Here are the relevant documents for the context:

  {context_msg}

  Existing Answer:
  {existing_answer}

  Instruction: Refine the existing answer using the provided context to assist the user.
  If the context isn't helpful, just repeat the existing answer and nothing more.
  """

DEFAULT_CONDENSE_PROMPT_TEMPLATE = """
  Given the following conversation between a user and an AI assistant and a follow up question from user,
  rephrase the follow up question to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:"""


class CondensePlusContextChatEngine(BaseChatEngine):
    """
    Condensed Conversation & Context Chat Engine.

    First condense a conversation and latest user message to a standalone question
    Then build a context for the standalone question from a retriever,
    Then pass the context along with prompt and user message to LLM to generate a response.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        context_prompt: Optional[Union[str, PromptTemplate]] = None,
        context_refine_prompt: Optional[Union[str, PromptTemplate]] = None,
        condense_prompt: Optional[Union[str, PromptTemplate]] = None,
        system_prompt: Optional[str] = None,
        skip_condense: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ):
        self._retriever = retriever
        self._llm = llm
        self._memory = memory

        context_prompt = context_prompt or DEFAULT_CONTEXT_PROMPT_TEMPLATE
        if isinstance(context_prompt, str):
            context_prompt = PromptTemplate(context_prompt)
        self._context_prompt_template = context_prompt

        context_refine_prompt = (
            context_refine_prompt or DEFAULT_CONTEXT_REFINE_PROMPT_TEMPLATE
        )
        if isinstance(context_refine_prompt, str):
            context_refine_prompt = PromptTemplate(context_refine_prompt)
        self._context_refine_prompt_template = context_refine_prompt

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

        self._token_counter = TokenCounter()
        self._verbose = verbose

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        context_prompt: Optional[Union[str, PromptTemplate]] = None,
        context_refine_prompt: Optional[Union[str, PromptTemplate]] = None,
        condense_prompt: Optional[Union[str, PromptTemplate]] = None,
        skip_condense: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "CondensePlusContextChatEngine":
        """Initialize a CondensePlusContextChatEngine from default parameters."""
        llm = llm or Settings.llm

        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=llm.metadata.context_window - 256
        )

        return cls(
            retriever=retriever,
            llm=llm,
            memory=memory,
            context_prompt=context_prompt,
            context_refine_prompt=context_refine_prompt,
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

        return str(self._llm.complete(llm_input))

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

        return str(await self._llm.acomplete(llm_input))

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

    def _get_response_synthesizer(
        self, chat_history: List[ChatMessage], streaming: bool = False
    ) -> CompactAndRefine:
        system_prompt = self._system_prompt or ""
        qa_messages = get_prefix_messages_with_context(
            self._context_prompt_template,
            system_prompt,
            [],
            chat_history,
            self._llm.metadata.system_role,
        )
        refine_messages = get_prefix_messages_with_context(
            self._context_refine_prompt_template,
            system_prompt,
            [],
            chat_history,
            self._llm.metadata.system_role,
        )

        return get_response_synthesizer(
            self._llm,
            self.callback_manager,
            qa_messages,
            refine_messages,
            streaming,
            qa_function_mappings=self._context_prompt_template.function_mappings,
            refine_function_mappings=self._context_refine_prompt_template.function_mappings,
        )

    def _run_c3(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        streaming: bool = False,
    ) -> Tuple[CompactAndRefine, ToolOutput, List[NodeWithScore]]:
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

        # build the response synthesizer
        response_synthesizer = self._get_response_synthesizer(
            chat_history, streaming=streaming
        )

        return response_synthesizer, context_source, context_nodes

    async def _arun_c3(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        streaming: bool = False,
    ) -> Tuple[CompactAndRefine, ToolOutput, List[NodeWithScore]]:
        if chat_history is not None:
            self._memory.set(chat_history)

        chat_history = self._memory.get(input=message)

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

        # build the response synthesizer
        response_synthesizer = self._get_response_synthesizer(
            chat_history, streaming=streaming
        )

        return response_synthesizer, context_source, context_nodes

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        synthesizer, context_source, context_nodes = self._run_c3(message, chat_history)

        response = synthesizer.synthesize(message, context_nodes)

        user_message = ChatMessage(content=message, role=MessageRole.USER)
        assistant_message = ChatMessage(
            content=str(response), role=MessageRole.ASSISTANT
        )
        self._memory.put(user_message)
        self._memory.put(assistant_message)

        return AgentChatResponse(
            response=str(response),
            sources=[context_source],
            source_nodes=context_nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        synthesizer, context_source, context_nodes = self._run_c3(
            message, chat_history, streaming=True
        )

        response = synthesizer.synthesize(message, context_nodes)
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

            user_message = ChatMessage(content=message, role=MessageRole.USER)
            assistant_message = ChatMessage(
                content=full_response, role=MessageRole.ASSISTANT
            )
            self._memory.put(user_message)
            self._memory.put(assistant_message)

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
        synthesizer, context_source, context_nodes = await self._arun_c3(
            message, chat_history
        )

        response = await synthesizer.asynthesize(message, context_nodes)

        user_message = ChatMessage(content=message, role=MessageRole.USER)
        assistant_message = ChatMessage(
            content=str(response), role=MessageRole.ASSISTANT
        )
        await self._memory.aput(user_message)
        await self._memory.aput(assistant_message)

        return AgentChatResponse(
            response=str(response),
            sources=[context_source],
            source_nodes=context_nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        synthesizer, context_source, context_nodes = await self._arun_c3(
            message, chat_history, streaming=True
        )

        response = await synthesizer.asynthesize(message, context_nodes)
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
