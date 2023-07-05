import logging
from typing import Any, List, Optional

from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.generic_utils import messages_to_history_str
from llama_index.prompts.base import Prompt
from llama_index.response.schema import RESPONSE_TYPE

logger = logging.getLogger(__name__)


DEFAULT_TEMPLATE = """\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History> 
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""

DEFAULT_PROMPT = Prompt(DEFAULT_TEMPLATE)


class CondenseQuestionChatEngine(BaseChatEngine):
    """Condense Question Chat Engine.

    First generate a standalone question from conversation context and last message,
    then query the query engine for a response.
    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        condense_question_prompt: Prompt,
        chat_history: List[ChatMessage],
        service_context: ServiceContext,
        verbose: bool = False,
    ) -> None:
        self._query_engine = query_engine
        self._condense_question_prompt = condense_question_prompt
        self._chat_history = chat_history
        self._service_context = service_context
        self._verbose = verbose

    @classmethod
    def from_defaults(
        cls,
        query_engine: BaseQueryEngine,
        condense_question_prompt: Optional[Prompt] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        service_context: Optional[ServiceContext] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "CondenseQuestionChatEngine":
        """Initialize a CondenseQuestionChatEngine from default parameters."""
        condense_question_prompt = condense_question_prompt or DEFAULT_PROMPT
        chat_history = chat_history or []
        service_context = service_context or ServiceContext.get_global() or ServiceContext.from_defaults()

        return cls(
            query_engine,
            condense_question_prompt,
            chat_history,
            service_context,
            verbose=verbose,
        )

    def _condense_question(
        self, chat_history: List[ChatMessage], last_message: str
    ) -> str:
        """
        Generate standalone question from conversation context and last message.
        """

        chat_history_str = messages_to_history_str(chat_history)
        logger.debug(chat_history_str)

        response = self._service_context.llm_predictor.predict(
            self._condense_question_prompt,
            question=last_message,
            chat_history=chat_history_str,
        )
        return response

    async def _acondense_question(
        self, chat_history: List[ChatMessage], last_message: str
    ) -> str:
        """
        Generate standalone question from conversation context and last message.
        """

        chat_history_str = messages_to_history_str(chat_history)
        logger.debug(chat_history_str)

        response = await self._service_context.llm_predictor.apredict(
            self._condense_question_prompt,
            question=last_message,
            chat_history=chat_history_str,
        )
        return response

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history

        # Generate standalone question from conversation context and last message
        condensed_question = self._condense_question(chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # Query with standalone question
        response = self._query_engine.query(condensed_question)

        # Record response
        chat_history.extend(
            [
                ChatMessage(role=MessageRole.USER, content=message),
                ChatMessage(role=MessageRole.ASSISTANT, content=str(response)),
            ]
        )
        return response

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history

        # Generate standalone question from conversation context and last message
        condensed_question = await self._acondense_question(chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # Query with standalone question
        response = await self._query_engine.aquery(condensed_question)

        # Record response
        chat_history.extend(
            [
                ChatMessage(role=MessageRole.USER, content=message),
                ChatMessage(role=MessageRole.ASSISTANT, content=str(response)),
            ]
        )
        return response

    def reset(self) -> None:
        # Clear chat history
        self._chat_history = []

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history as human and ai message pairs."""
        return self._chat_history
