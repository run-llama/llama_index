import logging
from typing import Any, Optional

from llama_index.chat_engine.types import BaseChatEngine
from llama_index.chat_engine.utils import get_user_chat_history
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
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
        chat_history: dict,
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
        chat_history: dict = {"default":""},
        service_context: Optional[ServiceContext] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "CondenseQuestionChatEngine":
        """Initialize a CondenseQuestionChatEngine from default parameters."""
        condense_question_prompt = condense_question_prompt or DEFAULT_PROMPT
        chat_history = chat_history
        service_context = service_context or ServiceContext.from_defaults()

        return cls(
            query_engine,
            condense_question_prompt,
            chat_history,
            service_context,
            verbose=verbose
        )

    def _condense_question(
        self, user_chat_history: str, last_message: str
    ) -> str:
        """
        Generate standalone question from conversation context and last message.
        """

        logger.debug(user_chat_history)

        response, _ = self._service_context.llm_predictor.predict(
            self._condense_question_prompt,
            question=last_message,
            chat_history=user_chat_history,
        )
        return response

    async def _acondense_question(
        self, user_chat_history: str, last_message: str
    ) -> str:
        """
        Generate standalone question from conversation context and last message.
        """

        logger.debug(user_chat_history)

        response, _ = await self._service_context.llm_predictor.apredict(
            self._condense_question_prompt,
            question=last_message,
            chat_history=user_chat_history,
        )
        return response

    def chat(self, message: str, user_id: str = None) -> RESPONSE_TYPE:

        # Retrieves specific chat history
        user_chat_history = get_user_chat_history(self._chat_history, user_id)

        # Generate standalone question from conversation context and last message
        condensed_question = self._condense_question(user_chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # Query with standalone question
        response = self._query_engine.query(condensed_question)

        # Adding new user query and response
        user_chat_history = user_chat_history + '\nHuman: ' + message + ' \nAssistant: ' + str(response)

        # Updating chat history
        if user_id:
            self._chat_history[user_id] = user_chat_history
        else:
            self._chat_history["default"] = user_chat_history
            
        return response

    async def achat(self, message: str, user_id: str = '') -> RESPONSE_TYPE:
        # Retrieves specific chat history
        user_chat_history = get_user_chat_history(self._chat_history, user_id)

        # Generate standalone question from conversation context and last message
        condensed_question = await self._acondense_question(user_chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # Query with standalone question
        response = await self._query_engine.aquery(condensed_question)

        # Adding new user query and response
        user_chat_history = user_chat_history + '\nHuman: ' + message + '\nAssistant: ' + str(response)

        # Updating chat history
        if user_id != '':
            self._chat_history[user_id] = user_chat_history
        else:
            self._chat_history["default"] = user_chat_history

        return response

    def reset(self, user_id: str = '', reset_all: bool = False) -> None:
        # Clear chat history for particular user_id or default chat history
        # from the dict. If reset_all is set to True, delete every conversation. 
        if reset_all:
            self._chat_history = {"default":""}
        
        elif user_id != '' and self._chat_history.get(user_id):
            del self._chat_history[user_id]
        
        else:
            self._chat_history["default"] = ""
        
