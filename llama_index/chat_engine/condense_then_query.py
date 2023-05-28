import logging
from typing import List, Optional, Tuple

from llama_index.chat_engine.types import BaseChatEngine
from llama_index.chat_engine.utils import get_chat_history
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.base import Prompt
from llama_index.response.schema import RESPONSE_TYPE

logger = logging.getLogger(__name__)


DEFAULT_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History: {chat_history}

Follow Up Input: {question}

Standalone question:"""

DEFAULT_PROMPT = Prompt(DEFAULT_TEMPLATE)


class CondenseQuestionChatEngine(BaseChatEngine):
    def __init__(
        self,
        query_engine: BaseQueryEngine,
        condense_question_prompt: Optional[str] = None,
        chat_history: List[Tuple[str, str]] = None,
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        self._query_engine = query_engine
        self._condense_question_prompt = condense_question_prompt or DEFAULT_PROMPT
        self._chat_history = chat_history or []
        self._service_context = service_context or ServiceContext.from_defaults()

    def _condense_question(self, chat_history: List[str], last_message: str) -> str:
        """
        Generate standalone question from conversation context and last message
        """

        chat_history_str = get_chat_history(chat_history)
        logger.debug(chat_history_str)

        return self._service_context.llm_predictor.predict(
            self._condense_question_prompt,
            question=last_message,
            chat_history=chat_history_str,
        )

    def chat(self, message: str) -> RESPONSE_TYPE:
        # Generate standalone question from conversation context and last message
        condensed_question = self._condense_question(self._chat_history, message)

        # Query with standalone question
        logger.info(f"Querying with: {condensed_question}")
        response = self._query_engine.query(condensed_question)

        # Record response
        self._chat_history.append((message, str(response)))
        return response

    async def achat(self, message: str) -> RESPONSE_TYPE:
        return self.chat(message)
