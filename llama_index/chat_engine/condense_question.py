import logging
from typing import List, Optional, Tuple

from langchain import LLMChain, OpenAI
from langchain.chains.conversational_retrieval.prompts import \
    CONDENSE_QUESTION_PROMPT

from llama_index.chat_engine.base import BaseChatEngine
from llama_index.chat_engine.utils import get_chat_history
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.prompts.base import Prompt
from llama_index.response.schema import RESPONSE_TYPE

logger = logging.getLogger(__name__)


class QueryChatEngine(BaseChatEngine):
    def __init__(
        self,
        query_engine: BaseQueryEngine,
        question_generator: LLMChain,
        chat_history: List[Tuple[str, str]] = None,
    ) -> None:
        self._chat_history = chat_history or []
        self._query_engine = query_engine
        self._question_generator = question_generator

    @classmethod
    def from_defaults(
        cls,
        query_engine: BaseQueryEngine,
        question_generation_prompt: Optional[Prompt] = None,
        chat_history: List[Tuple[str, str]] = None,
    ):
        prompt = question_generation_prompt or CONDENSE_QUESTION_PROMPT
        question_generator = question_generator or LLMChain(
            llm=OpenAI(temperature=0), prompt=prompt
        )
        chat_history = chat_history or []
        return cls(question_generator, query_engine, chat_history)

    def _condense_question(self, chat_history: List[str], last_message: str) -> str:
        """
        Generate standalone question from conversation context and last message
        """
        question_generator = LLMChain(
            llm=OpenAI(temperature=0), prompt=CONDENSE_QUESTION_PROMPT
        )

        chat_history_str = get_chat_history(chat_history)
        logger.debug(chat_history_str)
        new_question = question_generator.run(
            question=last_message, chat_history=chat_history_str
        )

        return new_question

    def chat(self, message: str) -> RESPONSE_TYPE:
        # Get chat history
        chat_history = self._chat_history

        # Generate standalone question from conversation context and last message
        condensed_question = self._condense_question(chat_history, message)

        # Query with standalone question
        logger.info(f"Querying with: {condensed_question}")
        response = self._query_engine.query(condensed_question)

        # Record response
        chat_history.append((message, str(response)))
        return response

    async def achat(self, message: str) -> RESPONSE_TYPE:
        return await super().achat(message)()
