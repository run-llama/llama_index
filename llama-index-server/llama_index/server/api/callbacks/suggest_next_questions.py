import logging
from typing import Any, Optional

from llama_index.server.api.callbacks.base import EventCallback
from llama_index.server.api.models import ChatRequest
from llama_index.server.services.suggest_next_question import (
    SuggestNextQuestionsService,
)

logger = logging.getLogger("uvicorn")


class SuggestNextQuestions(EventCallback):
    """Processor for generating next question suggestions."""

    def __init__(
        self, chat_request: ChatRequest, logger: Optional[logging.Logger] = None
    ):
        self.chat_request = chat_request
        self.accumulated_text = ""
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("uvicorn")

    async def on_complete(self, final_response: str) -> Any:
        if final_response == "":
            self.logger.warning(
                "SuggestNextQuestions is enabled but final response is empty, make sure your content generator accumulates text"
            )
            return None

        questions = await SuggestNextQuestionsService.run(
            self.chat_request.messages, final_response
        )
        if questions:
            return {
                "type": "suggested_questions",
                "data": questions,
            }
        return None

    @classmethod
    def from_default(cls, chat_request: ChatRequest) -> "SuggestNextQuestions":
        return cls(chat_request=chat_request)
