import logging
from abc import ABC, abstractmethod
from enum import Enum

from llama_index.response.schema import RESPONSE_TYPE

logger = logging.getLogger(__name__)


class BaseChatEngine(ABC):
    @abstractmethod
    def chat(self, message: str) -> RESPONSE_TYPE:
        pass

    @abstractmethod
    async def achat(self, message: str) -> RESPONSE_TYPE:
        pass


class ChatMode(str, Enum):
    CONDENSE_QUESTION =  "condense_question"
