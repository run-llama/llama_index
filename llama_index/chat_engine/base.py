import logging
from abc import ABC, abstractmethod

from llama_index.response.schema import RESPONSE_TYPE

logger = logging.getLogger(__name__)


class BaseChatEngine(ABC):
    @abstractmethod
    def chat(self, message: str) -> RESPONSE_TYPE:
        pass

    @abstractmethod
    async def achat(self, message: str) -> RESPONSE_TYPE:
        pass
