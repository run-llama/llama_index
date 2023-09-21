from abc import abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    List,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.base import ChatMessage, MessageRole

Model = TypeVar("Model", bound=BaseModel)

TokenGen = Generator[str, None, None]
TokenAsyncGen = AsyncGenerator[str, None]
RESPONSE_TEXT_TYPE = Union[str, TokenGen]


# TODO: move into a `core` folder
# NOTE: this is necessary to make it compatible with pydantic
@runtime_checkable
class BaseOutputParser(Protocol):
    """Output parser class."""

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""

    @abstractmethod
    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""

    def format_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Format a list of messages with structured output formatting instructions."""
        # NOTE: apply output parser to either the first message if it's a system message
        #       or the last message
        if messages:
            if messages[0].role == MessageRole.SYSTEM:
                messages[0].content = self.format(messages[0].content or "")
            else:
                messages[-1].content = self.format(messages[-1].content or "")

        return messages
