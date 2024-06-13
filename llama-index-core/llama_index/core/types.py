import threading
from abc import ABC, abstractmethod
from contextvars import copy_context
from enum import Enum
from functools import partial
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Generic,
    List,
    Type,
    TypeVar,
    Union,
)

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.instrumentation import DispatcherSpanMixin

Model = TypeVar("Model", bound=BaseModel)

TokenGen = Generator[str, None, None]
TokenAsyncGen = AsyncGenerator[str, None]
RESPONSE_TEXT_TYPE = Union[BaseModel, str, TokenGen, TokenAsyncGen]


# TODO: move into a `core` folder
# NOTE: this is necessary to make it compatible with pydantic
class BaseOutputParser(DispatcherSpanMixin, ABC):
    """Output parser class."""

    @classmethod
    def __modify_schema__(cls, schema: Dict[str, Any]) -> None:
        """Avoids serialization issues."""
        schema.update(type="object", default={})

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        return query

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


class BasePydanticProgram(DispatcherSpanMixin, ABC, Generic[Model]):
    """A base class for LLM-powered function that return a pydantic model.

    Note: this interface is not yet stable.
    """

    @property
    @abstractmethod
    def output_cls(self) -> Type[Model]:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Model:
        pass

    async def acall(self, *args: Any, **kwds: Any) -> Model:
        return self(*args, **kwds)


class PydanticProgramMode(str, Enum):
    """Pydantic program mode."""

    DEFAULT = "default"
    OPENAI = "openai"
    LLM = "llm"
    FUNCTION = "function"
    GUIDANCE = "guidance"
    LM_FORMAT_ENFORCER = "lm-format-enforcer"


class Thread(threading.Thread):
    """
    A wrapper for threading.Thread that copies the current context and uses the copy to run the target.
    """

    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None
    ) -> None:
        super().__init__(
            group=group,
            target=copy_context().run,
            name=name,
            args=(
                partial(target, *args, **(kwargs if isinstance(kwargs, dict) else {})),
            ),
            daemon=daemon,
        )
