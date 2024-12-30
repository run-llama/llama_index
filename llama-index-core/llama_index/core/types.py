import threading
from abc import ABC, abstractmethod
from contextvars import copy_context
from enum import Enum
from functools import partial
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from llama_index.core.bridge.pydantic_core import CoreSchema, core_schema
from llama_index.core.instrumentation import DispatcherSpanMixin

Model = TypeVar("Model", bound=BaseModel)

TokenGen = Generator[str, None, None]
TokenAsyncGen = AsyncGenerator[str, None]
RESPONSE_TEXT_TYPE = Union[BaseModel, str, TokenGen, TokenAsyncGen]


# TODO: move into a `core` folder
# NOTE: this is necessary to make it compatible with pydantic
class BaseOutputParser(DispatcherSpanMixin, ABC):
    """Output parser class."""

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
                message_content = messages[0].content or ""
                messages[0].content = self.format(message_content)
            else:
                message_content = messages[-1].content or ""
                messages[-1].content = self.format(message_content)

        return messages

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.any_schema()

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> Dict[str, Any]:
        json_schema = handler(core_schema)
        return handler.resolve_ref_schema(json_schema)


class BasePydanticProgram(DispatcherSpanMixin, ABC, Generic[Model]):
    """A base class for LLM-powered function that return a pydantic model.

    Note: this interface is not yet stable.
    """

    @property
    @abstractmethod
    def output_cls(self) -> Type[Model]:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Model:
        pass

    async def acall(self, *args: Any, **kwargs: Any) -> Model:
        return self(*args, **kwargs)

    def stream_call(
        self, *args: Any, **kwargs: Any
    ) -> Generator[Union[Model, List[Model]], None, None]:
        raise NotImplementedError("stream_call is not supported by default.")

    async def astream_call(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Union[Model, List[Model]], None]:
        raise NotImplementedError("astream_call is not supported by default.")


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
        self,
        group: Optional[Any] = None,
        target: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        *,
        daemon: Optional[bool] = None
    ) -> None:
        if target is not None:
            args = (
                partial(target, *args, **(kwargs if isinstance(kwargs, dict) else {})),
            )
        else:
            args = ()

        super().__init__(
            group=group,
            target=copy_context().run,
            name=name,
            args=args,
            daemon=daemon,
        )
