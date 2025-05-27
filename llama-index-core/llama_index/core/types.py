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
    TYPE_CHECKING,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
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

if TYPE_CHECKING:
    from llama_index.core.program.utils import FlexibleModel


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

    def _format_message(self, message: ChatMessage) -> ChatMessage:
        text_blocks: list[tuple[int, TextBlock]] = [
            (idx, block)
            for idx, block in enumerate(message.blocks)
            if isinstance(block, TextBlock)
        ]

        # add text to the last text block, or add a new text block
        format_text = ""
        if text_blocks:
            format_idx = text_blocks[-1][0]
            format_text = text_blocks[-1][1].text

            if format_idx != -1:
                # this should always be a text block
                assert isinstance(message.blocks[format_idx], TextBlock)
                message.blocks[format_idx].text = self.format(format_text)  # type: ignore
        else:
            message.blocks.append(TextBlock(text=self.format(format_text)))

        return message

    def format_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Format a list of messages with structured output formatting instructions."""
        # NOTE: apply output parser to either the first message if it's a system message
        #       or the last message
        if messages:
            if messages[0].role == MessageRole.SYSTEM:
                # get text from the last text blocks
                messages[0] = self._format_message(messages[0])
            else:
                messages[-1] = self._format_message(messages[-1])

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
    """
    A base class for LLM-powered function that return a pydantic model.

    Note: this interface is not yet stable.
    """

    @property
    @abstractmethod
    def output_cls(self) -> Type[Model]:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Union[Model, List[Model]]:
        pass

    async def acall(self, *args: Any, **kwargs: Any) -> Union[Model, List[Model]]:
        return self(*args, **kwargs)

    def stream_call(
        self, *args: Any, **kwargs: Any
    ) -> Generator[
        Union[Model, List[Model], "FlexibleModel", List["FlexibleModel"]], None, None
    ]:
        raise NotImplementedError("stream_call is not supported by default.")

    async def astream_call(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[
        Union[Model, List[Model], "FlexibleModel", List["FlexibleModel"]], None
    ]:
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
        daemon: Optional[bool] = None,
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
