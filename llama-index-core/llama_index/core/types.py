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
    """
    Base class for output parsers that process and structure LLM responses.
    
    This abstract class defines the interface for parsing raw LLM outputs into structured formats.
    Implementations should handle validation, error correction, and conversion to specific data types.
    
    Attributes:
        format (method): Format a query with structured output formatting instructions.
        format_messages (method): Format a list of messages with structured output formatting instructions.
        parse (method): Parse and validate the raw LLM output string into a structured format.
    """

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse and validate the raw LLM output string into a structured format.
        
        Args:
            output (str): The raw string output from the LLM to parse.
            
        Returns:
            Any: The parsed and structured output in the target format.
            
        Raises:
            ValueError: If the output cannot be parsed into the expected format.
        """

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions.
        
        Args:
            query (str): The input query to format.
            
        Returns:
            str: The formatted query with output instructions.
        """
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
    A base class for LLM-powered functions that return a Pydantic model.
    
    This class provides a structured way to convert LLM outputs into strongly-typed
    Pydantic models, enabling type safety and validation. It serves as a bridge
    between unstructured LLM responses and structured application data.

    Type Parameters:
        Model: The Pydantic model type that this program will output
        
    Attributes:
        output_cls (property): The Pydantic model class used for output validation.
        __call__ (method): Execute the program and return a validated model instance.
        acall (method): Async version of __call__.
        stream_call (method): Stream output as model instances.
        astream_call (method): Async version of stream_call.
        
    Note: 
        This interface is not yet stable and may change in future versions.
    """

    @property
    @abstractmethod
    def output_cls(self) -> Type[Model]:
        """Get the output Pydantic model class.
        
        Returns:
            Type[Model]: The Pydantic model class used for output validation.
        """
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Union[Model, List[Model]]:
        """Execute the program and return a validated model instance.
        
        Returns:
            Union[Model, List[Model]]: A single model instance or list of instances.
        """
        pass

    async def acall(self, *args: Any, **kwargs: Any) -> Union[Model, List[Model]]:
        """Async version of __call__.
        
        Returns:
            Union[Model, List[Model]]: A single model instance or list of instances.
        """
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
    A context-aware Thread wrapper that preserves the execution context.
    
    This class extends the standard threading.Thread to ensure that any context variables
    from the parent thread are properly propagated to the child thread. This is particularly
    useful for maintaining context in asynchronous operations, logging, and request tracking.
    
    The wrapper automatically copies the current context when the thread is created and
    ensures this context is active when the target function runs.

    Args:
        group (Optional[Any]): The thread group (unused, kept for compatibility)
        target (Optional[Callable]): The callable object to be invoked by the run() method
        name (Optional[str]): The thread name
        args (Tuple[Any, ...]): The argument tuple for target invocation
        kwargs (Optional[Dict]): A dictionary of keyword arguments for the target
        daemon (Optional[bool]): The thread's daemon flag
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
        # Wrap the target function with its arguments
        if target is not None:
            args = (
                partial(target, *args, **(kwargs if isinstance(kwargs, dict) else {})),
            )
        else:
            args = ()

        # Initialize with context copying
        super().__init__(
            group=group,
            target=copy_context().run,
            name=name,
            args=args,
            daemon=daemon,
        )
