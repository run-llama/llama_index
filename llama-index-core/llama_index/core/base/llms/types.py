import base64
import requests
from enum import Enum
from io import BytesIO
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Literal,
    Optional,
    Union,
    List,
    Any,
)

from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.schema import ImageType

try:
    from pydantic import BaseModel as V2BaseModel
    from pydantic.v1 import BaseModel as V1BaseModel
except ImportError:
    from pydantic import BaseModel as V2BaseModel

    V1BaseModel = V2BaseModel  # type: ignore


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"
    MODEL = "model"


# ===== Generic Model Input - Chat =====
class ContentBlockTypes(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class TextBlock(BaseModel):
    type: Literal[ContentBlockTypes.TEXT] = ContentBlockTypes.TEXT

    text: str


class ImageBlock(BaseModel):
    type: Literal[ContentBlockTypes.IMAGE] = ContentBlockTypes.IMAGE

    image: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    image_mimetype: Optional[str] = None

    def resolve_image(self) -> ImageType:
        """Resolve an image such that PIL can read it."""
        if self.image is not None:
            return BytesIO(base64.b64decode(self.image))
        elif self.image_path is not None:
            return self.image_path
        elif self.image_url is not None:
            # load image from URL
            response = requests.get(self.image_url)
            return BytesIO(response.content)
        else:
            raise ValueError("No image found in the chat message!")


class ChatMessage(BaseModel):
    """Chat message."""

    role: MessageRole = MessageRole.USER
    content: Optional[Any] = ""
    additional_kwargs: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"

    @classmethod
    def from_str(
        cls,
        content: str,
        role: Union[MessageRole, str] = MessageRole.USER,
        **kwargs: Any,
    ) -> "ChatMessage":
        if isinstance(role, str):
            role = MessageRole(role)
        return cls(role=role, content=content, **kwargs)

    def _recursive_serialization(self, value: Any) -> Any:
        if isinstance(value, (V1BaseModel, V2BaseModel)):
            return value.model_dump()  # type: ignore
        if isinstance(value, dict):
            return {
                key: self._recursive_serialization(value)
                for key, value in value.items()
            }
        if isinstance(value, list):
            return [self._recursive_serialization(item) for item in value]
        return value

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        return self.model_dump(**kwargs)

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        # ensure all additional_kwargs are serializable
        msg = super().model_dump(**kwargs)

        for key, value in msg.get("additional_kwargs", {}).items():
            value = self._recursive_serialization(value)
            if not isinstance(value, (str, int, float, bool, dict, list, type(None))):
                raise ValueError(
                    f"Failed to serialize additional_kwargs value: {value}"
                )
            msg["additional_kwargs"][key] = value

        return msg


class LogProb(BaseModel):
    """LogProb of a token."""

    token: str = Field(default_factory=str)
    logprob: float = Field(default_factory=float)
    bytes: List[int] = Field(default_factory=list)


# ===== Generic Model Output - Chat =====
class ChatResponse(BaseModel):
    """Chat response."""

    message: ChatMessage
    raw: Optional[Any] = None
    delta: Optional[str] = None
    logprobs: Optional[List[List[LogProb]]] = None
    additional_kwargs: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return str(self.message)


ChatResponseGen = Generator[ChatResponse, None, None]
ChatResponseAsyncGen = AsyncGenerator[ChatResponse, None]


# ===== Generic Model Output - Completion =====
class CompletionResponse(BaseModel):
    """
    Completion response.

    Fields:
        text: Text content of the response if not streaming, or if streaming,
            the current extent of streamed text.
        additional_kwargs: Additional information on the response(i.e. token
            counts, function calling information).
        raw: Optional raw JSON that was parsed to populate text, if relevant.
        delta: New text that just streamed in (only relevant when streaming).
    """

    text: str
    additional_kwargs: dict = Field(default_factory=dict)
    raw: Optional[Any] = None
    logprobs: Optional[List[List[LogProb]]] = None
    delta: Optional[str] = None

    def __str__(self) -> str:
        return self.text


CompletionResponseGen = Generator[CompletionResponse, None, None]
CompletionResponseAsyncGen = AsyncGenerator[CompletionResponse, None]


class LLMMetadata(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "Total number of tokens the model can be input and output for one response."
        ),
    )
    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of tokens the model can output when generating a response.",
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            "Set True if the model exposes a chat interface (i.e. can be passed a"
            " sequence of messages, rather than text), like OpenAI's"
            " /v1/chat/completions endpoint."
        ),
    )
    is_function_calling_model: bool = Field(
        default=False,
        # SEE: https://openai.com/blog/function-calling-and-other-api-updates
        description=(
            "Set True if the model supports function calling messages, similar to"
            " OpenAI's function calling API. For example, converting 'Email Anya to"
            " see if she wants to get coffee next Friday' to a function call like"
            " `send_email(to: string, body: string)`."
        ),
    )
    model_name: str = Field(
        default="unknown",
        description=(
            "The model's name used for logging, testing, and sanity checking. For some"
            " models this can be automatically discerned. For other models, like"
            " locally loaded models, this must be manually specified."
        ),
    )
    system_role: MessageRole = Field(
        default=MessageRole.SYSTEM,
        description="The role this specific LLM provider"
        "expects for system prompt. E.g. 'SYSTEM' for OpenAI, 'CHATBOT' for Cohere",
    )
