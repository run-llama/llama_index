import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.xinference.utils import (
    xinference_message_to_history,
    xinference_modelname_to_contextsize,
)

# an approximation of the ratio between llama and GPT2 tokens
TOKEN_RATIO = 2.5
DEFAULT_XINFERENCE_TEMP = 1.0


class Xinference(CustomLLM):
    """
    Xinference LLM.

    Examples:
        `pip install llama-index-llms-xinference`

        ```python
        from llama_index.llms.xinference import Xinference

        # Set up Xinference with required parameters
        llm = Xinference(
            model_name="xinference-1.0",
            app_id="ml",
            user_id="xinference",
            api_key="<YOUR XINFERENCE API KEY>"
            temperature=0.5,
            max_tokens=256,
        )

        # Call the complete function
        response = llm.complete("Hello World!")
        print(response)
        ```

    """

    model_uid: str = Field(description="The Xinference model to use.")
    endpoint: str = Field(description="The Xinference endpoint URL to use.")
    temperature: float = Field(
        description="The temperature to use for sampling.", ge=0.0, le=1.0
    )
    max_tokens: int = Field(
        description="The maximum new tokens to generate as answer.", gt=0
    )
    context_window: int = Field(
        description="The maximum number of context tokens for the model.", gt=0
    )
    model_description: Dict[str, Any] = Field(
        description="The model description from Xinference."
    )

    _generator: Any = PrivateAttr()

    def __init__(
        self,
        model_uid: str,
        endpoint: str,
        temperature: float = DEFAULT_XINFERENCE_TEMP,
        max_tokens: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        generator, context_window, model_description = self.load_model(
            model_uid, endpoint
        )
        if max_tokens is None:
            max_tokens = context_window // 4
        elif max_tokens > context_window:
            raise ValueError(
                f"received max_tokens {max_tokens} with context window {context_window}"
                "max_tokens can not exceed the context window of the model"
            )

        super().__init__(
            model_uid=model_uid,
            endpoint=endpoint,
            temperature=temperature,
            context_window=context_window,
            max_tokens=max_tokens,
            model_description=model_description,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )
        self._generator = generator

    def load_model(self, model_uid: str, endpoint: str) -> Tuple[Any, int, dict]:
        try:
            from xinference.client import RESTfulClient
        except ImportError:
            raise ImportError(
                "Could not import Xinference library."
                'Please install Xinference with `pip install "xinference[all]"`'
            )

        client = RESTfulClient(endpoint)

        try:
            assert isinstance(client, RESTfulClient)
        except AssertionError:
            raise RuntimeError(
                "Could not create RESTfulClient instance."
                "Please make sure Xinference endpoint is running at the correct port."
            )

        generator = client.get_model(model_uid)
        model_description = client.list_models()[model_uid]

        try:
            assert generator is not None
            assert model_description is not None
        except AssertionError:
            raise RuntimeError(
                "Could not get model from endpoint."
                "Please make sure Xinference endpoint is running at the correct port."
            )

        model = model_description["model_name"]
        if "context_length" in model_description:
            context_window = model_description["context_length"]
        else:
            warnings.warn(
                """
            Parameter `context_length` not found in model description,
            using `xinference_modelname_to_contextsize` that is no longer maintained.
            Please update Xinference to the newest version.
            """
            )
            context_window = xinference_modelname_to_contextsize(model)

        return generator, context_window, model_description

    @classmethod
    def class_name(cls) -> str:
        return "Xinference_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        assert isinstance(self.context_window, int)
        return LLMMetadata(
            context_window=int(self.context_window // TOKEN_RATIO),
            num_output=self.max_tokens,
            model_name=self.model_uid,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        assert self.context_window is not None
        base_kwargs = {
            "temperature": self.temperature,
            "max_length": self.context_window,
        }
        return {
            **base_kwargs,
            **self.model_description,
        }

    def _get_input_dict(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {"prompt": prompt, **self._model_kwargs, **kwargs}

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        assert self._generator is not None
        prompt = messages[-1].content if len(messages) > 0 else ""
        history = [xinference_message_to_history(message) for message in messages[:-1]]
        response_text = self._generator.chat(
            prompt=prompt,
            chat_history=history,
            generate_config={
                "stream": False,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )["choices"][0]["message"]["content"]
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_text,
            ),
            delta=None,
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        assert self._generator is not None
        prompt = messages[-1].content if len(messages) > 0 else ""
        history = [xinference_message_to_history(message) for message in messages[:-1]]
        response_iter = self._generator.chat(
            prompt=prompt,
            chat_history=history,
            generate_config={
                "stream": True,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )

        def gen() -> ChatResponseGen:
            text = ""
            for c in response_iter:
                delta = c["choices"][0]["delta"].get("content", "")
                text += delta
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=text,
                    ),
                    delta=delta,
                )

        return gen()

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        assert self._generator is not None
        response_text = self._generator.chat(
            prompt=prompt,
            chat_history=None,
            generate_config={
                "stream": False,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )["choices"][0]["message"]["content"]
        return CompletionResponse(
            delta=None,
            text=response_text,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        assert self._generator is not None
        response_iter = self._generator.chat(
            prompt=prompt,
            chat_history=None,
            generate_config={
                "stream": True,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )

        def gen() -> CompletionResponseGen:
            text = ""
            for c in response_iter:
                delta = c["choices"][0]["delta"].get("content", "")
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                )

        return gen()
