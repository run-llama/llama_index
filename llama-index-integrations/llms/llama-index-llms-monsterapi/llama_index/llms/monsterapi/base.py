from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

DEFAULT_MONSTER_TEMP = 0.75


class MonsterLLM(CustomLLM):
    """MonsterAPI LLM.

    Monster Deploy enables you to host any vLLM supported large language model (LLM) like Tinyllama, Mixtral, Phi-2 etc as a rest API endpoint on MonsterAPI's cost optimised GPU cloud.

    With MonsterAPI's integration in Llama index, you can use your deployed LLM API endpoints to create RAG system or RAG bot for use cases such as:
    - Answering questions on your documents
    - Improving the content of your documents
    - Finding context of importance in your documents


    Once deployment is launched use the base_url and api_auth_token once deployment is live and use them below.

    Note: When using LLama index to access Monster Deploy LLMs, you need to create a prompt with required template and send compiled prompt as input.
    See `LLama Index Prompt Template Usage example` section for more details.

    see (https://developer.monsterapi.ai/docs/monster-deploy-beta) for more details

    Once deployment is launched use the base_url and api_auth_token once deployment is live and use them below.

    Note: When using LLama index to access Monster Deploy LLMs, you need to create a prompt with reqhired template and send compiled prompt as input. see section `LLama Index Prompt Template
    Usage example` for more details.

    Examples:
        `pip install llama-index-llms-monsterapi`

        ```python
        llm = MonsterLLM(
            model="deploy-llm",
            base_url="https://ecc7deb6-26e0-419b-a7f2-0deb934af29a.monsterapi.ai",
            monster_api_key="a0f8a6ba-c32f-4407-af0c-169f1915490c",
            temperature=0.75,
        )

        response = llm.complete("What is the capital of France?")
        print(str(response))
        ```
    """

    model: str = Field(description="The MonsterAPI model to use.")
    monster_api_key: Optional[str] = Field(description="The MonsterAPI key to use.")
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    temperature: float = Field(
        default=DEFAULT_MONSTER_TEMP,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The number of context tokens available to the LLM.",
        gt=0,
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.monsterapi.ai/v1",
        monster_api_key: Optional[str] = None,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        temperature: float = DEFAULT_MONSTER_TEMP,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        self._client, available_llms = self.initialize_client(monster_api_key, base_url)

        # Check if provided model is supported
        if model not in available_llms:
            error_message = (
                f"Model: {model} is not supported. "
                f"Supported models are {available_llms}. "
                "Please update monsterapiclient to see if any models are added. "
                "pip install --upgrade monsterapi"
            )
            raise RuntimeError(error_message)

        super().__init__(
            model=model,
            monster_api_key=monster_api_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            context_window=context_window,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    def initialize_client(
        self, monster_api_key: Optional[str], base_url: Optional[str]
    ) -> Any:
        try:
            from monsterapi import client as MonsterClient
            from monsterapi.InputDataModels import MODEL_TYPES
        except ImportError:
            raise ImportError(
                "Could not import Monster API client library."
                "Please install it with `pip install monsterapi`"
            )

        llm_models_enabled = [i for i, j in MODEL_TYPES.items() if j == "LLM"]

        return MonsterClient(monster_api_key, base_url), llm_models_enabled

    @classmethod
    def class_name(cls) -> str:
        return "MonsterLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model,
        )

    def _get_input_dict(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_length": self.max_new_tokens,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        return self.complete(prompt, formatted=True, **kwargs)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, timeout: int = 100, **kwargs: Any
    ) -> CompletionResponse:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        stream = kwargs.pop("stream", False)

        if stream is True:
            raise NotImplementedError(
                "complete method cannot be used with stream=True, please use stream_complete method"
            )

        # Validate input args against input Pydantic model
        input_dict = self._get_input_dict(prompt, **kwargs)

        result = self._client.generate(
            model=self.model, data=input_dict, timeout=timeout
        )

        if isinstance(result, Exception):
            raise result

        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])

        if isinstance(result, dict) and "text" in result:
            if isinstance(result["text"], list):
                return CompletionResponse(text=result["text"][0])
            elif isinstance(result["text"], str):
                return CompletionResponse(text=result["text"])

        if isinstance(result, list):
            return CompletionResponse(text=result[0]["text"])

        raise RuntimeError("Unexpected Return please contact monsterapi support!")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        if "deploy" not in self.model:
            raise NotImplementedError(
                "stream_complete method can only be used with deploy models for now. Support for other models will be added soon."
            )

        # Validate input args against input Pydantic model
        input_dict = self._get_input_dict(prompt, **kwargs)
        input_dict["stream"] = True

        # Starting the stream
        result_stream = self._client.generate(model=self.model, data=input_dict)

        if isinstance(result_stream, Exception):
            raise result_stream

        if isinstance(result_stream, dict) and "error" in result_stream:
            raise RuntimeError(result_stream["error"])

        # Iterating over the generator
        try:
            for result in result_stream:
                yield CompletionResponse(text=result[0])
        except StopIteration:
            pass
