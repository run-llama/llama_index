"""PremAI's API integration with llama-index to interact with deployed projects."""

from typing import Any, Dict, Optional, Sequence, Callable

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.base.llms.generic_utils import (
    chat_to_completion_decorator,
    get_from_param_or_env,
    stream_chat_to_completion_decorator,
)

from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.llm import LLM

from llama_index.llms.premai.utils import prepare_messages_before_chat

from premai import Prem


class ChatPremError(Exception):
    pass


class PremAI(LLM):
    """PremAI LLM Provider."""

    project_id: int = Field(
        description=(
            "The project ID in which the experiments or deployments are carried out. can find all your projects here: https://app.premai.io/projects/"
        )
    )

    premai_api_key: Optional[str] = Field(
        description="Prem AI API Key. Get it here: https://app.premai.io/api_keys/"
    )

    model: Optional[str] = Field(
        description=(
            "Name of the model. This is an optional parameter. The default model is the one deployed from Prem's LaunchPad. An example: https://app.premai.io/projects/<project-id>/launchpad. If model name is other than default model then it will override the calls from the model deployed from launchpad."
        ),
    )
    system_prompt: Optional[str] = Field(
        description=(
            "System prompts helps the model to guide the generation and the way it acts. Default system prompt is the one set on your deployed LaunchPad model under the specified project."
        ),
    )

    max_tokens: Optional[int] = Field(
        description=("The max number of tokens to output from the LLM. ")
    )

    temperature: Optional[float] = Field(
        description="Model temperature. Value should be >= 0 and <= 1.0"
    )

    max_retries: Optional[int] = Field(
        description="Max number of retries to call the API"
    )

    repositories: Optional[dict] = Field(
        description="Add valid repository ids. This will be overriding existing connected repositories (if any) and will use RAG with the connected repos."
    )

    additional_kwargs: Optional[dict] = Field(
        description="Add any additional kwargs. This may override your existing settings."
    )

    _client: "Prem" = PrivateAttr()

    def __init__(
        self,
        project_id: int,
        premai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[str] = 128,
        temperature: Optional[float] = 0.1,
        max_retries: Optional[int] = 1,
        repositories: Optional[dict] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        **kwargs,
    ):
        callback_manager = callback_manager or CallbackManager([])

        api_key = get_from_param_or_env("api_key", premai_api_key, "PREMAI_API_KEY", "")

        if not api_key:
            raise ValueError(
                "You must provide an API key to use premai. "
                "You can either pass it in as an argument or set it `PREMAI_API_KEY`. You can get your API key here: https://app.premai.io/api_keys/"
            )

        additional_kwargs = {**(additional_kwargs or {}), **kwargs}

        super().__init__(
            project_id=project_id,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            api_key=api_key,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            additional_kwargs=additional_kwargs,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            max_retries=max_retries,
            repositories=repositories,
        )
        self._client = Prem(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "PremAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        # TODO: We need to fetch information from prem-sdk here
        return LLMMetadata(
            num_output=self.max_tokens,
            is_chat_model=True,
            temperature=self.temperature,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "repositories": self.repositories,
        }

    def _get_all_kwargs(self, **kwargs) -> Dict[str, Any]:
        kwargs_to_ignore = [
            "top_p",
            "tools",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "stop",
            "seed",
        ]
        keys_to_remove = []
        for key in kwargs:
            if key in kwargs_to_ignore:
                print(f"WARNING: Parameter {key} is not supported in kwargs.")
                keys_to_remove.append(key)

        for key in keys_to_remove:
            kwargs.pop(key)

        all_kwargs = {**self._model_kwargs, **kwargs}

        for key in list(self._model_kwargs.keys()):
            if all_kwargs.get(key) is None or all_kwargs.get(key) == "":
                all_kwargs.pop(key, None)
        return all_kwargs

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**{**self.additional_kwargs, **kwargs})

        chat_messages, all_kwargs = prepare_messages_before_chat(
            messages=messages, **all_kwargs
        )

        response = self._client.chat.completions.create(
            project_id=self.project_id, messages=chat_messages, **all_kwargs
        )
        if not response.choices:
            raise ChatPremError("ChatResponse must have at least one candidate")

        choice = response.choices[0]
        role = choice.message.role

        if role is None:
            raise ChatPremError(f"ChatResponse {choice} must have a role.")
        content = choice.message.content or ""

        return ChatResponse(
            message=ChatMessage(role=role, content=content),
            raw={
                "role": role,
                "content": content,
                "document_chunks": [
                    chunk.to_dict() for chunk in response.document_chunks
                ],
            },
        )

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        all_kwargs = self._get_all_kwargs(**{**self.additional_kwargs, **kwargs})

        chat_messages, all_kwargs = prepare_messages_before_chat(
            messages=messages, **all_kwargs
        )

        response_generator = self._client.chat.completions.create(
            project_id=self.project_id,
            messages=chat_messages,
            stream=True,
            **all_kwargs,
        )

        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for chunk in response_generator:
                delta = chunk.choices[0].delta
                if delta is None or delta["content"] is None:
                    continue

                chunk_content = delta["content"]
                content += chunk_content

                yield ChatResponse(
                    message=ChatMessage(content=content, role=role), delta=chunk_content
                )

        return gen()

    def achat(self):
        raise NotImplementedError(
            "Current version of premai does not support async calls."
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        kwargs["is_completion"] = True
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    def acomplete(self):
        raise NotImplementedError(
            "Current version of premai does not support async calls."
        )

    def astream_complete(self):
        raise NotImplementedError(
            "Current version of premai does not support async calls."
        )

    def astream_chat(self):
        raise NotImplementedError(
            "Current version of premai does not support async calls."
        )
