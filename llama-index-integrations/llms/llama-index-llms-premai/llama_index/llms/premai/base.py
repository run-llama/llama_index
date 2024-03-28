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

from premai import Prem


# FIXME: The current version does not support stop tokens and number of responses i.e. n > 1
# TODO: Fetch the default values from prem-sdk


class ChatPremError(Exception):
    pass


class PremAI(LLM):
    """PremAI LLM Provider."""

    project_id: int = Field(
        description=(
            "The project ID in which the experiments or deployments are carried out. can find all your projects here: https://app.premai.io/projects/"
        )
    )

    session_id: Optional[str] = Field(
        description="The ID of the session to use. It helps to track the chat history."
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

    top_p: Optional[float] = Field(
        description="top_p adjusts the number of choices for each predicted tokens based on cumulative probabilities. Value should be ranging between 0.0 and 1.0."
    )

    max_retries: Optional[int] = Field(
        description="Max number of retries to call the API"
    )

    tools: Optional[Dict[str, Any]] = Field(
        description="A list of tools the model may call. Currently, only functions are supported as a tool"
    )

    frequency_penalty: Optional[float] = Field(
        description=(
            "Number between -2.0 and 2.0. Positive values penalize new tokens based."
        ),
    )

    presence_penalty: Optional[float] = Field(
        description=(
            "Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far."
        ),
    )

    logit_bias: Optional[dict] = Field(
        description=(
            "JSON object that maps tokens to an associated bias value from -100 to 100."
        ),
    )

    seed: Optional[int] = Field(
        description=(
            "This feature is in Beta. If specified, our system will make a best effort to sample deterministically."
        ),
    )

    _client: "Prem" = PrivateAttr()

    def __init__(
        self,
        project_id: int,
        premai_api_key: Optional[str] = None,
        session_id: Optional[int] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[str] = 128,
        temperature: Optional[float] = 0.1,
        top_p: Optional[float] = 0.7,
        max_retries: Optional[int] = 1,
        tools: Optional[Dict[str, Any]] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        seed: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ):
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_key = get_from_param_or_env("api_key", premai_api_key, "PREMAI_API_KEY", "")

        if not api_key:
            raise ValueError(
                "You must provide an API key to use premai. "
                "You can either pass it in as an argument or set it `PREMAI_API_KEY`. You can get your API key here: https://app.premai.io/api_keys/"
            )

        self._client = Prem(api_key=api_key)

        super().__init__(
            project_id=project_id,
            session_id=session_id,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            api_key=api_key,
            callback_manager=callback_manager,
            top_p=top_p,
            system_prompt=system_prompt,
            additional_kwargs=additional_kwargs,
            logit_bias=logit_bias,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            seed=seed,
            max_retries=max_retries,
            tools=tools,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

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
            top_p=self.top_p,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "top_p": self.top_p,
            "system_prompt": self.system_prompt,
            "logit_bias": self.logit_bias,
            "tools": self.tools,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def _get_all_kwargs(self, **kwargs) -> Dict[str, Any]:
        all_kwargs = {**self._model_kwargs, **kwargs}
        _keys_that_cannot_be_none = [
            "system_prompt",
            "frequency_penalty",
            "presence_penalty",
            "tools",
            "model",
        ]

        for key in _keys_that_cannot_be_none:
            if all_kwargs.get(key) is None:
                all_kwargs.pop(key, None)
        return all_kwargs

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_messages = []

        for message in messages:
            if "system_prompt" in all_kwargs and message.role.value == "system":
                continue

            elif "system_prompt" not in all_kwargs and message.role.value == "system":
                all_kwargs["system_prompt"] = message.content
            else:
                chat_messages.append(
                    {"role": message.role.value, "content": message.content}
                )
        response = self._client.chat.completions.create(
            project_id=self.project_id, messages=chat_messages, **all_kwargs
        )
        if not response.choices:
            raise ChatPremError("ChatResponse must have at least one candidate")

        chat_responses: Sequence[ChatResponse] = []

        for choice in response.choices:
            role = choice.message.role
            if role is None:
                raise ChatPremError(f"ChatResponse {choice} must have a role.")
            content = choice.message.content or ""
            chat_responses.append(
                ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    raw={"role": role, "content": content},
                )
            )

        if "is_completion" in kwargs:
            return chat_responses[0]

        return chat_responses

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_messages = []

        for message in messages:
            if "system_prompt" in all_kwargs and message.role.value == "system":
                continue

            elif "system_prompt" not in all_kwargs and message.role.value == "system":
                all_kwargs["system_prompt"] = message.content
            else:
                chat_messages.append(
                    {"role": message.role.value, "content": message.content}
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
