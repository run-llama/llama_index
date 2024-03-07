"""PremAI's API to interact with deployed projects"""

import os
import typing
from typing import Any, Dict, Optional, Sequence, Callable

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    get_from_param_or_env,
    stream_chat_to_completion_decorator,
)

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.llm import LLM

from premai import Prem


# FIXME: The current version does not support stop tokens and number of responses i.e. n > 1

# TODO: Fetch the default values from prem-sdk


class PremAI(LLM):
    """PremAI LLM Provider"""

    project_id: int = Field(
        description=(
            "The project ID in which the experiments or deployements are carried out. can find all your projects here: https://app.premai.io/projects/"
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
            "Name of the model. This is an optional paramter. The default model is the one deployed from Prem's LaunchPad. An example: https://app.premai.io/projects/<project-id>/launchpad. If model name is other than default model then it will override the calls from the model deployed from launchpad."
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
        description="Model temperature. Value shoud be >= 0 and <= 1.0"
    )

    top_p: Optional[float] = Field(
        description="top_p adjusts the number of choices for each predicted tokens based on cumulative probabilities. Value should be ranging between 0.0 and 1.0."
    )

    max_retries: Optional[int] = Field(
        description="Max number of retries to call the API"
    )

    streaming: Optional[bool] = Field(
        description="Whether to stream token responses or not."
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

    _client: "premai.Prem" = PrivateAttr()

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

    def chat(self):
        raise NotImplementedError

    def achat(self):
        raise NotImplementedError

    def stream_chat(self):
        raise NotImplementedError

    def astream_chat(self):
        raise NotImplementedError

    def complete(self):
        raise NotImplementedError

    def acomplete(self):
        raise NotImplementedError

    def stream_complete(self):
        raise NotImplementedError

    def astream_complete(self):
        raise NotImplementedError
