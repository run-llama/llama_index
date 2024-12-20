from typing import Any, Callable, Dict, Optional, Sequence

import requests

from llama_index.core.base.llms.types import ChatMessage, LLMMetadata
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.openai import OpenAI

from pydantic import Field

DEFAULT_API_BASE = "https://llm.monsterapi.ai/v1"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


class MonsterLLM(OpenAI):
    model_info: dict = Field(
        description="Model info field with pricing and other llm model information in json structure.",
        default={},
    )

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

        1. MonsterAPI Private LLM Deployment use case
        ```python
        from llama_index.llms.monsterapi import MonsterLLM
        # User monsterAPI Deploy service to launch a deployment
        # then get api_endpoint and api_auth_token and use them as api_base and api_key respectively.
        llm = MonsterLLM(
            model = "whatever is the basemodel used to deploy the llm",
            api_base="https://ecc7deb6-26e0-419b-a7f2-0deb934af29a.monsterapi.ai",
            api_key="a0f8a6ba-c32f-4407-af0c-169f1915490c",
            temperature=0.75,
        )

        response = llm.complete("What is the capital of France?")
        ```

        2. Monster API General Available LLMs
        ```python3
        from llama_index.llms.monsterapi import MonsterLLM
        llm = MonsterLLM(
            model="microsoft/Phi-3-mini-4k-instruct"
        )

        response = llm.complete("What is the capital of France?")
        print(str(response))
        ```
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        api_base: Optional[str] = DEFAULT_API_BASE,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_base = get_from_param_or_env("api_base", api_base, "MONSTER_API_BASE")
        api_key = get_from_param_or_env("api_key", api_key, "MONSTER_API_KEY")

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self.model_info = self._fetch_model_details(api_base, api_key)

    @classmethod
    def class_name(cls) -> str:
        return "MonsterAPI LLMs"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            is_function_calling_model=False,
        )

    @property
    def _is_chat_model(self) -> bool:
        return True

    def _fetch_model_details(self, api_base: str, api_key: str):
        headers = {"Authorization": f"Bearer {api_key}", "accept": "application/json"}
        response = requests.get(f"{api_base}/models/info", headers=headers)
        response.raise_for_status()

        details = response.json()
        return details["maximum_context_length"]

    def _modelname_to_contextsize(self, model_name):
        return self.model_info.get(model_name)
