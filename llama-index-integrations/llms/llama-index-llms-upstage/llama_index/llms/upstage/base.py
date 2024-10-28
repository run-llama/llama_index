import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Sequence, Callable, Union
import warnings

import httpx
from llama_index.readers.upstage import UpstageDocumentParseReader
from llama_index.llms.openai.utils import (
    create_retry_decorator,
)
from llama_index.core.base.llms.types import (
    LLMMetadata,
    ChatMessage,
    ChatResponseGen,
    ChatResponse,
    ChatResponseAsyncGen,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import to_openai_message_dicts

from llama_index.llms.upstage.utils import (
    SOLAR_TOKENIZERS,
    is_doc_parsing_model,
    resolve_upstage_credentials,
    is_chat_model,
    upstage_modelname_to_contextsize,
    is_function_calling_model,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from tokenizers import Tokenizer
from pydantic import ConfigDict, Field, PrivateAttr
from openai import OpenAI as SyncOpenAI
from openai import AsyncOpenAI

DEFAULT_UPSTAGE_MODEL = "solar-1-mini-chat"

llm_retry_decorator = create_retry_decorator(
    max_retries=6,
    random_exponential=True,
    stop_after_delay_seconds=60,
    min_seconds=1,
    max_seconds=20,
)


class Upstage(OpenAI):
    """
    Upstage LLM.

    Examples:
        `pip install llama-index-llms-upstage`

        ```python
        from llama_index.llms.upstage import Upstage
        import os

        os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"

        llm = Upstage()
        stream = llm.stream("Hello, how are you?")

        for response in stream:
            print(response.delta, end="")

        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
    model: str = Field(
        default=DEFAULT_UPSTAGE_MODEL, description="The Upstage model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate."
    )
    logprobs: Optional[bool] = Field(
        description="Whether to return logprobs per token."
    )
    top_logprobs: int = Field(
        description="The number of top token logprobs to return.",
        default=0,
        gte=0,
        lte=20,
    )
    additional_kwargs: Dict[str, Any] = Field(
        description="Additional kwargs for the Upstage API.", default_factory=dict
    )
    max_retries: int = Field(
        description="The maximum number of API retries.", default=3, gte=0
    )
    timeout: float = Field(
        description="The timeout, in seconds, for API requests.", default=60.0, gte=0.0
    )
    reuse_client: bool = Field(
        description=(
            "Reuse the OpenAI client between requests. When doing anything with large "
            "volumes of async API calls, setting this to false can improve stability."
        ),
        default=True,
    )
    tokenizer_name: str = Field(
        description=(
            "Huggingface pretrained tokenizer name "
            "upstage opened solar tokenizer in Huggingface. https://huggingface.co/upstage/solar-1-mini-tokenizer"
        ),
        default=SOLAR_TOKENIZERS[DEFAULT_UPSTAGE_MODEL],
    )

    api_key: str = Field(
        default=None, alias="upstage_api_key", description="The Upstage API key."
    )
    api_base: str = Field(
        default="https://api.upstage.ai/v1/solar",
        description="The Upstage API base URL.",
    )

    _client: Optional[SyncOpenAI] = PrivateAttr()
    _aclient: Optional[AsyncOpenAI] = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_UPSTAGE_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: int = 0,
        additional_kwargs: Dict[str, Any] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        reuse_client: bool = True,
        tokenizer_name: str = "upstage/solar-1-mini-tokenizer",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,  # from base class
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        **kwargs: Any,
    ) -> None:
        if "upstage_api_key" in kwargs:
            api_key = kwargs.pop("upstage_api_key")
        additional_kwargs = additional_kwargs or {}
        api_key, api_base = resolve_upstage_credentials(
            api_key=api_key, api_base=api_base
        )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            timeout=timeout,
            reuse_client=reuse_client,
            api_key=api_key,
            api_base=api_base,
            callback_manager=callback_manager,
            default_headers=default_headers,
            http_client=http_client,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            **kwargs,
        )

        self.tokenizer_name = tokenizer_name
        self._client = None
        self._aclient = None
        self._http_client = http_client

    def _get_model_name(self) -> str:
        return self.model

    @classmethod
    def class_name(cls) -> str:
        return "upstage_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=upstage_modelname_to_contextsize(
                modelname=self._get_model_name()
            ),
            num_output=self.max_tokens or -1,
            is_chat_model=is_chat_model(model=self._get_model_name()),
            is_function_calling_model=is_function_calling_model(
                model=self._get_model_name()
            ),
            model_name=self.model,
        )

    @property
    def _tokenizer(self) -> Optional[Tokenizer]:
        """
        Get a Huggingface tokenizer for solar models.
        """
        if SOLAR_TOKENIZERS.get(self.model) != self.tokenizer_name:
            warnings.warn(
                f"You are using a different tokenizer than the one specified in the model. This may cause issues with token counting. Please use {SOLAR_TOKENIZERS[self.model]} as the tokenizer name."
            )
        return Tokenizer.from_pretrained(self.tokenizer_name)

    def get_num_tokens_from_message(self, messages: Sequence[ChatMessage]) -> int:
        tokens_per_message = 5  # <|im_start|>{role}\n{message}<|im_end|>
        tokens_prefix = 1  # <|startoftext|>
        tokens_suffix = 3  # <|im_start|>assistant\n

        num_tokens = 0

        num_tokens += tokens_prefix

        message_dicts = to_openai_message_dicts(messages)
        for message in message_dicts:
            num_tokens += tokens_per_message
            for value in message.values():
                num_tokens += len(
                    self._tokenizer.encode(str(value), add_special_tokens=False)
                )
        num_tokens += tokens_suffix
        return num_tokens

    @llm_retry_decorator
    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if is_doc_parsing_model(self.model, kwargs):
            document_contents = self._parse_documents(kwargs.pop("file_path"))
            messages.append(ChatMessage(role="user", content=document_contents))
        return super()._chat(messages, **kwargs)

    @llm_retry_decorator
    def _achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if is_doc_parsing_model(self.model, kwargs):
            document_contents = self._parse_documents(kwargs.pop("file_path"))
            messages.append(ChatMessage(role="user", content=document_contents))
        return super()._achat(messages, **kwargs)

    @llm_retry_decorator
    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if is_doc_parsing_model(self.model, kwargs):
            document_contents = self._parse_documents(kwargs.pop("file_path"))
            messages.append(ChatMessage(role="user", content=document_contents))
        return super()._stream_chat(messages, **kwargs)

    @llm_retry_decorator
    def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        if is_doc_parsing_model(self.model, kwargs):
            document_contents = self._parse_documents(kwargs.pop("file_path"))
            messages.append(ChatMessage(role="user", content=document_contents))
        return super()._astream_chat(messages, **kwargs)

    def _parse_documents(
        self, file_path: Union[str, Path, List[str], List[Path]]
    ) -> str:
        document_contents = "Documents:\n"

        loader = UpstageDocumentParseReader(
            api_key=self.api_key, output_format="text", coordinates=False
        )
        docs = loader.load_data(file_path)

        if isinstance(file_path, list):
            file_titles = [os.path.basename(path) for path in file_path]
        else:
            file_titles = [os.path.basename(file_path)]

        for i, doc in enumerate(docs):
            file_title = file_titles[min(i, len(file_titles) - 1)]
            document_contents += f"{file_title}:\n{doc.text}\n\n"
        return document_contents
