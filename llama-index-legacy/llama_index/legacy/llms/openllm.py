import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
)

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.generic_utils import (
    completion_response_to_chat_response,
)
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.llms.llm import LLM
from llama_index.types import PydanticProgramMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import TypeVar

    M = TypeVar("M")
    T = TypeVar("T")
    Metadata = Any


class OpenLLM(LLM):
    """OpenLLM LLM."""

    model_id: str = Field(
        description="Given Model ID from HuggingFace Hub. This can be either a pretrained ID or local path. This is synonymous to HuggingFace's '.from_pretrained' first argument"
    )
    model_version: Optional[str] = Field(
        description="Optional model version to save the model as."
    )
    model_tag: Optional[str] = Field(
        description="Optional tag to save to BentoML store."
    )
    prompt_template: Optional[str] = Field(
        description="Optional prompt template to pass for this LLM."
    )
    backend: Optional[Literal["vllm", "pt"]] = Field(
        description="Optional backend to pass for this LLM. By default, it will use vLLM if vLLM is available in local system. Otherwise, it will fallback to PyTorch."
    )
    quantize: Optional[Literal["awq", "gptq", "int8", "int4", "squeezellm"]] = Field(
        description="Optional quantization methods to use with this LLM. See OpenLLM's --quantize options from `openllm start` for more information."
    )
    serialization: Literal["safetensors", "legacy"] = Field(
        description="Optional serialization methods for this LLM to be save as. Default to 'safetensors', but will fallback to PyTorch pickle `.bin` on some models."
    )
    trust_remote_code: bool = Field(
        description="Optional flag to trust remote code. This is synonymous to Transformers' `trust_remote_code`. Default to False."
    )
    if TYPE_CHECKING:
        from typing import Generic

        try:
            import openllm

            _llm: openllm.LLM[Any, Any]
        except ImportError:
            _llm: Any  # type: ignore[no-redef]
    else:
        _llm: Any = PrivateAttr()

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        model_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        backend: Optional[Literal["vllm", "pt"]] = None,
        *args: Any,
        quantize: Optional[Literal["awq", "gptq", "int8", "int4", "squeezellm"]] = None,
        serialization: Literal["safetensors", "legacy"] = "safetensors",
        trust_remote_code: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        **attrs: Any,
    ):
        try:
            import openllm
        except ImportError:
            raise ImportError(
                "OpenLLM is not installed. Please install OpenLLM via `pip install openllm`"
            )
        self._llm = openllm.LLM[Any, Any](
            model_id,
            model_version=model_version,
            model_tag=model_tag,
            prompt_template=prompt_template,
            system_message=system_prompt,
            backend=backend,
            quantize=quantize,
            serialisation=serialization,
            trust_remote_code=trust_remote_code,
            embedded=True,
            **attrs,
        )
        if messages_to_prompt is None:
            messages_to_prompt = self._tokenizer_messages_to_prompt

        # NOTE: We need to do this here to ensure model is saved and revision is set correctly.
        assert self._llm.bentomodel

        super().__init__(
            model_id=model_id,
            model_version=self._llm.revision,
            model_tag=str(self._llm.tag),
            prompt_template=prompt_template,
            backend=self._llm.__llm_backend__,
            quantize=self._llm.quantise,
            serialization=self._llm._serialisation,
            trust_remote_code=self._llm.trust_remote_code,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OpenLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            num_output=self._llm.config["max_new_tokens"],
            model_name=self.model_id,
        )

    def _tokenizer_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Use the tokenizer to convert messages to prompt. Fallback to generic."""
        if hasattr(self._llm.tokenizer, "apply_chat_template"):
            return self._llm.tokenizer.apply_chat_template(
                [message.dict() for message in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        return generic_messages_to_prompt(messages)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return asyncio.run(self.acomplete(prompt, **kwargs))

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return asyncio.run(self.achat(messages, **kwargs))

    @property
    def _loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        return loop

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        generator = self.astream_complete(prompt, **kwargs)
        # Yield items from the queue synchronously
        while True:
            try:
                yield self._loop.run_until_complete(generator.__anext__())
            except StopAsyncIteration:
                break

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        generator = self.astream_chat(messages, **kwargs)
        # Yield items from the queue synchronously
        while True:
            try:
                yield self._loop.run_until_complete(generator.__anext__())
            except StopAsyncIteration:
                break

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        response = await self.acomplete(self.messages_to_prompt(messages), **kwargs)
        return completion_response_to_chat_response(response)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = await self._llm.generate(prompt, **kwargs)
        return CompletionResponse(
            text=response.outputs[0].text,
            raw=response.model_dump(),
            additional_kwargs={
                "prompt_token_ids": response.prompt_token_ids,
                "prompt_logprobs": response.prompt_logprobs,
                "finished": response.finished,
                "outputs": {
                    "token_ids": response.outputs[0].token_ids,
                    "cumulative_logprob": response.outputs[0].cumulative_logprob,
                    "logprobs": response.outputs[0].logprobs,
                    "finish_reason": response.outputs[0].finish_reason,
                },
            },
        )

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        async for response_chunk in self.astream_complete(
            self.messages_to_prompt(messages), **kwargs
        ):
            yield completion_response_to_chat_response(response_chunk)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        config = self._llm.config.model_construct_env(**kwargs)
        if config["n"] > 1:
            logger.warning("Currently only support n=1")

        texts: List[List[str]] = [[]] * config["n"]

        async for response_chunk in self._llm.generate_iterator(prompt, **kwargs):
            for output in response_chunk.outputs:
                texts[output.index].append(output.text)
            yield CompletionResponse(
                text=response_chunk.outputs[0].text,
                delta=response_chunk.outputs[0].text,
                raw=response_chunk.model_dump(),
                additional_kwargs={
                    "prompt_token_ids": response_chunk.prompt_token_ids,
                    "prompt_logprobs": response_chunk.prompt_logprobs,
                    "finished": response_chunk.finished,
                    "outputs": {
                        "text": response_chunk.outputs[0].text,
                        "token_ids": response_chunk.outputs[0].token_ids,
                        "cumulative_logprob": response_chunk.outputs[
                            0
                        ].cumulative_logprob,
                        "logprobs": response_chunk.outputs[0].logprobs,
                        "finish_reason": response_chunk.outputs[0].finish_reason,
                    },
                },
            )


class OpenLLMAPI(LLM):
    """OpenLLM Client interface. This is useful when interacting with a remote OpenLLM server."""

    address: Optional[str] = Field(
        description="OpenLLM server address. This could either be set here or via OPENLLM_ENDPOINT"
    )
    timeout: int = Field(description="Timeout for sending requests.")
    max_retries: int = Field(description="Maximum number of retries.")
    api_version: Literal["v1"] = Field(description="OpenLLM Server API version.")

    if TYPE_CHECKING:
        try:
            from openllm_client import AsyncHTTPClient, HTTPClient

            _sync_client: HTTPClient
            _async_client: AsyncHTTPClient
        except ImportError:
            _sync_client: Any  # type: ignore[no-redef]
            _async_client: Any  # type: ignore[no-redef]
    else:
        _sync_client: Any = PrivateAttr()
        _async_client: Any = PrivateAttr()

    def __init__(
        self,
        address: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 2,
        api_version: Literal["v1"] = "v1",
        **kwargs: Any,
    ):
        try:
            from openllm_client import AsyncHTTPClient, HTTPClient
        except ImportError:
            raise ImportError(
                f'"{type(self).__name__}" requires "openllm-client". Make sure to install with `pip install openllm-client`'
            )
        super().__init__(
            address=address,
            timeout=timeout,
            max_retries=max_retries,
            api_version=api_version,
            **kwargs,
        )
        self._sync_client = HTTPClient(
            address=address,
            timeout=timeout,
            max_retries=max_retries,
            api_version=api_version,
        )
        self._async_client = AsyncHTTPClient(
            address=address,
            timeout=timeout,
            max_retries=max_retries,
            api_version=api_version,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OpenLLM_Client"

    @property
    def _server_metadata(self) -> "Metadata":
        return self._sync_client._metadata

    @property
    def _server_config(self) -> Dict[str, Any]:
        return self._sync_client._config

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            num_output=self._server_config["max_new_tokens"],
            model_name=self._server_metadata.model_id.replace("/", "--"),
        )

    def _convert_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return self._sync_client.helpers.messages(
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            add_generation_prompt=True,
        )

    async def _async_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return await self._async_client.helpers.messages(
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            add_generation_prompt=True,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = self._sync_client.generate(prompt, **kwargs)
        return CompletionResponse(
            text=response.outputs[0].text,
            raw=response.model_dump(),
            additional_kwargs={
                "prompt_token_ids": response.prompt_token_ids,
                "prompt_logprobs": response.prompt_logprobs,
                "finished": response.finished,
                "outputs": {
                    "token_ids": response.outputs[0].token_ids,
                    "cumulative_logprob": response.outputs[0].cumulative_logprob,
                    "logprobs": response.outputs[0].logprobs,
                    "finish_reason": response.outputs[0].finish_reason,
                },
            },
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        for response_chunk in self._sync_client.generate_stream(prompt, **kwargs):
            yield CompletionResponse(
                text=response_chunk.text,
                delta=response_chunk.text,
                raw=response_chunk.model_dump(),
                additional_kwargs={"token_ids": response_chunk.token_ids},
            )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return completion_response_to_chat_response(
            self.complete(self._convert_messages_to_prompt(messages), **kwargs)
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        for response_chunk in self.stream_complete(
            self._convert_messages_to_prompt(messages), **kwargs
        ):
            yield completion_response_to_chat_response(response_chunk)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = await self._async_client.generate(prompt, **kwargs)
        return CompletionResponse(
            text=response.outputs[0].text,
            raw=response.model_dump(),
            additional_kwargs={
                "prompt_token_ids": response.prompt_token_ids,
                "prompt_logprobs": response.prompt_logprobs,
                "finished": response.finished,
                "outputs": {
                    "token_ids": response.outputs[0].token_ids,
                    "cumulative_logprob": response.outputs[0].cumulative_logprob,
                    "logprobs": response.outputs[0].logprobs,
                    "finish_reason": response.outputs[0].finish_reason,
                },
            },
        )

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async for response_chunk in self._async_client.generate_stream(
            prompt, **kwargs
        ):
            yield CompletionResponse(
                text=response_chunk.text,
                delta=response_chunk.text,
                raw=response_chunk.model_dump(),
                additional_kwargs={"token_ids": response_chunk.token_ids},
            )

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return completion_response_to_chat_response(
            await self.acomplete(
                await self._async_messages_to_prompt(messages), **kwargs
            )
        )

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async for response_chunk in self.astream_complete(
            await self._async_messages_to_prompt(messages), **kwargs
        ):
            yield completion_response_to_chat_response(response_chunk)
