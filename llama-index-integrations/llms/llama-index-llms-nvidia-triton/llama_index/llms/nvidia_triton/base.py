# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import random
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
)

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.base.llms.generic_utils import (
    completion_to_chat_decorator,
    stream_completion_to_chat_decorator,
)
from llama_index.core.llms.llm import LLM
from llama_index.llms.nvidia_triton.utils import GrpcTritonClient

DEFAULT_SERVER_URL = "localhost:8001"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 60.0
DEFAULT_MODEL = "ensemble"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0
DEFAULT_TOP_K = 1.0
DEFAULT_MAX_TOKENS = 100
DEFAULT_BEAM_WIDTH = 1
DEFAULT_REPTITION_PENALTY = 1.0
DEFAULT_LENGTH_PENALTY = 1.0
DEFAULT_REUSE_CLIENT = False
DEFAULT_TRITON_LOAD_MODEL = False


class NvidiaTriton(LLM):
    """
    Nvidia Triton LLM.

    Nvidia's Triton is an inference server that provides API access to hosted LLM models. This connector allows for llama_index to remotely interact with a Triton inference server over GRPC to
    accelerate inference operations.

    [Triton Inference Server Github](https://github.com/triton-inference-server/server)

    Examples:
        `pip install llama-index-llms-nvidia-triton`

        ```python
        from llama_index.llms.nvidia_triton import NvidiaTriton

        # Ensure a Triton server instance is running and provide the correct URL for your Triton server instance
        triton_url = "localhost:8001"

        # Instantiate the NvidiaTriton class
        triton_client = NvidiaTriton()

        # Call the complete method with a prompt
        resp = triton_client.complete("The tallest mountain in North America is ")
        print(resp)
        ```

    """

    server_url: str = Field(
        default=DEFAULT_SERVER_URL,
        description="The URL of the Triton inference server to use.",
    )
    model_name: str = Field(
        default=DEFAULT_MODEL,
        description="The name of the Triton hosted model this client should use",
    )
    temperature: Optional[float] = Field(
        default=DEFAULT_TEMPERATURE, description="Temperature to use for sampling"
    )
    top_p: Optional[float] = Field(
        default=DEFAULT_TOP_P, description="The top-p value to use for sampling"
    )
    top_k: Optional[float] = Field(
        default=DEFAULT_TOP_K, description="The top k value to use for sampling"
    )
    tokens: Optional[int] = Field(
        default=DEFAULT_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
    )
    beam_width: Optional[int] = Field(
        default=DEFAULT_BEAM_WIDTH, description="Last n number of tokens to penalize"
    )
    repetition_penalty: Optional[float] = Field(
        default=DEFAULT_REPTITION_PENALTY,
        description="Last n number of tokens to penalize",
    )
    length_penalty: Optional[float] = Field(
        default=DEFAULT_LENGTH_PENALTY,
        description="The penalty to apply repeated tokens",
    )
    max_retries: Optional[int] = Field(
        default=DEFAULT_MAX_RETRIES,
        description="Maximum number of attempts to retry Triton client invocation before erroring",
    )
    timeout: Optional[float] = Field(
        default=DEFAULT_TIMEOUT,
        description="Maximum time (seconds) allowed for a Triton client call before erroring",
    )
    reuse_client: Optional[bool] = Field(
        default=DEFAULT_REUSE_CLIENT,
        description="True for reusing the same client instance between invocations",
    )
    triton_load_model_call: Optional[bool] = Field(
        default=DEFAULT_TRITON_LOAD_MODEL,
        description="True if a Triton load model API call should be made before using the client",
    )

    _client: Optional[GrpcTritonClient] = PrivateAttr()

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: float = DEFAULT_TOP_K,
        tokens: Optional[int] = DEFAULT_MAX_TOKENS,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        repetition_penalty: float = DEFAULT_REPTITION_PENALTY,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
        reuse_client: bool = DEFAULT_REUSE_CLIENT,
        triton_load_model_call: bool = DEFAULT_TRITON_LOAD_MODEL,
        callback_manager: Optional[CallbackManager] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        super().__init__(
            server_url=server_url,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            tokens=tokens,
            beam_width=beam_width,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            max_retries=max_retries,
            timeout=timeout,
            reuse_client=reuse_client,
            triton_load_model_call=triton_load_model_call,
            callback_manager=callback_manager,
            additional_kwargs=additional_kwargs,
            **kwargs,
        )

        try:
            self._client = GrpcTritonClient(server_url)
        except ImportError as err:
            raise ImportError(
                "Could not import triton client python package. "
                "Please install it with `pip install tritonclient`."
            ) from err

    @property
    def _get_model_default_parameters(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "beam_width": self.beam_width,
        }

    @property
    def _invocation_params(self, **kwargs: Any) -> Dict[str, Any]:
        return {**self._get_model_default_parameters, **kwargs}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get all the identifying parameters."""
        return {
            "server_url": self.server_url,
            "model_name": self.model_name,
        }

    def _get_client(self) -> Any:
        """Create or reuse a Triton client connection."""
        if not self.reuse_client:
            return GrpcTritonClient(self.server_url)

        if self._client is None:
            self._client = GrpcTritonClient(self.server_url)
        return self._client

    @property
    def metadata(self) -> LLMMetadata:
        """Gather and return metadata about the user Triton configured LLM model."""
        return LLMMetadata(
            model_name=self.model_name,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        chat_fn = completion_to_chat_decorator(self.complete)
        return chat_fn(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        chat_stream_fn = stream_completion_to_chat_decorator(self.stream_complete)
        return chat_stream_fn(messages, **kwargs)

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        from tritonclient.utils import InferenceServerException

        client = self._get_client()

        invocation_params = self._get_model_default_parameters
        invocation_params.update(kwargs)
        invocation_params["prompt"] = [[prompt]]
        model_params = self._identifying_params
        model_params.update(kwargs)
        request_id = str(random.randint(1, 9999999))  # nosec

        if self.triton_load_model_call:
            client.load_model(model_params["model_name"])

        result_queue = client.request_streaming(
            model_params["model_name"], request_id, **invocation_params
        )
        response = ""
        for token in result_queue:
            if isinstance(token, InferenceServerException):
                client.stop_stream(model_params["model_name"], request_id)
                raise token
            response += token

        return CompletionResponse(
            text=response,
        )

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        from tritonclient.utils import InferenceServerException

        client = self._get_client()

        invocation_params = self._get_model_default_parameters
        invocation_params.update(kwargs)
        invocation_params["prompt"] = [[prompt]]
        model_params = self._identifying_params
        model_params.update(kwargs)
        request_id = str(random.randint(1, 9999999))  # nosec

        if self.triton_load_model_call:
            client.load_model(model_params["model_name"])

        result_queue = client.request_streaming(
            model_params["model_name"], request_id, **invocation_params
        )

        def gen() -> CompletionResponseGen:
            text = ""
            for token in result_queue:
                if isinstance(token, InferenceServerException):
                    client.stop_stream(model_params["model_name"], request_id)
                    raise token
                text += token
                yield CompletionResponse(text=text, delta=token)

        return gen()

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError
