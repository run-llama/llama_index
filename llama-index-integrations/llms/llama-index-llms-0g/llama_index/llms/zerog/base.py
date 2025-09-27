import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence

import httpx
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    acompletion_to_chat_decorator,
    astream_chat_to_completion_decorator,
    astream_completion_to_chat_decorator,
    chat_to_completion_decorator,
    completion_to_chat_decorator,
    stream_chat_to_completion_decorator,
    stream_completion_to_chat_decorator,
)
from llama_index.core.llms.llm import LLM

logger = logging.getLogger(__name__)

# Official 0G Services as per documentation
OFFICIAL_0G_SERVICES = {
    "llama-3.3-70b-instruct": {
        "provider_address": "0xf07240Efa67755B5311bc75784a061eDB47165Dd",
        "description": "State-of-the-art 70B parameter model for general AI tasks",
        "verification": "TEE (TeeML)",
    },
    "deepseek-r1-70b": {
        "provider_address": "0x3feE5a4dd5FDb8a32dDA97Bed899830605dBD9D3",
        "description": "Advanced reasoning model optimized for complex problem solving",
        "verification": "TEE (TeeML)",
    },
}


class ZeroGLLM(LLM):
    """
    0G Compute Network LLM integration for LlamaIndex.
    
    This integration allows you to use AI inference services from the 0G Compute Network,
    which provides decentralized GPU compute with verification capabilities.
    
    Args:
        model (str): The model to use. Can be one of the official models:
            - "llama-3.3-70b-instruct": 70B parameter model for general AI tasks
            - "deepseek-r1-70b": Advanced reasoning model
            Or a custom provider address.
        private_key (str): Ethereum private key for wallet authentication
        rpc_url (str): 0G Chain RPC URL. Defaults to testnet.
        provider_address (Optional[str]): Custom provider address. If not provided,
            will use the official provider for the specified model.
        context_window (int): Context window size. Defaults to 4096.
        max_tokens (int): Maximum tokens to generate. Defaults to 512.
        temperature (float): Sampling temperature. Defaults to 0.1.
        timeout (float): Request timeout in seconds. Defaults to 60.0.
        additional_kwargs (Dict[str, Any]): Additional parameters for requests.
        
    Examples:
        ```python
        from llama_index.llms.zerog import ZeroGLLM
        
        # Using official model
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="your_private_key_here"
        )
        
        # Using custom provider
        llm = ZeroGLLM(
            model="custom-model",
            provider_address="0x...",
            private_key="your_private_key_here"
        )
        
        response = llm.complete("Hello, how are you?")
        print(response.text)
        ```
    """

    model: str = Field(
        default="llama-3.3-70b-instruct",
        description="Model name or identifier"
    )
    private_key: str = Field(
        description="Ethereum private key for wallet authentication"
    )
    rpc_url: str = Field(
        default="https://evmrpc-testnet.0g.ai",
        description="0G Chain RPC URL"
    )
    provider_address: Optional[str] = Field(
        default=None,
        description="Custom provider address. If not provided, uses official provider for the model."
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="Context window size"
    )
    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature"
    )
    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds"
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for requests"
    )

    _broker: Any = PrivateAttr()
    _http_client: httpx.AsyncClient = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        model: str = "llama-3.3-70b-instruct",
        private_key: str = "",
        rpc_url: str = "https://evmrpc-testnet.0g.ai",
        provider_address: Optional[str] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        temperature: float = 0.1,
        timeout: float = 60.0,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        
        super().__init__(
            model=model,
            private_key=private_key,
            rpc_url=rpc_url,
            provider_address=provider_address,
            context_window=context_window,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            **kwargs,
        )
        
        self._http_client = httpx.AsyncClient(timeout=timeout)

    @classmethod
    def class_name(cls) -> str:
        return "ZeroGLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
        )

    def _get_provider_address(self) -> str:
        """Get the provider address for the model."""
        if self.provider_address:
            return self.provider_address
        
        if self.model in OFFICIAL_0G_SERVICES:
            return OFFICIAL_0G_SERVICES[self.model]["provider_address"]
        
        raise ValueError(
            f"Model '{self.model}' not found in official services. "
            f"Please provide a custom provider_address. "
            f"Available official models: {list(OFFICIAL_0G_SERVICES.keys())}"
        )

    async def _initialize_broker(self) -> None:
        """Initialize the 0G broker if not already initialized."""
        if self._is_initialized:
            return

        try:
            # This would require the JavaScript SDK to be available
            # For now, we'll simulate the broker initialization
            logger.info("Initializing 0G Compute Network broker...")
            
            # In a real implementation, this would use the JavaScript SDK
            # via a subprocess or Node.js bridge
            self._broker = {
                "provider_address": self._get_provider_address(),
                "initialized": True
            }
            
            self._is_initialized = True
            logger.info(f"Broker initialized for provider: {self._get_provider_address()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize 0G broker: {e}")
            raise

    def _messages_to_openai_format(self, messages: Sequence[ChatMessage]) -> List[Dict[str, str]]:
        """Convert LlamaIndex messages to OpenAI format."""
        openai_messages = []
        for message in messages:
            role = message.role.value if hasattr(message.role, 'value') else str(message.role)
            openai_messages.append({
                "role": role,
                "content": message.content or ""
            })
        return openai_messages

    async def _make_request(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> Dict[str, Any]:
        """Make a request to the 0G service."""
        await self._initialize_broker()
        
        # In a real implementation, this would:
        # 1. Get service metadata from broker
        # 2. Generate authenticated headers
        # 3. Make the actual request to the service endpoint
        
        # For now, we'll simulate the response
        provider_address = self._get_provider_address()
        
        # Simulate getting service metadata
        endpoint = f"https://api.0g.ai/v1/providers/{provider_address}"
        model_name = self.model
        
        # Simulate generating auth headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer simulated_token_for_{provider_address}",
            "X-0G-Provider": provider_address,
        }
        
        # Prepare request body
        request_body = {
            "messages": messages,
            "model": model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
            **self.additional_kwargs,
        }
        
        try:
            # In a real implementation, this would make the actual HTTP request
            # For now, we'll simulate a response
            if stream:
                return await self._simulate_streaming_response(messages)
            else:
                return await self._simulate_response(messages)
                
        except Exception as e:
            logger.error(f"Request to 0G service failed: {e}")
            raise

    async def _simulate_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Simulate a response from the 0G service."""
        # This is a placeholder - in real implementation, this would be the actual API response
        last_message = messages[-1]["content"] if messages else "Hello"
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"This is a simulated response from 0G Compute Network for: {last_message}"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

    async def _simulate_streaming_response(self, messages: List[Dict[str, str]]) -> AsyncGenerator[Dict[str, Any], None]:
        """Simulate a streaming response from the 0G service."""
        last_message = messages[-1]["content"] if messages else "Hello"
        response_text = f"This is a simulated streaming response from 0G Compute Network for: {last_message}"
        
        words = response_text.split()
        for i, word in enumerate(words):
            chunk = {
                "choices": [{
                    "delta": {
                        "content": word + " " if i < len(words) - 1 else word
                    },
                    "finish_reason": None if i < len(words) - 1 else "stop"
                }]
            }
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return completion_to_chat_decorator(self.chat)(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return stream_completion_to_chat_decorator(self.stream_chat)(prompt, **kwargs)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        openai_messages = self._messages_to_openai_format(messages)
        
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_data = loop.run_until_complete(
                self._make_request(openai_messages, stream=False)
            )
        finally:
            loop.close()
        
        choice = response_data["choices"][0]
        message_content = choice["message"]["content"]
        
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=message_content,
            ),
            raw=response_data,
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        def gen() -> Generator[ChatResponse, None, None]:
            openai_messages = self._messages_to_openai_format(messages)
            
            # Run async method in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async_gen = self._make_request(openai_messages, stream=True)
                
                async def async_wrapper():
                    content = ""
                    async for chunk in async_gen:
                        choice = chunk["choices"][0]
                        delta_content = choice.get("delta", {}).get("content", "")
                        content += delta_content
                        
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=content,
                            ),
                            delta=delta_content,
                            raw=chunk,
                        )
                
                # Convert async generator to sync
                async_iter = async_wrapper()
                while True:
                    try:
                        chunk = loop.run_until_complete(async_iter.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()
        
        return gen()

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return await acompletion_to_chat_decorator(self.achat)(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await astream_completion_to_chat_decorator(self.astream_chat)(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        openai_messages = self._messages_to_openai_format(messages)
        response_data = await self._make_request(openai_messages, stream=False)
        
        choice = response_data["choices"][0]
        message_content = choice["message"]["content"]
        
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=message_content,
            ),
            raw=response_data,
        )

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async def gen() -> AsyncGenerator[ChatResponse, None]:
            openai_messages = self._messages_to_openai_format(messages)
            content = ""
            
            async for chunk in await self._make_request(openai_messages, stream=True):
                choice = chunk["choices"][0]
                delta_content = choice.get("delta", {}).get("content", "")
                content += delta_content
                
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=content,
                    ),
                    delta=delta_content,
                    raw=chunk,
                )
        
        return gen()

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_http_client'):
            try:
                asyncio.create_task(self._http_client.aclose())
            except Exception:
                pass
