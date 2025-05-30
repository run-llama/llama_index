import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

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
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
import uuid
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.cohere.utils import (
    CHAT_MODELS,
    _get_message_cohere_format,
    _message_to_cohere_tool_results,
    _messages_to_cohere_tool_results_curr_chat_turn,
    acompletion_with_retry,
    cohere_modelname_to_contextsize,
    completion_with_retry,
    is_cohere_function_calling_model,
    remove_documents_from_messages,
    format_to_cohere_tools,
)
from llama_index.core.tools.types import BaseTool
import cohere
from cohere.types import (
    ToolCall,
)


class Cohere(FunctionCallingLLM):
    """
    Cohere LLM.

    Examples:
        `pip install llama-index-llms-cohere`

        ```python
        from llama_index.llms.cohere import Cohere

        llm = Cohere(model="command", api_key=api_key)
        resp = llm.complete("Paul Graham is ")
        print(resp)
        ```

    """

    model: str = Field(description="The cohere model to use.")
    temperature: Optional[float] = Field(
        description="The temperature to use for sampling.", default=None
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Cohere API."
    )
    max_tokens: int = Field(description="The maximum number of tokens to generate.")

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "command-r",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 8192,
        timeout: Optional[float] = None,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            temperature=temperature,
            additional_kwargs=additional_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            model=model,
            callback_manager=callback_manager,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )
        self._client = cohere.Client(api_key, client_name="llama_index")
        self._aclient = cohere.AsyncClient(api_key, client_name="llama_index")

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Cohere_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=cohere_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            system_role=MessageRole.CHATBOT,
            is_function_calling_model=is_cohere_function_calling_model(self.model),
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the chat with tools."""
        chat_history = chat_history or []

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        if user_msg is not None:
            chat_history.append(user_msg)

        tools_cohere_format = format_to_cohere_tools(tools)
        return {
            "messages": chat_history,
            "tools": tools_cohere_format or [],
            # switch to tool_choice on V2
            **({"force_single_step": True} if tool_required else {}),
            **kwargs,
        }

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = False,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls: List[ToolCall] = (
            response.message.additional_kwargs.get("tool_calls", []) or []
        )

        if len(tool_calls) < 1 and error_on_no_tool_call:
            raise ValueError(
                f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
            )

        tool_selections = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, ToolCall):
                raise ValueError("Invalid tool_call object")
            tool_selections.append(
                ToolSelection(
                    tool_id=uuid.uuid4().hex[:],
                    tool_name=tool_call.name,
                    tool_kwargs=tool_call.parameters,
                )
            )

        return tool_selections

    def get_cohere_chat_request(
        self,
        messages: List[ChatMessage],
        *,
        connectors: Optional[List[Dict[str, str]]] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Get the request for the Cohere chat API.

        Args:
            messages: The messages.
            connectors: The connectors.
            **kwargs: The keyword arguments.

        Returns:
            The request for the Cohere chat API.

        """
        additional_kwargs = messages[-1].additional_kwargs

        # cohere SDK will fail loudly if both connectors and documents are provided
        if additional_kwargs.get("documents", []) and documents and len(documents) > 0:
            raise ValueError(
                "Received documents both as a keyword argument and as an prompt additional keyword argument. Please choose only one option."
            )

        messages, documents = remove_documents_from_messages(messages)

        tool_results: Optional[List[Dict[str, Any]]] = (
            _messages_to_cohere_tool_results_curr_chat_turn(messages)
            or kwargs.get("tool_results")
        )
        if not tool_results:
            tool_results = None

        chat_history = []
        temp_tool_results = []
        # if force_single_step is set to False, then only message is empty in request if there is tool call
        if not kwargs.get("force_single_step"):
            for i, message in enumerate(messages[:-1]):
                # If there are multiple tool messages, then we need to aggregate them into one single tool message to pass into chat history
                if message.role == MessageRole.TOOL:
                    temp_tool_results += _message_to_cohere_tool_results(messages, i)

                    if (i == len(messages) - 1) or messages[
                        i + 1
                    ].role != MessageRole.TOOL:
                        cohere_message = _get_message_cohere_format(
                            message, temp_tool_results
                        )
                        chat_history.append(cohere_message)
                        temp_tool_results = []
                else:
                    chat_history.append(_get_message_cohere_format(message, None))

            message_str = "" if tool_results else messages[-1].content

        else:
            message_str = ""
            # if force_single_step is set to True, then message is the last human message in the conversation
            for message in messages[:-1]:
                if message.role in (
                    MessageRole.CHATBOT,
                    MessageRole.ASSISTANT,
                ) and message.additional_kwargs.get("tool_calls"):
                    continue

                # If there are multiple tool messages, then we need to aggregate them into one single tool message to pass into chat history
                if message.role == MessageRole.TOOL:
                    temp_tool_results += _message_to_cohere_tool_results(messages, i)

                    if (i == len(messages) - 1) or messages[
                        i + 1
                    ].role != MessageRole.TOOL:
                        cohere_message = _get_message_cohere_format(
                            message, temp_tool_results
                        )
                        chat_history.append(cohere_message)
                        temp_tool_results = []
                else:
                    chat_history.append(_get_message_cohere_format(message, None))
            # Add the last human message in the conversation to the message string
            for message in messages[::-1]:
                if (message.role == MessageRole.USER) and (message.content):
                    message_str = message.content
                    break

        req = {
            "message": message_str,
            "chat_history": chat_history,
            "tool_results": tool_results,
            "documents": documents,
            "connectors": connectors,
            "stop_sequences": stop_sequences,
            **kwargs,
        }
        return {k: v for k, v in req.items() if v is not None}

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        chat_request = self.get_cohere_chat_request(messages=messages, **all_kwargs)

        if all_kwargs["model"] not in CHAT_MODELS:
            raise ValueError(f"{all_kwargs['model']} not supported for chat")

        if "stream" in all_kwargs:
            warnings.warn(
                "Parameter `stream` is not supported by the `chat` method."
                "Use the `stream_chat` method instead"
            )

        response = completion_with_retry(
            client=self._client, max_retries=self.max_retries, chat=True, **chat_request
        )
        if not isinstance(response, cohere.NonStreamedChatResponse):
            tool_calls = response.get("tool_calls")
            content = response.get("text")
            response_raw = response

        else:
            tool_calls = response.tool_calls
            content = response.text
            response_raw = response.__dict__

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=response_raw,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        if "stream" in all_kwargs:
            warnings.warn(
                "Parameter `stream` is not supported by the `chat` method."
                "Use the `stream_chat` method instead"
            )

        response = completion_with_retry(
            client=self._client,
            max_retries=self.max_retries,
            chat=False,
            prompt=prompt,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.generations[0].text,
            raw=response.__dict__,
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        all_kwargs["stream"] = True
        if all_kwargs["model"] not in CHAT_MODELS:
            raise ValueError(f"{all_kwargs['model']} not supported for chat")

        chat_request = self.get_cohere_chat_request(messages=messages, **all_kwargs)

        response = completion_with_retry(
            client=self._client, max_retries=self.max_retries, chat=True, **chat_request
        )

        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for r in response:
                if "text" in r.__dict__:
                    content_delta = r.text
                else:
                    content_delta = ""
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=r.__dict__,
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        all_kwargs["stream"] = True

        response = completion_with_retry(
            client=self._client,
            max_retries=self.max_retries,
            chat=False,
            prompt=prompt,
            **all_kwargs,
        )

        def gen() -> CompletionResponseGen:
            content = ""
            for r in response:
                content_delta = r.text
                content += content_delta
                yield CompletionResponse(
                    text=content, delta=content_delta, raw=r._asdict()
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        if all_kwargs["model"] not in CHAT_MODELS:
            raise ValueError(f"{all_kwargs['model']} not supported for chat")
        if "stream" in all_kwargs:
            warnings.warn(
                "Parameter `stream` is not supported by the `chat` method."
                "Use the `stream_chat` method instead"
            )

        chat_request = self.get_cohere_chat_request(messages=messages, **all_kwargs)

        response = await acompletion_with_retry(
            aclient=self._aclient,
            max_retries=self.max_retries,
            chat=True,
            **chat_request,
        )

        if not isinstance(response, cohere.NonStreamedChatResponse):
            tool_calls = response.get("tool_calls")
            content = response.get("text")
            response_raw = response

        else:
            tool_calls = response.tool_calls
            content = response.text
            response_raw = response.__dict__

        if not isinstance(response, cohere.NonStreamedChatResponse):
            tool_calls = response.get("tool_calls")
            content = response.get("text")
            response_raw = response

        else:
            tool_calls = response.tool_calls
            content = response.text
            response_raw = response.__dict__

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=response_raw,
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        if "stream" in all_kwargs:
            warnings.warn(
                "Parameter `stream` is not supported by the `chat` method."
                "Use the `stream_chat` method instead"
            )

        response = await acompletion_with_retry(
            aclient=self._aclient,
            max_retries=self.max_retries,
            chat=False,
            prompt=prompt,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.generations[0].text,
            raw=response.__dict__,
        )

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        all_kwargs["stream"] = True
        if all_kwargs["model"] not in CHAT_MODELS:
            raise ValueError(f"{all_kwargs['model']} not supported for chat")

        chat_request = self.get_cohere_chat_request(messages, **all_kwargs)

        response = completion_with_retry(
            client=self._client, max_retries=self.max_retries, chat=True, **chat_request
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            role = MessageRole.ASSISTANT
            async for r in response:
                if "text" in r.__dict__:
                    content_delta = r.text
                else:
                    content_delta = ""
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=r.__dict__,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        all_kwargs["stream"] = True

        response = await acompletion_with_retry(
            aclient=self._aclient,
            max_retries=self.max_retries,
            chat=False,
            prompt=prompt,
            **all_kwargs,
        )

        async def gen() -> CompletionResponseAsyncGen:
            content = ""
            async for r in response:
                content_delta = r.text
                content += content_delta
                yield CompletionResponse(
                    text=content, delta=content_delta, raw=r._asdict()
                )

        return gen()
