import json
import logging
from huggingface_hub import AsyncInferenceClient, InferenceClient, model_info
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.inference._generated.types import (
    ChatCompletionOutput,
    ChatCompletionStreamOutput,
    ChatCompletionOutputToolCall,
    ChatCompletionOutputFunctionDefinition,
)
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
    astream_completion_response_to_chat_response,
)
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
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import BaseTool


logger = logging.getLogger(__name__)


class HuggingFaceInferenceAPI(FunctionCallingLLM):
    """
    Wrapper on the Hugging Face's Inference API.

    Overview of the design:
    - Synchronous uses InferenceClient, asynchronous uses AsyncInferenceClient
    - chat uses the conversational task: https://huggingface.co/tasks/conversational
    - complete uses the text generation task: https://huggingface.co/tasks/text-generation

    Note: some models that support the text generation task can leverage Hugging
    Face's optimized deployment toolkit called text-generation-inference (TGI).
    Use InferenceClient.get_model_status to check if TGI is being used.

    Relevant links:
    - General Docs: https://huggingface.co/docs/api-inference/index
    - API Docs: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client
    - Source: https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub/inference
    """

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceInferenceAPI"

    model: Optional[str] = Field(
        default=None,
        description=(
            "The model to run inference with. Can be a model id hosted on the Hugging"
            " Face Hub, e.g. bigcode/starcoder or a URL to a deployed Inference"
            " Endpoint. Defaults to None, in which case a recommended model is"
            " automatically selected for the task (see Field below)."
        ),
    )

    # TODO: deprecate this field
    model_name: Optional[str] = Field(
        default=None,
        description=(
            "The model to run inference with. Can be a model id hosted on the Hugging"
            " Face Hub, e.g. bigcode/starcoder or a URL to a deployed Inference"
            " Endpoint. Defaults to None, in which case a recommended model is"
            " automatically selected for the task (see Field below)."
        ),
    )
    token: Union[str, bool, None] = Field(
        default=None,
        description=(
            "Hugging Face token. Will default to the locally saved token. Pass "
            "token=False if you don’t want to send your token to the server."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description=(
            "The maximum number of seconds to wait for a response from the server."
            " Loading a new model in Inference API can take up to several minutes."
            " Defaults to None, meaning it will loop until the server is available."
        ),
    )
    headers: Dict[str, str] = Field(
        default=None,
        description=(
            "Additional headers to send to the server. By default only the"
            " authorization and user-agent headers are sent. Values in this dictionary"
            " will override the default values."
        ),
    )
    cookies: Dict[str, str] = Field(
        default=None, description="Additional cookies to send to the server."
    )
    task: Optional[str] = Field(
        default=None,
        description=(
            "Optional task to pick Hugging Face's recommended model, used when"
            " model_name is left as default of None."
        ),
    )

    _sync_client: InferenceClient = PrivateAttr()
    _async_client: AsyncInferenceClient = PrivateAttr()
    _get_model_info: Callable[..., ModelInfo] = PrivateAttr()

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            LLMMetadata.model_fields["context_window"].description
            + " This may be looked up in a model's `config.json`."
        ),
    )
    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description=LLMMetadata.model_fields["num_output"].description,
    )
    temperature: float = Field(
        default=0.1,
        description="The temperature to use for the model.",
        gt=0.0,
    )
    is_chat_model: bool = Field(
        default=True,
        description="Controls whether the chat or text generation methods are used.",
    )
    is_function_calling_model: bool = Field(
        default=False,
        description="Controls whether the function calling methods are used.",
    )

    def __init__(self, **kwargs: Any) -> None:
        model_name = kwargs.get("model_name") or kwargs.get("model")
        if model_name is None:
            task = kwargs.get("task", "")
            # NOTE: task being None or empty string leads to ValueError,
            # which ensures model is present
            kwargs["model_name"] = InferenceClient.get_recommended_model(task=task)
            logger.debug(
                f"Using Hugging Face's recommended model {kwargs['model_name']}"
                f" given task {task}."
            )

        if kwargs.get("task") is None:
            task = "conversational"
        else:
            task = kwargs["task"].lower()

        if kwargs.get("is_function_calling_model", False):
            print(
                "Function calling is currently not supported for Hugging Face Inference API, setting is_function_calling_model to False"
            )
            kwargs["is_function_calling_model"] = False

        super().__init__(**kwargs)  # Populate pydantic Fields
        self._sync_client = InferenceClient(**self._get_inference_client_kwargs())
        self._async_client = AsyncInferenceClient(**self._get_inference_client_kwargs())

        # set context window if not provided, if we can get the endpoint info
        try:
            info = self._sync_client.get_endpoint_info()
            if "max_input_tokens" in info and kwargs.get("context_window") is None:
                self.context_window = info["max_input_tokens"]
        except Exception:
            pass

    def _get_inference_client_kwargs(self) -> Dict[str, Any]:
        """Extract the Hugging Face InferenceClient construction parameters."""
        return {
            "model": self.model_name or self.model,
            "token": self.token,
            "timeout": self.timeout,
            "headers": self.headers,
            "cookies": self.cookies,
        }

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model_name or self.model,
            "max_tokens": self.num_output,
            "temperature": self.temperature,
        }
        return {**base_kwargs, **kwargs}

    def _to_huggingface_messages(
        self, messages: Sequence[ChatMessage]
    ) -> List[Dict[str, Any]]:
        hf_dicts = []
        for m in messages:
            hf_dicts.append(
                {"role": m.role.value, "content": m.content if m.content else ""}
            )
            if m.additional_kwargs.get("tool_calls", []):
                tool_call_dicts = []
                for tool_call in m.additional_kwargs["tool_calls"]:
                    function_dict = {
                        "name": tool_call.id,
                        "arguments": tool_call.function.arguments,
                    }
                    tool_call_dicts.append(
                        {"type": "function", "function": function_dict}
                    )

                hf_dicts[-1]["tool_calls"] = tool_call_dicts

            if m.role == MessageRole.TOOL:
                hf_dicts[-1]["name"] = m.additional_kwargs.get("tool_call_id")

        return hf_dicts

    def _parse_streaming_tool_calls(
        self, tool_call_strs: List[str]
    ) -> List[Union[ToolSelection, str]]:
        tool_calls = []
        # Try to parse into complete objects, otherwise keep as strings
        for tool_call_str in tool_call_strs:
            try:
                tool_call_dict = json.loads(tool_call_str)
                args = tool_call_dict["function"]
                name = args.pop("_name")
                tool_calls.append(
                    ChatCompletionOutputToolCall(
                        id=name,
                        type="function",
                        function=ChatCompletionOutputFunctionDefinition(
                            arguments=args,
                            name=name,
                        ),
                    )
                )
            except Exception as e:
                tool_calls.append(tool_call_str)

        return tool_calls

    def get_model_info(self, **kwargs: Any) -> "ModelInfo":
        """Get metadata on the current model from Hugging Face."""
        model_name = self.model_name or self.model
        return model_info(model_name, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model_name or self.model,
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self.task == "conversational" or self.task is None:
            model_kwargs = self._get_model_kwargs(**kwargs)

            output: ChatCompletionOutput = self._sync_client.chat_completion(
                messages=self._to_huggingface_messages(messages),
                **model_kwargs,
            )

            content = output.choices[0].message.content or ""
            tool_calls = output.choices[0].message.tool_calls or []
            additional_kwargs = {"tool_calls": tool_calls} if tool_calls else {}

            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=content,
                    additional_kwargs=additional_kwargs,
                ),
                raw=output,
            )
        else:
            # try and use text generation
            prompt = self.messages_to_prompt(messages)
            completion = self.complete(prompt, formatted=True, **kwargs)
            return completion_response_to_chat_response(completion)

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        model_kwargs = self._get_model_kwargs(**kwargs)
        model_kwargs["max_new_tokens"] = model_kwargs["max_tokens"]
        del model_kwargs["max_tokens"]

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        return CompletionResponse(
            text=self._sync_client.text_generation(prompt, **model_kwargs)
        )

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if self.task == "conversational" or self.task is None:
            model_kwargs = self._get_model_kwargs(**kwargs)

            def gen() -> ChatResponseGen:
                response = ""
                tool_call_strs = []
                cur_index = -1
                for chunk in self._sync_client.chat_completion(
                    messages=self._to_huggingface_messages(messages),
                    stream=True,
                    **model_kwargs,
                ):
                    chunk: ChatCompletionStreamOutput = chunk

                    delta = chunk.choices[0].delta.content or ""
                    response += delta
                    tool_call_delta = chunk.choices[0].delta.tool_calls
                    if tool_call_delta:
                        if tool_call_delta.index != cur_index:
                            cur_index = tool_call_delta.index
                            tool_call_strs.append(tool_call_delta.function.arguments)
                        else:
                            tool_call_strs[
                                cur_index
                            ] += tool_call_delta.function.arguments

                    tool_calls = self._parse_streaming_tool_calls(tool_call_strs)
                    additional_kwargs = {"tool_calls": tool_calls} if tool_calls else {}
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=response,
                            additional_kwargs=additional_kwargs,
                        ),
                        delta=delta,
                        raw=chunk,
                    )

            return gen()
        else:
            prompt = self.messages_to_prompt(messages)
            completion_stream = self.stream_complete(prompt, formatted=True, **kwargs)
            return stream_completion_response_to_chat_response(completion_stream)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        model_kwargs = self._get_model_kwargs(**kwargs)
        model_kwargs["max_new_tokens"] = model_kwargs["max_tokens"]
        del model_kwargs["max_tokens"]

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        def gen() -> CompletionResponseGen:
            response = ""
            for delta in self._sync_client.text_generation(
                prompt, stream=True, **model_kwargs
            ):
                response += delta
                yield CompletionResponse(text=response, delta=delta)

        return gen()

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        if self.task == "conversational" or self.task is None:
            model_kwargs = self._get_model_kwargs(**kwargs)

            output: ChatCompletionOutput = await self._async_client.chat_completion(
                messages=self._to_huggingface_messages(messages),
                **model_kwargs,
            )

            content = output.choices[0].message.content or ""
            tool_calls = output.choices[0].message.tool_calls or []
            additional_kwargs = {"tool_calls": tool_calls} if tool_calls else {}

            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=content,
                    additional_kwargs=additional_kwargs,
                ),
                raw=output,
            )
        else:
            # try and use text generation
            prompt = self.messages_to_prompt(messages)
            completion = await self.acomplete(prompt, formatted=True, **kwargs)
            return completion_response_to_chat_response(completion)

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        model_kwargs = self._get_model_kwargs(**kwargs)
        model_kwargs["max_new_tokens"] = model_kwargs["max_tokens"]
        del model_kwargs["max_tokens"]

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        return CompletionResponse(
            text=await self._async_client.text_generation(prompt, **model_kwargs)
        )

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        if self.task == "conversational" or self.task is None:
            model_kwargs = self._get_model_kwargs(**kwargs)

            async def gen() -> ChatResponseAsyncGen:
                response = ""
                tool_call_strs = []
                cur_index = -1
                async for chunk in await self._async_client.chat_completion(
                    messages=self._to_huggingface_messages(messages),
                    stream=True,
                    **model_kwargs,
                ):
                    if chunk.choices[0].finish_reason is not None:
                        break

                    chunk: ChatCompletionStreamOutput = chunk

                    delta = chunk.choices[0].delta.content or ""
                    response += delta
                    tool_call_delta = chunk.choices[0].delta.tool_calls
                    if tool_call_delta:
                        if tool_call_delta.index != cur_index:
                            cur_index = tool_call_delta.index
                            tool_call_strs.append(tool_call_delta.function.arguments)
                        else:
                            tool_call_strs[
                                cur_index
                            ] += tool_call_delta.function.arguments

                    tool_calls = self._parse_streaming_tool_calls(tool_call_strs)

                    additional_kwargs = {"tool_calls": tool_calls} if tool_calls else {}

                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=response,
                            additional_kwargs=additional_kwargs,
                        ),
                        delta=delta,
                        raw=chunk,
                    )

                await self._async_client.close()

            return gen()
        else:
            prompt = self.messages_to_prompt(messages)
            completion_stream = await self.astream_complete(
                prompt, formatted=True, **kwargs
            )
            return astream_completion_response_to_chat_response(completion_stream)

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        model_kwargs = self._get_model_kwargs(**kwargs)
        model_kwargs["max_new_tokens"] = model_kwargs["max_tokens"]
        del model_kwargs["max_tokens"]

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        async def gen() -> CompletionResponseAsyncGen:
            response = ""
            async for delta in await self._async_client.text_generation(
                prompt, stream=True, **model_kwargs
            ):
                response += delta
                yield CompletionResponse(text=response, delta=delta)

            await self._async_client.close()

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: List["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Validate the response from chat_with_tools."""
        if not allow_parallel_tool_calls and response.message.additional_kwargs.get(
            "tool_calls", []
        ):
            response.additional_kwargs[
                "tool_calls"
            ] = response.message.additional_kwargs["tool_calls"][0]

        return response

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls: List[
            ChatCompletionOutputToolCall
        ] = response.message.additional_kwargs.get("tool_calls", [])
        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            # while streaming, tool_call is a string
            if isinstance(tool_call, str):
                continue

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_kwargs=tool_call.function.arguments,
                )
            )

        return tool_selections
