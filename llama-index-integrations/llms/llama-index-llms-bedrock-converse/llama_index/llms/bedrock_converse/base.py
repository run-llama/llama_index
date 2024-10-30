import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM, ToolSelection
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.bedrock_converse.utils import (
    bedrock_modelname_to_context_size,
    converse_with_retry,
    converse_with_retry_async,
    force_single_tool_call,
    is_bedrock_function_calling_model,
    join_two_dicts,
    messages_to_converse_messages,
    tools_to_converse_tools,
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


class BedrockConverse(FunctionCallingLLM):
    """
    Bedrock Converse LLM.

    Examples:
        `pip install llama-index-llms-bedrock-converse`

        ```python
        from llama_index.llms.bedrock_converse import BedrockConverse

        llm = BedrockConverse(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            aws_access_key_id="AWS Access Key ID to use",
            aws_secret_access_key="AWS Secret Access Key to use",
            aws_session_token="AWS Session Token to use",
            region_name="AWS Region to use, eg. us-east-1",
        )

        resp = llm.complete("Paul Graham is ")
        print(resp)
        ```
    """

    model: str = Field(description="The modelId of the Bedrock model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    profile_name: Optional[str] = Field(
        description="The name of aws profile to use. If not given, then the default profile is used."
    )
    aws_access_key_id: Optional[str] = Field(
        description="AWS Access Key ID to use", exclude=True
    )
    aws_secret_access_key: Optional[str] = Field(
        description="AWS Secret Access Key to use", exclude=True
    )
    aws_session_token: Optional[str] = Field(
        description="AWS Session Token to use", exclude=True
    )
    region_name: Optional[str] = Field(
        description="AWS region name to use. Uses region configured in AWS CLI if not passed",
        exclude=True,
    )
    botocore_session: Optional[Any] = Field(
        description="Use this Botocore session instead of creating a new default one.",
        exclude=True,
    )
    botocore_config: Optional[Any] = Field(
        description="Custom configuration object to use instead of the default generated one.",
        exclude=True,
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", gt=0
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout for the Bedrock API request in seconds. It will be used for both connect and read timeouts.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the bedrock invokeModel request.",
    )

    _config: Any = PrivateAttr()
    _client: Any = PrivateAttr()
    _asession: Any = PrivateAttr()

    def __init__(
        self,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 512,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        botocore_session: Optional[Any] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 10,
        botocore_config: Optional[Any] = None,
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

        session_kwargs = {
            "profile_name": profile_name,
            "region_name": region_name,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
            "botocore_session": botocore_session,
        }

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            model=model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            profile_name=profile_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            botocore_session=botocore_session,
            botocore_config=botocore_config,
        )

        self._config = None
        try:
            import boto3
            import aioboto3
            from botocore.config import Config

            self._config = (
                Config(
                    retries={"max_attempts": max_retries, "mode": "standard"},
                    connect_timeout=timeout,
                    read_timeout=timeout,
                )
                if botocore_config is None
                else botocore_config
            )
            session = boto3.Session(**session_kwargs)
            self._asession = aioboto3.Session(**session_kwargs)
        except ImportError:
            raise ImportError(
                "boto3 and/or aioboto3 package not found, install with"
                "'pip install boto3 aioboto3"
            )

        # Prior to general availability, custom boto3 wheel files were
        # distributed that used the bedrock service to invokeModel.
        # This check prevents any services still using those wheel files
        # from breaking
        if client is not None:
            self._client = client
        elif "bedrock-runtime" in session.get_available_services():
            self._client = session.client("bedrock-runtime", config=self._config)
        else:
            self._client = session.client("bedrock", config=self._config)

    @classmethod
    def class_name(cls) -> str:
        return "Bedrock_Converse_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=bedrock_modelname_to_context_size(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            is_function_calling_model=is_bedrock_function_calling_model(self.model),
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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

    def _get_content_and_tool_calls(
        self, response: Optional[Dict[str, Any]] = None, content: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any], List[str], List[str]]:
        assert (
            response is not None or content is not None
        ), f"Either response or content must be provided. Got response: {response}, content: {content}"
        assert (
            response is None or content is None
        ), f"Only one of response or content should be provided. Got response: {response}, content: {content}"
        tool_calls = []
        tool_call_ids = []
        status = []
        text_content = ""
        if content is not None:
            content_list = [content]
        else:
            content_list = response["output"]["message"]["content"]
        for content_block in content_list:
            if text := content_block.get("text", None):
                text_content += text
            if tool_usage := content_block.get("toolUse", None):
                tool_calls.append(tool_usage)
            if tool_result := content_block.get("toolResult", None):
                for tool_result_content in tool_result["content"]:
                    if text := tool_result_content.get("text", None):
                        text_content += text
                tool_call_ids.append(tool_result_content.get("toolUseId", ""))
                status.append(tool_result.get("status", ""))

        return text_content, tool_calls, tool_call_ids, status

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # convert Llama Index messages to AWS Bedrock Converse messages
        converse_messages, system_prompt = messages_to_converse_messages(messages)
        if len(system_prompt) > 0 or self.system_prompt is None:
            self.system_prompt = system_prompt
        all_kwargs = self._get_all_kwargs(**kwargs)

        # invoke LLM in AWS Bedrock Converse with retry
        response = converse_with_retry(
            client=self._client,
            messages=converse_messages,
            system_prompt=self.system_prompt,
            max_retries=self.max_retries,
            stream=False,
            **all_kwargs,
        )

        content, tool_calls, tool_call_ids, status = self._get_content_and_tool_calls(
            response
        )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs={
                    "tool_calls": tool_calls,
                    "tool_call_id": tool_call_ids,
                    "status": status,
                },
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        # convert Llama Index messages to AWS Bedrock Converse messages
        converse_messages, system_prompt = messages_to_converse_messages(messages)
        if len(system_prompt) > 0 or self.system_prompt is None:
            self.system_prompt = system_prompt
        all_kwargs = self._get_all_kwargs(**kwargs)

        # invoke LLM in AWS Bedrock Converse with retry
        response = converse_with_retry(
            client=self._client,
            messages=converse_messages,
            system_prompt=self.system_prompt,
            max_retries=self.max_retries,
            stream=True,
            **all_kwargs,
        )

        def gen() -> ChatResponseGen:
            content = {}
            role = MessageRole.ASSISTANT
            for chunk in response["stream"]:
                if content_block_delta := chunk.get("contentBlockDelta"):
                    content_delta = content_block_delta["delta"]
                    content = join_two_dicts(content, content_delta)
                    (
                        _,
                        tool_calls,
                        tool_call_ids,
                        status,
                    ) = self._get_content_and_tool_calls(content=content)

                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=content.get("text", ""),
                            additional_kwargs={
                                "tool_calls": tool_calls,
                                "tool_call_id": tool_call_ids,
                                "status": status,
                            },
                        ),
                        delta=content_delta.get("text", ""),
                        raw=response,
                    )
                elif content_block_start := chunk.get("contentBlockStart"):
                    tool_use = content_block_start["toolUse"]
                    content = join_two_dicts(content, tool_use)
                    (
                        _,
                        tool_calls,
                        tool_call_ids,
                        status,
                    ) = self._get_content_and_tool_calls(content=content)

                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=content.get("text", ""),
                            additional_kwargs={
                                "tool_calls": tool_calls,
                                "tool_call_id": tool_call_ids,
                                "status": status,
                            },
                        ),
                        raw=response,
                    )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        # convert Llama Index messages to AWS Bedrock Converse messages
        converse_messages, system_prompt = messages_to_converse_messages(messages)
        if len(system_prompt) > 0 or self.system_prompt is None:
            self.system_prompt = system_prompt
        all_kwargs = self._get_all_kwargs(**kwargs)

        # invoke LLM in AWS Bedrock Converse with retry
        response = await converse_with_retry_async(
            session=self._asession,
            config=self._config,
            messages=converse_messages,
            system_prompt=self.system_prompt,
            max_retries=self.max_retries,
            stream=False,
            **all_kwargs,
        )

        content, tool_calls, tool_call_ids, status = self._get_content_and_tool_calls(
            response
        )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs={
                    "tool_calls": tool_calls,
                    "tool_call_id": tool_call_ids,
                    "status": status,
                },
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = achat_to_completion_decorator(self.achat)
        return await complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        # convert Llama Index messages to AWS Bedrock Converse messages
        converse_messages, system_prompt = messages_to_converse_messages(messages)
        if len(system_prompt) > 0 or self.system_prompt is None:
            self.system_prompt = system_prompt
        all_kwargs = self._get_all_kwargs(**kwargs)

        # invoke LLM in AWS Bedrock Converse with retry
        response = await converse_with_retry_async(
            session=self._asession,
            config=self._config,
            messages=converse_messages,
            system_prompt=self.system_prompt,
            max_retries=self.max_retries,
            stream=True,
            **all_kwargs,
        )

        async def gen() -> ChatResponseAsyncGen:
            content = {}
            role = MessageRole.ASSISTANT
            async for chunk in response["stream"]:
                if content_block_delta := chunk.get("contentBlockDelta"):
                    content_delta = content_block_delta["delta"]
                    content = join_two_dicts(content, content_delta)
                    (
                        _,
                        tool_calls,
                        tool_call_ids,
                        status,
                    ) = self._get_content_and_tool_calls(content=content)

                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=content.get("text", ""),
                            additional_kwargs={
                                "tool_calls": tool_calls,
                                "tool_call_id": tool_call_ids,
                                "status": status,
                            },
                        ),
                        delta=content_delta.get("text", ""),
                        raw=response,
                    )
                elif content_block_start := chunk.get("contentBlockStart"):
                    tool_use = content_block_start["toolUse"]
                    content = join_two_dicts(content, tool_use)
                    (
                        _,
                        tool_calls,
                        tool_call_ids,
                        status,
                    ) = self._get_content_and_tool_calls(content=content)

                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=content.get("text", ""),
                            additional_kwargs={
                                "tool_calls": tool_calls,
                                "tool_call_id": tool_call_ids,
                                "status": status,
                            },
                        ),
                        raw=response,
                    )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Optional[dict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the arguments needed to let the LLM chat with tools."""
        chat_history = chat_history or []

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
            chat_history.append(user_msg)

        # convert Llama Index tools to AWS Bedrock Converse tools
        tool_config = tools_to_converse_tools(tools)
        if tool_choice:
            # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            # e.g. { "auto": {} }
            tool_config["toolChoice"] = tool_choice

        return {
            "messages": chat_history,
            "tools": tool_config,
            **kwargs,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: List["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Validate the response from chat_with_tools."""
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)

        return response

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            if (
                "input" not in tool_call
                or "toolUseId" not in tool_call
                or "name" not in tool_call
            ):
                raise ValueError("Invalid tool call.")
            argument_dict = (
                json.loads(tool_call["input"])
                if isinstance(tool_call["input"], str)
                else tool_call["input"]
            )

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call["toolUseId"],
                    tool_name=tool_call["name"],
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections
