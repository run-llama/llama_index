import deprecated
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Optional,
    Sequence,
    Union,
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
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.llms.function_calling import FunctionCallingLLM, ToolSelection
from llama_index.core.utilities.gemini_utils import merge_neighboring_same_role_messages
from llama_index.llms.vertex.gemini_utils import create_gemini_client, is_gemini_model
from llama_index.llms.vertex.utils import (
    CHAT_MODELS,
    CODE_CHAT_MODELS,
    CODE_MODELS,
    TEXT_MODELS,
    _parse_chat_history,
    _parse_examples,
    _parse_message,
    acompletion_with_retry,
    completion_with_retry,
    init_vertexai,
    force_single_tool_call,
)
from vertexai.generative_models._generative_models import SafetySettingsType
from vertexai.generative_models import ToolConfig

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


@deprecated.deprecated(
    reason=(
        "Should use `llama-index-llms-google-genai` instead, using Google's latest unified SDK. "
        "See: https://docs.llamaindex.ai/en/stable/examples/llm/google_genai/"
    )
)
class Vertex(FunctionCallingLLM):
    """
    Vertext LLM.

    Examples:
        `pip install llama-index-llms-vertex`

        ```python
        from llama_index.llms.vertex import Vertex

        # Set up necessary variables
        credentials = {
            "project_id": "INSERT_PROJECT_ID",
            "api_key": "INSERT_API_KEY",
        }

        # Create an instance of the Vertex class
        llm = Vertex(
            model="text-bison",
            project=credentials["project_id"],
            credentials=credentials,
            context_window=4096,
        )

        # Access the complete method from the instance
        response = llm.complete("Hello world!")
        print(str(response))
        ```

    """

    model: str = Field(description="The vertex model to use.")
    temperature: float = Field(description="The temperature to use for sampling.")
    context_window: int = Field(
        default=4096, description="The context window to use for sampling."
    )
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    examples: Optional[Sequence[ChatMessage]] = Field(
        description="Example messages for the chat model."
    )
    max_retries: int = Field(default=10, description="The maximum number of retries.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Vertex."
    )
    iscode: bool = Field(
        default=False, description="Flag to determine if current model is a Code Model"
    )
    _is_gemini: bool = PrivateAttr()
    _is_chat_model: bool = PrivateAttr()
    _client: Any = PrivateAttr()
    _chat_client: Any = PrivateAttr()
    _safety_settings: Dict[str, Any] = PrivateAttr()

    def __init__(
        self,
        model: str = "text-bison",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[Any] = None,
        examples: Optional[Sequence[ChatMessage]] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        context_window: int = 4096,
        max_retries: int = 10,
        iscode: bool = False,
        safety_settings: Optional[SafetySettingsType] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        init_vertexai(project=project, location=location, credentials=credentials)

        safety_settings = safety_settings or {}
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            temperature=temperature,
            context_window=context_window,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            model=model,
            examples=examples,
            iscode=iscode,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._safety_settings = safety_settings
        self._is_gemini = False
        self._is_chat_model = False
        if model in CHAT_MODELS:
            from vertexai.language_models import ChatModel

            self._chat_client = ChatModel.from_pretrained(model)
            self._is_chat_model = True
        elif model in CODE_CHAT_MODELS:
            from vertexai.language_models import CodeChatModel

            self._chat_client = CodeChatModel.from_pretrained(model)
            iscode = True
            self._is_chat_model = True
        elif model in CODE_MODELS:
            from vertexai.language_models import CodeGenerationModel

            self._client = CodeGenerationModel.from_pretrained(model)
            iscode = True
        elif model in TEXT_MODELS:
            from vertexai.language_models import TextGenerationModel

            self._client = TextGenerationModel.from_pretrained(model)
        elif is_gemini_model(model):
            self._client = create_gemini_client(model, self._safety_settings)
            self._chat_client = self._client
            self._is_gemini = True
            self._is_chat_model = True
        else:
            raise (ValueError(f"Model {model} not found, please verify the model name"))

    @classmethod
    def class_name(cls) -> str:
        return "Vertex"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            num_output=self.max_tokens,
            context_window=self.context_window,
            is_chat_model=self._is_chat_model,
            is_function_calling_model=self._is_gemini,
            model_name=self.model,
            system_role=(
                MessageRole.USER if self._is_gemini else MessageRole.SYSTEM
            ),  # Gemini does not support the default: MessageRole.SYSTEM
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
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

    def _get_content_and_tool_calls(self, response: Any) -> Tuple[str, List]:
        tool_calls = []
        if response.candidates[0].function_calls:
            for tool_call in response.candidates[0].function_calls:
                tool_calls.append(tool_call)
        try:
            content = response.text
        except Exception:
            content = ""
        return content, tool_calls

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        merged_messages = (
            merge_neighboring_same_role_messages(messages)
            if self._is_gemini
            else messages
        )
        question = _parse_message(merged_messages[-1], self._is_gemini)
        chat_history = _parse_chat_history(merged_messages[:-1], self._is_gemini)

        chat_params = {**chat_history}

        kwargs = kwargs if kwargs else {}

        params = {**self._model_kwargs, **kwargs}

        if self.iscode and "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the codey model's"))
        if self.examples and "examples" not in params:
            chat_params["examples"] = _parse_examples(self.examples)
        elif "examples" in params:
            raise (
                ValueError(
                    "examples are not supported in chat generation pass them as a constructor parameter"
                )
            )

        generation = completion_with_retry(
            client=self._chat_client,
            prompt=question,
            chat=True,
            stream=False,
            is_gemini=self._is_gemini,
            params=chat_params,
            max_retries=self.max_retries,
            **params,
        )

        content, tool_calls = self._get_content_and_tool_calls(generation)

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=generation.__dict__,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        if self.iscode and "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the codey model's"))

        completion = completion_with_retry(
            self._client,
            prompt,
            max_retries=self.max_retries,
            is_gemini=self._is_gemini,
            **params,
        )
        return CompletionResponse(text=completion.text, raw=completion.__dict__)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        merged_messages = (
            merge_neighboring_same_role_messages(messages)
            if self._is_gemini
            else messages
        )
        question = _parse_message(merged_messages[-1], self._is_gemini)
        chat_history = _parse_chat_history(merged_messages[:-1], self._is_gemini)
        chat_params = {**chat_history}
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        if self.iscode and "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the codey model's"))
        if self.examples and "examples" not in params:
            chat_params["examples"] = _parse_examples(self.examples)
        elif "examples" in params:
            raise (
                ValueError(
                    "examples are not supported in chat generation pass them as a constructor parameter"
                )
            )

        response = completion_with_retry(
            client=self._chat_client,
            prompt=question,
            chat=True,
            stream=True,
            is_gemini=self._is_gemini,
            params=chat_params,
            max_retries=self.max_retries,
            **params,
        )

        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for r in response:
                content_delta = r.text
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
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        if "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the streaming"))

        completion = completion_with_retry(
            client=self._client,
            prompt=prompt,
            stream=True,
            is_gemini=self._is_gemini,
            max_retries=self.max_retries,
            **params,
        )

        def gen() -> CompletionResponseGen:
            content = ""
            for r in completion:
                content_delta = r.text
                content += content_delta
                yield CompletionResponse(
                    text=content, delta=content_delta, raw=r.__dict__
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        merged_messages = (
            merge_neighboring_same_role_messages(messages)
            if self._is_gemini
            else messages
        )
        question = _parse_message(merged_messages[-1], self._is_gemini)
        chat_history = _parse_chat_history(merged_messages[:-1], self._is_gemini)
        chat_params = {**chat_history}
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        if self.iscode and "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the codey model's"))
        if self.examples and "examples" not in params:
            chat_params["examples"] = _parse_examples(self.examples)
        elif "examples" in params:
            raise (
                ValueError(
                    "examples are not supported in chat generation pass them as a constructor parameter"
                )
            )
        generation = await acompletion_with_retry(
            client=self._chat_client,
            prompt=question,
            chat=True,
            is_gemini=self._is_gemini,
            params=chat_params,
            max_retries=self.max_retries,
            **params,
        )
        ##this is due to a bug in vertex AI we have to await twice
        if self.iscode:
            generation = await generation

        content, tool_calls = self._get_content_and_tool_calls(generation)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=generation.__dict__,
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        if self.iscode and "candidate_count" in params:
            raise (ValueError("candidate_count is not supported by the codey model's"))
        completion = await acompletion_with_retry(
            client=self._client,
            prompt=prompt,
            max_retries=self.max_retries,
            is_gemini=self._is_gemini,
            **params,
        )
        return CompletionResponse(text=completion.text)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise (ValueError("Not Implemented"))

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,  # theoretically supported, but not implemented
        tool_required: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the arguments needed to let the LLM chat with tools."""
        chat_history = chat_history or []

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
            chat_history.append(user_msg)

        tool_dicts = []
        for tool in tools:
            tool_dicts.append(
                {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.get_parameters_dict(),
                }
            )

        tool_config = (
            {"tool_config": self._to_function_calling_config(tool_required)}
            if self._is_gemini
            else {}
        )

        print("tool_config", tool_config)
        return {
            "messages": chat_history,
            "tools": tool_dicts or None,
            **tool_config,
            **kwargs,
        }

    def _to_function_calling_config(self, tool_required: bool) -> ToolConfig:
        return ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY
                if tool_required
                else ToolConfig.FunctionCallingConfig.Mode.AUTO,
                allowed_function_names=None,
            )
        )

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
            response_dict = tool_call.to_dict()
            if "args" not in response_dict or "name" not in response_dict:
                raise ValueError("Invalid tool call.")
            argument_dict = response_dict["args"]

            tool_selections.append(
                ToolSelection(
                    tool_id="None",
                    tool_name=tool_call.name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections
