from collections import ChainMap
from typing import (
    Any,
    Dict,
    List,
    Generator,
    AsyncGenerator,
    Optional,
    Protocol,
    Sequence,
    Union,
    get_args,
    runtime_checkable,
    TYPE_CHECKING,
)
from typing_extensions import Annotated

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.base.query_pipeline.query import (
    InputKeys,
    OutputKeys,
    QueryComponent,
    StringableInput,
    validate_and_convert_stringable,
)
from llama_index.core.bridge.pydantic import (
    BaseModel,
    WithJsonSchema,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.base.llms.generic_utils import (
    prompt_to_messages,
)
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.types import (
    BaseOutputParser,
    PydanticProgramMode,
    TokenAsyncGen,
    TokenGen,
    Model,
)
from llama_index.core.instrumentation.events.llm import (
    LLMPredictEndEvent,
    LLMPredictStartEvent,
    LLMStructuredPredictInProgressEvent,
    LLMStructuredPredictEndEvent,
    LLMStructuredPredictStartEvent,
)

import llama_index.core.instrumentation as instrument
from llama_index.core.base.llms.types import (
    ChatMessage,
)

dispatcher = instrument.get_dispatcher(__name__)

if TYPE_CHECKING:
    from llama_index.core.chat_engine.types import AgentChatResponse
    from llama_index.core.tools.types import BaseTool
    from llama_index.core.llms.structured_llm import StructuredLLM


class ToolSelection(BaseModel):
    """Tool selection."""

    tool_id: str = Field(description="Tool ID to select.")
    tool_name: str = Field(description="Tool name to select.")
    tool_kwargs: Dict[str, Any] = Field(description="Keyword arguments for the tool.")
    # NOTE: no args for now


# NOTE: These two protocols are needed to appease mypy
@runtime_checkable
class MessagesToPromptType(Protocol):
    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        pass


@runtime_checkable
class CompletionToPromptType(Protocol):
    def __call__(self, prompt: str) -> str:
        pass


def stream_completion_response_to_tokens(
    completion_response_gen: CompletionResponseGen,
) -> TokenGen:
    """Convert a stream completion response to a stream of tokens."""

    def gen() -> TokenGen:
        for response in completion_response_gen:
            yield response.delta or ""

    return gen()


def stream_chat_response_to_tokens(
    chat_response_gen: ChatResponseGen,
) -> TokenGen:
    """Convert a stream completion response to a stream of tokens."""

    def gen() -> TokenGen:
        for response in chat_response_gen:
            yield response.delta or ""

    return gen()


async def astream_completion_response_to_tokens(
    completion_response_gen: CompletionResponseAsyncGen,
) -> TokenAsyncGen:
    """Convert a stream completion response to a stream of tokens."""

    async def gen() -> TokenAsyncGen:
        async for response in completion_response_gen:
            yield response.delta or ""

    return gen()


async def astream_chat_response_to_tokens(
    chat_response_gen: ChatResponseAsyncGen,
) -> TokenAsyncGen:
    """Convert a stream completion response to a stream of tokens."""

    async def gen() -> TokenAsyncGen:
        async for response in chat_response_gen:
            yield response.delta or ""

    return gen()


def default_completion_to_prompt(prompt: str) -> str:
    return prompt


MessagesToPromptCallable = Annotated[
    Optional[MessagesToPromptType],
    WithJsonSchema({"type": "string"}),
]


CompletionToPromptCallable = Annotated[
    Optional[CompletionToPromptType],
    WithJsonSchema({"type": "string"}),
]


class LLM(BaseLLM):
    """
    The LLM class is the main class for interacting with language models.

    Attributes:
        system_prompt (Optional[str]):
            System prompt for LLM calls.
        messages_to_prompt (Callable):
            Function to convert a list of messages to an LLM prompt.
        completion_to_prompt (Callable):
            Function to convert a completion to an LLM prompt.
        output_parser (Optional[BaseOutputParser]):
            Output parser to parse, validate, and correct errors programmatically.
        pydantic_program_mode (PydanticProgramMode):
            Pydantic program mode to use for structured prediction.
    """

    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for LLM calls."
    )
    messages_to_prompt: MessagesToPromptCallable = Field(
        description="Function to convert a list of messages to an LLM prompt.",
        default=None,
        exclude=True,
    )
    completion_to_prompt: CompletionToPromptCallable = Field(
        description="Function to convert a completion to an LLM prompt.",
        default=None,
        exclude=True,
    )
    output_parser: Optional[BaseOutputParser] = Field(
        description="Output parser to parse, validate, and correct errors programmatically.",
        default=None,
        exclude=True,
    )
    pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT

    # deprecated
    query_wrapper_prompt: Optional[BasePromptTemplate] = Field(
        description="Query wrapper prompt for LLM calls.",
        default=None,
        exclude=True,
    )

    # -- Pydantic Configs --

    @field_validator("messages_to_prompt")
    @classmethod
    def set_messages_to_prompt(
        cls, messages_to_prompt: Optional[MessagesToPromptType]
    ) -> MessagesToPromptType:
        return messages_to_prompt or generic_messages_to_prompt

    @field_validator("completion_to_prompt")
    @classmethod
    def set_completion_to_prompt(
        cls, completion_to_prompt: Optional[CompletionToPromptType]
    ) -> CompletionToPromptType:
        return completion_to_prompt or default_completion_to_prompt

    @model_validator(mode="after")
    def check_prompts(self) -> "LLM":
        if self.completion_to_prompt is None:
            self.completion_to_prompt = default_completion_to_prompt
        if self.messages_to_prompt is None:
            self.messages_to_prompt = generic_messages_to_prompt
        return self

    # -- Utils --

    def _log_template_data(
        self, prompt: BasePromptTemplate, **prompt_args: Any
    ) -> None:
        template_vars = {
            k: v
            for k, v in ChainMap(prompt.kwargs, prompt_args).items()
            if k in prompt.template_vars
        }
        with self.callback_manager.event(
            CBEventType.TEMPLATING,
            payload={
                EventPayload.TEMPLATE: prompt.get_template(llm=self),
                EventPayload.TEMPLATE_VARS: template_vars,
                EventPayload.SYSTEM_PROMPT: self.system_prompt,
                EventPayload.QUERY_WRAPPER_PROMPT: self.query_wrapper_prompt,
            },
        ):
            pass

    def _get_prompt(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        formatted_prompt = prompt.format(
            llm=self,
            messages_to_prompt=self.messages_to_prompt,
            completion_to_prompt=self.completion_to_prompt,
            **prompt_args,
        )
        if self.output_parser is not None:
            formatted_prompt = self.output_parser.format(formatted_prompt)
        return self._extend_prompt(formatted_prompt)

    def _get_messages(
        self, prompt: BasePromptTemplate, **prompt_args: Any
    ) -> List[ChatMessage]:
        messages = prompt.format_messages(llm=self, **prompt_args)
        if self.output_parser is not None:
            messages = self.output_parser.format_messages(messages)
        return self._extend_messages(messages)

    def _parse_output(self, output: str) -> str:
        if self.output_parser is not None:
            return str(self.output_parser.parse(output))

        return output

    def _extend_prompt(
        self,
        formatted_prompt: str,
    ) -> str:
        """Add system and query wrapper prompts to base prompt."""
        extended_prompt = formatted_prompt

        if self.system_prompt:
            extended_prompt = self.system_prompt + "\n\n" + extended_prompt

        if self.query_wrapper_prompt:
            extended_prompt = self.query_wrapper_prompt.format(
                query_str=extended_prompt
            )

        return extended_prompt

    def _extend_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Add system prompt to chat message list."""
        if self.system_prompt:
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                *messages,
            ]
        return messages

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """Return query component."""
        if self.metadata.is_chat_model:
            return LLMChatComponent(llm=self, **kwargs)
        else:
            return LLMCompleteComponent(llm=self, **kwargs)

    # -- Structured outputs --

    @dispatcher.span
    def structured_predict(
        self,
        output_cls: BaseModel,
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> BaseModel:
        r"""Structured predict.

        Args:
            output_cls (BaseModel):
                Output class to use for structured prediction.
            prompt (PromptTemplate):
                Prompt template to use for structured prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            BaseModel: The structured prediction output.

        Examples:
            ```python
            from pydantic import BaseModel

            class Test(BaseModel):
                \"\"\"My test class.\"\"\"
                name: str

            from llama_index.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please predict a Test with a random name related to {topic}.")
            output = llm.structured_predict(Test, prompt, topic="cats")
            print(output.name)
            ```
        """
        from llama_index.core.program.utils import get_program_for_llm

        dispatcher.event(
            LLMStructuredPredictStartEvent(
                output_cls=output_cls, template=prompt, template_args=prompt_args
            )
        )
        program = get_program_for_llm(
            output_cls,
            prompt,
            self,
            pydantic_program_mode=self.pydantic_program_mode,
        )

        result = program(llm_kwargs=llm_kwargs, **prompt_args)
        dispatcher.event(LLMStructuredPredictEndEvent(output=result))
        return result

    @dispatcher.span
    async def astructured_predict(
        self,
        output_cls: BaseModel,
        prompt: PromptTemplate,
        **prompt_args: Any,
    ) -> BaseModel:
        r"""Async Structured predict.

        Args:
            output_cls (BaseModel):
                Output class to use for structured prediction.
            prompt (PromptTemplate):
                Prompt template to use for structured prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            BaseModel: The structured prediction output.

        Examples:
            ```python
            from pydantic import BaseModel

            class Test(BaseModel):
                \"\"\"My test class.\"\"\"
                name: str

            from llama_index.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please predict a Test with a random name related to {topic}.")
            output = await llm.astructured_predict(Test, prompt, topic="cats")
            print(output.name)
            ```
        """
        from llama_index.core.program.utils import get_program_for_llm

        dispatcher.event(
            LLMStructuredPredictStartEvent(
                output_cls=output_cls, template=prompt, template_args=prompt_args
            )
        )

        program = get_program_for_llm(
            output_cls,
            prompt,
            self,
            pydantic_program_mode=self.pydantic_program_mode,
        )

        result = await program.acall(**prompt_args)
        dispatcher.event(LLMStructuredPredictEndEvent(output=result))
        return result

    @dispatcher.span
    def stream_structured_predict(
        self,
        output_cls: BaseModel,
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Generator[Union[Model, List[Model]], None, None]:
        r"""Stream Structured predict.

        Args:
            output_cls (BaseModel):
                Output class to use for structured prediction.
            prompt (PromptTemplate):
                Prompt template to use for structured prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            Generator: A generator returning partial copies of the model or list of models.

        Examples:
            ```python
            from pydantic import BaseModel

            class Test(BaseModel):
                \"\"\"My test class.\"\"\"
                name: str

            from llama_index.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please predict a Test with a random name related to {topic}.")
            stream_output = llm.stream_structured_predict(Test, prompt, topic="cats")
            for partial_output in stream_output:
                # stream partial outputs until completion
                print(partial_output.name)
            ```
        """
        from llama_index.core.program.utils import get_program_for_llm

        dispatcher.event(
            LLMStructuredPredictStartEvent(
                output_cls=output_cls, template=prompt, template_args=prompt_args
            )
        )
        program = get_program_for_llm(
            output_cls,
            prompt,
            self,
            pydantic_program_mode=self.pydantic_program_mode,
        )

        result = program.stream_call(llm_kwargs=llm_kwargs, **prompt_args)
        for r in result:
            dispatcher.event(LLMStructuredPredictInProgressEvent(output=r))
            yield r

        dispatcher.event(LLMStructuredPredictEndEvent(output=r))

    @dispatcher.span
    async def astream_structured_predict(
        self,
        output_cls: BaseModel,
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[Union[Model, List[Model]], None]:
        r"""Async Stream Structured predict.

        Args:
            output_cls (BaseModel):
                Output class to use for structured prediction.
            prompt (PromptTemplate):
                Prompt template to use for structured prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            Generator: A generator returning partial copies of the model or list of models.

        Examples:
            ```python
            from pydantic import BaseModel

            class Test(BaseModel):
                \"\"\"My test class.\"\"\"
                name: str

            from llama_index.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please predict a Test with a random name related to {topic}.")
            stream_output = await llm.astream_structured_predict(Test, prompt, topic="cats")
            async for partial_output in stream_output:
                # stream partial outputs until completion
                print(partial_output.name)
            ```
        """

        async def gen() -> AsyncGenerator[Union[Model, List[Model]], None]:
            from llama_index.core.program.utils import get_program_for_llm

            dispatcher.event(
                LLMStructuredPredictStartEvent(
                    output_cls=output_cls, template=prompt, template_args=prompt_args
                )
            )
            program = get_program_for_llm(
                output_cls,
                prompt,
                self,
                pydantic_program_mode=self.pydantic_program_mode,
            )

            result = await program.astream_call(llm_kwargs=llm_kwargs, **prompt_args)
            async for r in result:
                dispatcher.event(LLMStructuredPredictInProgressEvent(output=r))
                yield r

            dispatcher.event(LLMStructuredPredictEndEvent(output=r))

        return gen()

    # -- Prompt Chaining --

    @dispatcher.span
    def predict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> str:
        """Predict for a given prompt.

        Args:
            prompt (BasePromptTemplate):
                The prompt to use for prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            str: The prediction output.

        Examples:
            ```python
            from llama_index.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please write a random name related to {topic}.")
            output = llm.predict(prompt, topic="cats")
            print(output)
            ```
        """
        dispatcher.event(
            LLMPredictStartEvent(template=prompt, template_args=prompt_args)
        )
        self._log_template_data(prompt, **prompt_args)

        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = self.chat(messages)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = self.complete(formatted_prompt, formatted=True)
            output = response.text
        parsed_output = self._parse_output(output)
        dispatcher.event(LLMPredictEndEvent(output=parsed_output))
        return parsed_output

    @dispatcher.span
    def stream(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> TokenGen:
        """Stream predict for a given prompt.

        Args:
            prompt (BasePromptTemplate):
                The prompt to use for prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Yields:
            str: Each streamed token.

        Examples:
            ```python
            from llama_index.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please write a random name related to {topic}.")
            gen = llm.stream_predict(prompt, topic="cats")
            for token in gen:
                print(token, end="", flush=True)
            ```
        """
        self._log_template_data(prompt, **prompt_args)

        dispatcher.event(
            LLMPredictStartEvent(template=prompt, template_args=prompt_args)
        )
        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = self.stream_chat(messages)
            stream_tokens = stream_chat_response_to_tokens(chat_response)
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            stream_response = self.stream_complete(formatted_prompt, formatted=True)
            stream_tokens = stream_completion_response_to_tokens(stream_response)

        if prompt.output_parser is not None or self.output_parser is not None:
            raise NotImplementedError("Output parser is not supported for streaming.")

        return stream_tokens

    @dispatcher.span
    async def apredict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> str:
        """Async Predict for a given prompt.

        Args:
            prompt (BasePromptTemplate):
                The prompt to use for prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            str: The prediction output.

        Examples:
            ```python
            from llama_index.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please write a random name related to {topic}.")
            output = await llm.apredict(prompt, topic="cats")
            print(output)
            ```
        """
        dispatcher.event(
            LLMPredictStartEvent(template=prompt, template_args=prompt_args)
        )
        self._log_template_data(prompt, **prompt_args)

        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = await self.achat(messages)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = await self.acomplete(formatted_prompt, formatted=True)
            output = response.text

        parsed_output = self._parse_output(output)
        dispatcher.event(LLMPredictEndEvent(output=parsed_output))
        return parsed_output

    @dispatcher.span
    async def astream(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> TokenAsyncGen:
        """Async stream predict for a given prompt.

        Args:
        prompt (BasePromptTemplate):
            The prompt to use for prediction.
        prompt_args (Any):
            Additional arguments to format the prompt with.

        Yields:
            str: An async generator that yields strings of tokens.

        Examples:
            ```python
            from llama_index.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please write a random name related to {topic}.")
            gen = await llm.astream_predict(prompt, topic="cats")
            async for token in gen:
                print(token, end="", flush=True)
            ```
        """
        self._log_template_data(prompt, **prompt_args)

        dispatcher.event(
            LLMPredictStartEvent(template=prompt, template_args=prompt_args)
        )
        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = await self.astream_chat(messages)
            stream_tokens = await astream_chat_response_to_tokens(chat_response)
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            stream_response = await self.astream_complete(
                formatted_prompt, formatted=True
            )
            stream_tokens = await astream_completion_response_to_tokens(stream_response)

        if prompt.output_parser is not None or self.output_parser is not None:
            raise NotImplementedError("Output parser is not supported for streaming.")

        return stream_tokens

    @dispatcher.span
    def predict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        """Predict and call the tool.

        By default uses a ReAct agent to do tool calling (through text prompting),
        but function calling LLMs will implement this differently.

        """
        from llama_index.core.agent.react import ReActAgentWorker
        from llama_index.core.agent.types import Task
        from llama_index.core.chat_engine.types import AgentChatResponse
        from llama_index.core.memory import ChatMemoryBuffer

        worker = ReActAgentWorker(
            tools,
            llm=self,
            callback_manager=self.callback_manager,
            verbose=verbose,
            max_iterations=kwargs.get("max_iterations", 10),
            react_chat_formatter=kwargs.get("react_chat_formatter", None),
            output_parser=kwargs.get("output_parser", None),
            tool_retriever=kwargs.get("tool_retriever", None),
            handle_reasoning_failure_fn=kwargs.get("handle_reasoning_failure_fn", None),
        )

        if isinstance(user_msg, ChatMessage) and isinstance(user_msg.content, str):
            user_msg = user_msg.content
        elif isinstance(user_msg, str):
            pass
        elif (
            not user_msg
            and chat_history is not None
            and len(chat_history) > 0
            and isinstance(chat_history[-1].content, str)
        ):
            user_msg = chat_history[-1].content
        else:
            raise ValueError("No user message provided or found in chat history.")

        task = Task(
            input=user_msg,
            memory=ChatMemoryBuffer.from_defaults(chat_history=chat_history),
            extra_state={},
            callback_manager=self.callback_manager,
        )
        step = worker.initialize_step(task)

        try:
            output = worker.run_step(step, task).output

            # react agent worker inserts a "Observation: " prefix to the response
            if output.response and output.response.startswith("Observation: "):
                output.response = output.response.replace("Observation: ", "")
        except Exception as e:
            output = AgentChatResponse(
                response="An error occurred while running the tool: " + str(e),
                sources=[],
            )

        return output

    @dispatcher.span
    async def apredict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        """Predict and call the tool."""
        from llama_index.core.agent.react import ReActAgentWorker
        from llama_index.core.agent.types import Task
        from llama_index.core.chat_engine.types import AgentChatResponse
        from llama_index.core.memory import ChatMemoryBuffer

        worker = ReActAgentWorker(
            tools,
            llm=self,
            callback_manager=self.callback_manager,
            verbose=verbose,
            max_iterations=kwargs.get("max_iterations", 10),
            react_chat_formatter=kwargs.get("react_chat_formatter", None),
            output_parser=kwargs.get("output_parser", None),
            tool_retriever=kwargs.get("tool_retriever", None),
            handle_reasoning_failure_fn=kwargs.get("handle_reasoning_failure_fn", None),
        )

        if isinstance(user_msg, ChatMessage) and isinstance(user_msg.content, str):
            user_msg = user_msg.content
        elif isinstance(user_msg, str):
            pass
        elif (
            not user_msg
            and chat_history is not None
            and len(chat_history) > 0
            and isinstance(chat_history[-1].content, str)
        ):
            user_msg = chat_history[-1].content
        else:
            raise ValueError("No user message provided or found in chat history.")

        task = Task(
            input=user_msg,
            memory=ChatMemoryBuffer.from_defaults(chat_history=chat_history),
            extra_state={},
            callback_manager=self.callback_manager,
        )
        step = worker.initialize_step(task)

        try:
            output = (await worker.arun_step(step, task)).output

            # react agent worker inserts a "Observation: " prefix to the response
            if output.response and output.response.startswith("Observation: "):
                output.response = output.response.replace("Observation: ", "")
        except Exception as e:
            output = AgentChatResponse(
                response="An error occurred while running the tool: " + str(e),
                sources=[],
            )

        return output

    def as_structured_llm(
        self,
        output_cls: BaseModel,
        **kwargs: Any,
    ) -> "StructuredLLM":
        """Return a structured LLM around a given object."""
        from llama_index.core.llms.structured_llm import StructuredLLM

        return StructuredLLM(llm=self, output_cls=output_cls, **kwargs)


class BaseLLMComponent(QueryComponent):
    """Base LLM component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: LLM = Field(..., description="LLM")
    streaming: bool = Field(default=False, description="Streaming mode")

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""
        self.llm.callback_manager = callback_manager


class LLMCompleteComponent(BaseLLMComponent):
    """LLM completion component."""

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "prompt" not in input:
            raise ValueError("Prompt must be in input dict.")

        # do special check to see if prompt is a list of chat messages
        if isinstance(input["prompt"], get_args(List[ChatMessage])):
            if self.llm.messages_to_prompt:
                input["prompt"] = self.llm.messages_to_prompt(input["prompt"])
            input["prompt"] = validate_and_convert_stringable(input["prompt"])
        else:
            input["prompt"] = validate_and_convert_stringable(input["prompt"])
            if self.llm.completion_to_prompt:
                input["prompt"] = self.llm.completion_to_prompt(input["prompt"])

        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # TODO: support only complete for now
        # non-trivial to figure how to support chat/complete/etc.
        prompt = kwargs["prompt"]
        # ignore all other kwargs for now

        response: Any
        if self.streaming:
            response = self.llm.stream_complete(prompt, formatted=True)
        else:
            response = self.llm.complete(prompt, formatted=True)
        return {"output": response}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # TODO: support only complete for now
        # non-trivial to figure how to support chat/complete/etc.
        prompt = kwargs["prompt"]
        # ignore all other kwargs for now
        response = await self.llm.acomplete(prompt, formatted=True)
        return {"output": response}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        # TODO: support only complete for now
        return InputKeys.from_keys({"prompt"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


class LLMChatComponent(BaseLLMComponent):
    """LLM chat component."""

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "messages" not in input:
            raise ValueError("Messages must be in input dict.")

        # if `messages` is a string, convert to a list of chat message
        if isinstance(input["messages"], get_args(StringableInput)):
            input["messages"] = validate_and_convert_stringable(input["messages"])
            input["messages"] = prompt_to_messages(str(input["messages"]))

        for message in input["messages"]:
            if not isinstance(message, ChatMessage):
                raise ValueError("Messages must be a list of ChatMessage")
        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # TODO: support only complete for now
        # non-trivial to figure how to support chat/complete/etc.
        messages = kwargs["messages"]

        response: Any
        if self.streaming:
            response = self.llm.stream_chat(messages)
        else:
            response = self.llm.chat(messages)
        return {"output": response}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # TODO: support only complete for now
        # non-trivial to figure how to support chat/complete/etc.
        messages = kwargs["messages"]

        response: Any
        if self.streaming:
            response = await self.llm.astream_chat(messages)
        else:
            response = await self.llm.achat(messages)
        return {"output": response}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        # TODO: support only complete for now
        return InputKeys.from_keys({"messages"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})
