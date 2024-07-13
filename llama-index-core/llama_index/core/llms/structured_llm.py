from typing import (
    Any,
    List,
    Optional,
    Union,
    TYPE_CHECKING,
    Type,
    Sequence, 
    Dict
)
import asyncio

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole
)
from llama_index.core.llms.llm import LLM, BaseLLMComponent, LLMChatComponent, LLMCompleteComponent

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
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.prompts.base import ChatPromptTemplate
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.core.base.query_pipeline.query import (
    InputKeys,
    OutputKeys,
    QueryComponent,
    StringableInput,
    validate_and_convert_stringable,
)

if TYPE_CHECKING:
    from llama_index.core.chat_engine.types import AgentChatResponse
    from llama_index.core.tools.types import BaseTool
    from llama_index.core.types import BasePydanticProgram


class StructuredLLM(LLM):
    """
    A structured LLM takes in an inner LLM along with a designated output class,
    and all methods will return outputs in that structure.

    """
    llm: LLM
    output_cls: Type[BaseModel] = Field(..., description="Output class for the structured LLM.", exclude=True)

    @classmethod
    def class_name(cls) -> str:
        return "structured_llm"


    @property
    def metadata(self) -> LLMMetadata:
        return self.llm.metadata

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM."""

        # TODO: 
        from llama_index.core.program.utils import get_program_for_llm

        # NOTE: we are wrapping existing messages in a ChatPromptTemplate to
        # make this work with our FunctionCallingProgram, even though
        # the messages don't technically have any variables (they are already formatted)

        chat_prompt = ChatPromptTemplate(message_templates=messages)
        
        output = self.llm.structured_predict(
            output_cls=self.output_cls,
            prompt=chat_prompt,
            **kwargs
        )
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=output.json()),
            raw=output,
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError("stream_chat is not supported by default.")

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Stream completion endpoint for LLM."""
        raise NotImplementedError("stream_complete is not supported by default.")

    # ===== Async Endpoints =====
    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> "AgentChatResponse":
        # NOTE: we are wrapping existing messages in a ChatPromptTemplate to
        # make this work with our FunctionCallingProgram, even though
        # the messages don't technically have any variables (they are already formatted)

        chat_prompt = ChatPromptTemplate(message_templates=messages)
        
        output = await self.llm.astructured_predict(
            output_cls=self.output_cls,
            prompt=chat_prompt,
            **kwargs
        )
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=output.json()),
            raw=output,
        )

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async stream chat endpoint for LLM."""
        raise NotImplementedError("astream_chat is not supported by default.")

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = achat_to_completion_decorator(self.achat)
        return await complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Async stream completion endpoint for LLM."""
        raise NotImplementedError("astream_complete is not supported by default.")

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """Return query component."""
        if self.metadata.is_chat_model:
            base_component = LLMChatComponent(llm=self, **kwargs)
        else:
            base_component = LLMCompleteComponent(llm=self, **kwargs)
        # llm_component = self.llm.as_query_component()
        return StructuredLLMComponent(llm_component=base_component)


class StructuredLLMComponent(QueryComponent):
    """Structured LLM component.

    Wraps an existing LLM component, directly returns structured output.
    
    """

    llm_component: BaseLLMComponent

    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""
        self.llm_component.set_callback_manager(callback_manager)

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        return self.llm_component.validate_component_inputs(input)

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        output = self.llm_component.run_component(**kwargs)["output"]
        # NOTE: can either be a CompletionResponse or ChatResponse
        # other types are not supported at the moment
        if isinstance(output, CompletionResponse):
            return {"output": output.raw}
        elif isinstance(output, ChatResponse):
            return {"output": output.raw}
        else:
            raise ValueError("Unsupported output type from LLM component.")

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        output = await self.llm_component.arun_component(**kwargs)["output"]
        # NOTE: can either be a CompletionResponse or ChatResponse
        # other types are not supported at the moment
        if isinstance(output, CompletionResponse):
            return {"output": output.raw}
        elif isinstance(output, ChatResponse):
            return {"output": output.raw}
        else:
            raise ValueError("Unsupported output type from LLM component.")

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return self.llm_component.input_keys

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})

        