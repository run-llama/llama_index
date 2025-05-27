import logging
from typing import Any, Callable, List, Optional, Sequence, Union

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field
from llama_index.llms.huggingface.base import HuggingFaceLLM
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.prompts.base import PromptTemplate

from optimum.intel.ipex import IPEXModelForCausalLM
import torch

DEFAULT_HUGGINGFACE_MODEL = "Intel/neural-chat-7b-v3-1"

logger = logging.getLogger(__name__)


class OptimumIntelLLM(HuggingFaceLLM):
    r"""
    OptimumIntelLLM LLM.

    Examples:
        `pip install llama-index-llms-optimum-intel`

        ```python
        from llama_index.llms.optimum_intel import OptimumIntelLLM

        def messages_to_prompt(messages):
            prompt = ""
            for message in messages:
                if message.role == 'system':
                prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == 'user':
                prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == 'assistant':
                prompt += f"<|assistant|>\n{message.content}</s>\n"

            # ensure we start with a system prompt, insert blank if needed
            if not prompt.startswith("<|system|>\n"):
                prompt = "<|system|>\n</s>\n" + prompt

            # add final assistant prompt
            prompt = prompt + "<|assistant|>\n"

            return prompt

        def completion_to_prompt(completion):
            return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

        import torch
        from llama_index.core.prompts import PromptTemplate
        from llama_index.llms.optimum-intel import OptimumIntelLLM

        llm = OptimumIntelLLM(
            model_name="HuggingFaceH4/zephyr-7b-beta",
            tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
            context_window=3900,
            max_new_tokens=256,,
            generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            device_map="auto",
        )

        response = llm.complete("What is the meaning of life?")
        print(str(response))
        ```

    """

    model_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The model name to use from HuggingFace. "
            "Unused if `model` is passed in directly."
        ),
    )
    tokenizer_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The name of the tokenizer to use from HuggingFace. "
            "Unused if `tokenizer` is passed in directly."
        ),
    )

    def __init__(
        self,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
        tokenizer_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device_map: Optional[str] = "auto",
        stopping_ids: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[dict] = None,
        tokenizer_outputs_to_remove: Optional[list] = None,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: str = "",
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """Initialize params."""
        model_kwargs = model_kwargs or {}

        model = IPEXModelForCausalLM.from_pretrained(
            model_name,
            export=True,
            **model_kwargs,
            torch_dtype=torch.bfloat16,  # keep or remove the dtype????
        )

        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
            stopping_ids=stopping_ids or [],
            tokenizer_kwargs=tokenizer_kwargs or {},
            tokenizer_outputs_to_remove=tokenizer_outputs_to_remove or [],
            model_kwargs=model_kwargs or {},
            generate_kwargs=generate_kwargs or {},
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OptimumIntelLLM"
