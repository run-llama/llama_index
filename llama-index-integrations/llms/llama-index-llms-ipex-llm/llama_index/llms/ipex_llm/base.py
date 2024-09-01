# This file is adapted from
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/
# llms/llama-index-llms-huggingface/llama_index/llms/huggingface/base.py

import logging
from typing import Any, Callable, List, Optional, Sequence

import torch
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.custom import CustomLLM

from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode, Thread
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import AutoTokenizer, LlamaTokenizer


DEFAULT_HUGGINGFACE_MODEL = "meta-llama/Llama-2-7b-chat-hf"

logger = logging.getLogger(__name__)


class IpexLLM(CustomLLM):
    r"""IPEX-LLM.

    Example:
        .. code-block:: python

            from llama_index.llms.ipex_llm import IpexLLM
            llm = IpexLLM(model_path="/path/to/llama/model")
    """

    model_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The model name to use from HuggingFace. "
            "Unused if `model` is passed in directly."
        ),
    )
    load_in_4bit: bool = Field(
        default=True,
        description=(
            "Whether to load model in 4bit." "Unused if `load_in_low_bit` is not None."
        ),
    )
    load_in_low_bit: str = Field(
        default=None,
        description=(
            "Which low bit precisions to use when loading model. "
            "Example values: 'sym_int4', 'asym_int4', 'fp4', 'nf4', 'fp8', etc."
            "Will override `load_in_4bit` if this is specified."
        ),
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of tokens available for input.",
        gt=0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    tokenizer_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The name of the tokenizer to use from HuggingFace. "
            "Unused if `tokenizer` is passed in directly."
        ),
    )
    device_map: str = Field(
        default="cpu", description="The device_map to use. Defaults to 'cpu'."
    )
    stopping_ids: List[int] = Field(
        default_factory=list,
        description=(
            "The stopping ids to use. "
            "Generation stops when these token IDs are predicted."
        ),
    )
    tokenizer_outputs_to_remove: list = Field(
        default_factory=list,
        description=(
            "The outputs to remove from the tokenizer. "
            "Sometimes huggingface tokenizers return extra inputs that cause errors."
        ),
    )
    tokenizer_kwargs: dict = Field(
        default_factory=dict, description="The kwargs to pass to the tokenizer."
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during initialization.",
    )
    generate_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during generation.",
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            LLMMetadata.__fields__["is_chat_model"].field_info.description
            + " Be sure to verify that you either pass an appropriate tokenizer "
            "that can convert prompts to properly formatted chat messages or a "
            "`messages_to_prompt` that does so."
        ),
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _stopping_criteria: Any = PrivateAttr()

    def __init__(
        self,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        tokenizer_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model_name: str = DEFAULT_HUGGINGFACE_MODEL,
        load_in_4bit: Optional[bool] = True,
        load_in_low_bit: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device_map: str = "cpu",
        stopping_ids: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[dict] = None,
        tokenizer_outputs_to_remove: Optional[list] = None,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        low_bit_model: Optional[bool] = False,
    ) -> None:
        """
        Construct IpexLLM.

        Args:
            context_window: The maximum number of tokens available for input.
            max_new_tokens: The maximum number of tokens to generate.
            tokenizer_name: The name of the tokenizer to use from HuggingFace.
                        Unused if `tokenizer` is passed in directly.
            model_name: The model name to use from HuggingFace.
                        Unused if `model` is passed in directly.
            model: The HuggingFace model.
            tokenizer: The tokenizer.
            device_map: The device_map to use. Defaults to 'cpu'.
            stopping_ids: The stopping ids to use.
                        Generation stops when these token IDs are predicted.
            tokenizer_kwargs: The kwargs to pass to the tokenizer.
            tokenizer_outputs_to_remove: The outputs to remove from the tokenizer.
                        Sometimes huggingface tokenizers return extra inputs that cause errors.
            model_kwargs: The kwargs to pass to the model during initialization.
            generate_kwargs: The kwargs to pass to the model during generation.
            is_chat_model: Whether the model is `chat`
            callback_manager: Callback manager.
            messages_to_prompt: Function to convert messages to prompt.
            completion_to_prompt: Function to convert messages to prompt.
            pydantic_program_mode: DEFAULT.
            output_parser: BaseOutputParser.

        Returns:
            None.
        """
        model_kwargs = model_kwargs or {}

        if model:
            model = model
        else:
            model = self._load_model(
                low_bit_model, load_in_4bit, load_in_low_bit, model_name, model_kwargs
            )
        if device_map not in ["cpu", "xpu"] and not device_map.startswith("xpu:"):
            raise ValueError(
                "IpexLLMEmbedding currently only supports device to be 'cpu', 'xpu', "
                f"or 'xpu:<device_id>', but you have: {device_map}."
            )
        if "xpu" in device_map:
            model = model.to(device_map)

        # check context_window
        config_dict = model.config.to_dict()
        model_context_window = int(
            config_dict.get("max_position_embeddings", context_window)
        )
        if model_context_window and model_context_window < context_window:
            logger.warning(
                f"Supplied context_window {context_window} is greater "
                f"than the model's max input size {model_context_window}. "
                "Disable this warning by setting a lower context_window."
            )
            context_window = model_context_window

        tokenizer_kwargs = tokenizer_kwargs or {}
        if "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = context_window

        if tokenizer:
            tokenizer = tokenizer
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, **tokenizer_kwargs
                )
            except Exception:
                tokenizer = LlamaTokenizer.from_pretrained(
                    tokenizer_name, trust_remote_code=True
                )

        if tokenizer_name != model_name:
            logger.warning(
                f"The model `{model_name}` and tokenizer `{tokenizer_name}` "
                f"are from different paths, please ensure that they are compatible."
            )

        # setup stopping criteria
        stopping_ids_list = stopping_ids or []

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs: Any,
            ) -> bool:
                for stop_id in stopping_ids_list:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        messages_to_prompt = messages_to_prompt or self._tokenizer_messages_to_prompt

        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            device_map=device_map,
            stopping_ids=stopping_ids or [],
            tokenizer_kwargs=tokenizer_kwargs or {},
            tokenizer_outputs_to_remove=tokenizer_outputs_to_remove or [],
            model_kwargs=model_kwargs or {},
            generate_kwargs=generate_kwargs or {},
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._model = model
        self._tokenizer = tokenizer
        self._stopping_criteria = stopping_criteria

    @classmethod
    def from_model_id(
        cls,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        tokenizer_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model_name: str = DEFAULT_HUGGINGFACE_MODEL,
        load_in_4bit: Optional[bool] = True,
        load_in_low_bit: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device_map: str = "cpu",
        stopping_ids: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[dict] = None,
        tokenizer_outputs_to_remove: Optional[list] = None,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ):
        return cls(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            load_in_low_bit=load_in_low_bit,
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
            stopping_ids=stopping_ids,
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_outputs_to_remove=tokenizer_outputs_to_remove,
            model_kwargs=model_kwargs,
            generate_kwargs=generate_kwargs,
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            low_bit_model=False,
        )

    @classmethod
    def from_model_id_low_bit(
        cls,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        tokenizer_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device_map: str = "cpu",
        stopping_ids: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[dict] = None,
        tokenizer_outputs_to_remove: Optional[list] = None,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ):
        return cls(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
            stopping_ids=stopping_ids,
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_outputs_to_remove=tokenizer_outputs_to_remove,
            model_kwargs=model_kwargs,
            generate_kwargs=generate_kwargs,
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            low_bit_model=True,
        )

    @classmethod
    def class_name(cls) -> str:
        return "IpexLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
        )

    def _load_model(
        self,
        low_bit_model: bool,
        load_in_4bit: bool,
        load_in_low_bit: str,
        model_name: str,
        model_kwargs: Any,
    ) -> Any:
        """Attempts to load a model with AutoModelForCausalLM and falls back to AutoModel on failure."""
        from ipex_llm.transformers import AutoModelForCausalLM, AutoModel

        load_kwargs = {"use_cache": True, "trust_remote_code": True}

        if not low_bit_model:
            if load_in_low_bit is not None:
                load_function_name = "from_pretrained"
                load_kwargs["load_in_low_bit"] = load_in_low_bit
            else:
                load_function_name = "from_pretrained"
                load_kwargs["load_in_4bit"] = load_in_4bit
        else:
            load_function_name = "load_low_bit"

        try:
            # Attempt to load with AutoModelForCausalLM
            return self._load_model_general(
                AutoModelForCausalLM,
                load_function_name,
                model_name,
                load_kwargs,
                model_kwargs,
            )
        except Exception:
            # Fallback to AutoModel if there's an exception
            return self._load_model_general(
                AutoModel, load_function_name, model_name, load_kwargs, model_kwargs
            )

    def _load_model_general(
        self,
        model_class: Any,
        load_function_name: str,
        model_name: str,
        load_kwargs,
        model_kwargs: dict,
    ) -> Any:
        """General function to attempt to load a model."""
        try:
            load_function = getattr(model_class, load_function_name)
            return load_function(model_name, **{**load_kwargs, **model_kwargs})
        except Exception as e:
            logger.error(
                f"Failed to load model using {model_class.__name__}.{load_function_name}: {e}"
            )

    def _tokenizer_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """
        Use the tokenizer to convert messages to prompt. Fallback to generic.

        Args:
            messages: Sequence of ChatMessage.

        Returns:
            Str of response.
        """
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages_dict = [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ]
            tokens = self._tokenizer.apply_chat_template(messages_dict)
            return self._tokenizer.decode(tokens)

        return generic_messages_to_prompt(messages)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """
        Complete by LLM.

        Args:
            prompt: Prompt for completion.
            formatted: Whether the prompt is formatted by wrapper.
            kwargs: Other kwargs for complete.

        Returns:
            CompletionReponse after generation.
        """
        if not formatted:
            prompt = self.completion_to_prompt(prompt)
        input_ids = self._tokenizer(prompt, return_tensors="pt")
        input_ids = input_ids.to(self._model.device)
        # remove keys from the tokenizer if needed, to avoid HF errors
        for key in self.tokenizer_outputs_to_remove:
            if key in input_ids:
                input_ids.pop(key, None)
        tokens = self._model.generate(
            **input_ids,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            pad_token_id=self._tokenizer.pad_token_id,
            **self.generate_kwargs,
        )
        completion_tokens = tokens[0][input_ids["input_ids"].size(1) :]
        completion = self._tokenizer.decode(completion_tokens, skip_special_tokens=True)

        return CompletionResponse(text=completion, raw={"model_output": tokens})

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """
        Complete by LLM in stream.

        Args:
            prompt: Prompt for completion.
            formatted: Whether the prompt is formatted by wrapper.
            kwargs: Other kwargs for complete.

        Returns:
            CompletionReponse after generation.
        """
        from transformers import TextIteratorStreamer

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self._model.device)

        for key in self.tokenizer_outputs_to_remove:
            if key in input_ids:
                input_ids.pop(key, None)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            pad_token_id=self._tokenizer.pad_token_id,
            **self.generate_kwargs,
        )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        # create generator based off of streamer
        def gen() -> CompletionResponseGen:
            text = ""
            for x in streamer:
                text += x
                yield CompletionResponse(text=text, delta=x)

        return gen()
