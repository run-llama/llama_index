from typing import Any, Callable, Dict, Optional, Sequence, List, TYPE_CHECKING

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
    LogProb,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

if TYPE_CHECKING:
    from mistralrs import (
        Runner,
        Which,
    )

DEFAULT_TOPK = 32
DEFAULT_TOPP = 0.1
DEFAULT_REPEAT_LAST_N = 64
DEFAULT_MAX_SEQS = 16
DEFAULT_PREFIX_CACHE_N = 16


def llama_index_to_mistralrs_messages(
    messages: Sequence[ChatMessage],
) -> List[Dict[str, str]]:
    """
    Convert llamaindex to mistralrs messages. Raises an exception if the role is not user or assistant.
    """
    messages_new = []
    for message in messages:
        if message.role == "user":
            messages_new.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            messages_new.append({"role": "assistant", "content": message.content})
        elif message.role == "system":
            messages_new.append({"role": "system", "content": message.content})
        else:
            raise ValueError(
                f"Unsupported chat role `{message.role}` for `mistralrs` automatic chat templating: supported are `user`, `assistant`, `system`. Please specify `messages_to_prompt`."
            )
    return messages_new


def extract_logprobs_choice(choice) -> Optional[List[LogProb]]:
    if choice.logprobs is not None:
        logprobs = []
        for logprob in choice.logprobs.content:
            logprobs.append(
                LogProb(
                    logprob=logprob.logprob,
                    bytes=logprob.bytes,
                    token=logprob.token,
                )
            )
    else:
        logprobs = None
    return logprobs


def extract_logprobs(response) -> Optional[List[List[LogProb]]]:
    if response.choices[0].logprobs is not None:
        choice_logprobs = []
        for choice in response.choices:
            choice_logprobs.append(extract_logprobs_choice(choice))
    else:
        choice_logprobs = None
    return choice_logprobs


def extract_logprobs_stream(response) -> Optional[List[List[LogProb]]]:
    if response.choices[0].logprobs is not None:
        logprobs = [extract_logprobs_choice(response.choices[0])]
    else:
        logprobs = None
    return logprobs


class MistralRS(CustomLLM):
    r"""MistralRS LLM.

    Examples:
        Install `mistralrs` following instructions:
        https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/README.md#installation-from-pypi

        Then `pip install llama-index-llms-mistral-rs`

        This LLM provides automatic chat templating as an option. If you do not provide `messages_to_prompt`,
        mistral.rs will automatically determine one. You can specify a JINJA chat template by passing it in
        `model_kwargs` in the `chat_template` key.

        ```python
        from llama_index.llms.mistral_rs import MistralRS
        from mistralrs import Which

        llm = MistralRS(
            which = Which.XLora(
                model_id=None,  # Automatically determine from ordering file
                tokenizer_json=None,
                repeat_last_n=64,
                xlora_model_id="lamm-mit/x-lora"
                order="xlora-paper-ordering.json", # Make sure you copy the ordering file from `mistral.rs/orderings`
                tgt_non_granular_index=None,
                arch=Architecture.Mistral,
            ),
            temperature=0.1,
            max_new_tokens=256,
            context_window=3900,
            generate_kwargs={},
            verbose=True,
        )

        response = llm.complete("Hello, how are you?")
        print(str(response))
        ```
    """

    model_url: Optional[str] = Field(description="local")
    model_path: Optional[str] = Field(description="local")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )
    _runner: "Runner" = PrivateAttr("Mistral.rs model runner.")
    _has_messages_to_prompt: bool = PrivateAttr("If `messages_to_prompt` is provided.")

    def __init__(
        self,
        which: "Which",
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        top_k: int = DEFAULT_TOPK,
        top_p: int = DEFAULT_TOPP,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        in_situ_quant: Optional[str] = None,
        max_seqs: int = DEFAULT_MAX_SEQS,
        token_source: str = "cache",
        prefix_cache_n: str = DEFAULT_PREFIX_CACHE_N,
        no_kv_cache: bool = False,
        chat_template: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
            {
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "top_logprobs": top_logprobs,
                "logprobs": top_logprobs is not None,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
        )

        super().__init__(
            model_path="local",
            model_url="local",
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs={},
            verbose=True,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._runner = Runner(
            which=which,
            token_source=token_source,
            max_seqs=max_seqs,
            prefix_cache_n=prefix_cache_n,
            no_kv_cache=no_kv_cache,
            chat_template=chat_template,
            in_situ_quant=in_situ_quant,
        )
        self._has_messages_to_prompt = messages_to_prompt is not None

    @classmethod
    def class_name(cls) -> str:
        return "MistralRS"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_path,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        try:
            from mistralrs import ChatCompletionRequest
        except ImportError as e:
            raise ValueError(
                "Missing `mistralrs` package. Install via `pip install mistralrs`."
            ) from e
        if self._has_messages_to_prompt:
            messages = self.messages_to_prompt(messages)
        else:
            messages = llama_index_to_mistralrs_messages(messages)
        self.generate_kwargs.update({"stream": False})

        request = ChatCompletionRequest(
            messages=messages,
            model="",
            logit_bias=None,
            **self.generate_kwargs,
        )

        response = self._runner.send_chat_completion_request(request)
        return CompletionResponse(
            text=response.choices[0].message.content,
            logprobs=extract_logprobs(response),
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        try:
            from mistralrs import ChatCompletionRequest
        except ImportError as e:
            raise ValueError(
                "Missing `mistralrs` package. Install via `pip install mistralrs`."
            ) from e
        if self._has_messages_to_prompt:
            messages = self.messages_to_prompt(messages)
        else:
            messages = llama_index_to_mistralrs_messages(messages)
        self.generate_kwargs.update({"stream": True})

        request = ChatCompletionRequest(
            messages=messages,
            model="",
            logit_bias=None,
            **self.generate_kwargs,
        )

        streamer = self._runner.send_chat_completion_request(request)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in streamer:
                delta = response.choices[0].delta.content
                text += delta
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=delta,
                    ),
                    delta=delta,
                    logprobs=extract_logprobs_stream(response),
                )

        return gen()

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        try:
            from mistralrs import ChatCompletionRequest
        except ImportError as e:
            raise ValueError(
                "Missing `mistralrs` package. Install via `pip install mistralrs`."
            ) from e
        self.generate_kwargs.update({"stream": False})
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        request = ChatCompletionRequest(
            messages=prompt,
            model="",
            logit_bias=None,
            **self.generate_kwargs,
        )
        completion_response = self._runner.send_chat_completion_request(request)
        return CompletionResponse(
            text=completion_response.choices[0].message.content,
            logprobs=extract_logprobs(completion_response),
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        try:
            from mistralrs import ChatCompletionRequest
        except ImportError as e:
            raise ValueError(
                "Missing `mistralrs` package. Install via `pip install mistralrs`."
            ) from e
        self.generate_kwargs.update({"stream": True})
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        request = ChatCompletionRequest(
            messages=prompt,
            model="",
            logit_bias=None,
            **self.generate_kwargs,
        )

        streamer = self._runner.send_chat_completion_request(request)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in streamer:
                delta = response.choices[0].delta.content
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                    logprobs=extract_logprobs_stream(response),
                )

        return gen()
