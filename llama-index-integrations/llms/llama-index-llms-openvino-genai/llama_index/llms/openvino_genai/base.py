import logging
from typing import Any, Callable, Optional, Sequence, Union

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
from llama_index.core.prompts.base import PromptTemplate
import queue
import openvino_genai
import openvino as ov
from threading import Event, Thread

logger = logging.getLogger(__name__)


class OpenVINOGenAILLM(CustomLLM):
    r"""
    OpenVINO GenAI LLM.

    Examples:
        `pip install llama-index-llms-openvino-genai`

        ```python
        from llama_index.llms.openvino_genai import OpenVINOgenAILLM

        llm = OpenVINOGenAILLM(
            model_path=model_path,
            device="CPU",
        )

        response = llm.complete("What is the meaning of life?")
        print(str(response))
        ```

    """

    model_path: str = Field(
        default=None,
        description=("The model path to use from local. "),
    )
    system_prompt: str = Field(
        default="",
        description=(
            "The system prompt, containing any extra instructions or context. "
            "The model card on HuggingFace should specify if this is needed."
        ),
    )
    query_wrapper_prompt: PromptTemplate = Field(
        default=PromptTemplate("{query_str}"),
        description=(
            "The query wrapper prompt, containing the query placeholder. "
            "The model card on HuggingFace should specify if this is needed. "
            "Should contain a `{query_str}` placeholder."
        ),
    )
    device: str = Field(
        default="auto", description="The device to use. Defaults to 'auto'."
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            LLMMetadata.model_fields["is_chat_model"].description
            + " Be sure to verify that you either pass an appropriate tokenizer "
            "that can convert prompts to properly formatted chat messages or a "
            "`messages_to_prompt` that does so."
        ),
    )

    config: str = Field(
        default=None,
        description=("The LLM generation configurations."),
    )

    _pip: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _streamer: Any = PrivateAttr()

    def __init__(
        self,
        model_path: str,
        config: Optional[dict] = {},
        tokenizer: Optional[Any] = None,
        device: Optional[str] = "CPU",
        query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: str = "",
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        **kwargs: Any,
    ) -> None:
        class IterableStreamer(openvino_genai.StreamerBase):
            """
            A custom streamer class for handling token streaming
            and detokenization with buffering.

            Attributes:
                tokenizer (Tokenizer): The tokenizer used for encoding
                and decoding tokens.
                tokens_cache (list): A buffer to accumulate tokens
                for detokenization.
                text_queue (Queue): A synchronized queue
                for storing decoded text chunks.
                print_len (int): The length of the printed text
                to manage incremental decoding.

            """

            def __init__(self, tokenizer: Any) -> None:
                """
                Initializes the IterableStreamer with the given tokenizer.

                Args:
                    tokenizer (Tokenizer): The tokenizer to use for encoding
                    and decoding tokens.

                """
                super().__init__()
                self.tokenizer = tokenizer
                self.tokens_cache: list[int] = []
                self.text_queue: Any = queue.Queue()
                self.print_len = 0

            def __iter__(self) -> self:
                """
                Returns the iterator object itself.
                """
                return self

            def __next__(self) -> str:
                """
                Returns the next value from the text queue.

                Returns:
                    str: The next decoded text chunk.

                Raises:
                    StopIteration: If there are no more elements in the queue.

                """
                value = (
                    self.text_queue.get()
                )  # get() will be blocked until a token is available.
                if value is None:
                    raise StopIteration
                return value

            def get_stop_flag(self) -> bool:
                """
                Checks whether the generation process should be stopped.

                Returns:
                    bool: Always returns False in this implementation.

                """
                return False

            def put_word(self, word: Any) -> None:
                """
                Puts a word into the text queue.

                Args:
                    word (str): The word to put into the queue.

                """
                self.text_queue.put(word)

            def put(self, token_id: int) -> bool:
                """
                Processes a token and manages the decoding buffer.
                Adds decoded text to the queue.

                Args:
                    token_id (int): The token_id to process.

                Returns:
                    bool: True if generation should be stopped, False otherwise.

                """
                self.tokens_cache.append(token_id)
                text = self.tokenizer.decode(
                    self.tokens_cache, skip_special_tokens=True
                )

                word = ""
                if len(text) > self.print_len and text[-1] == "\n":
                    word = text[self.print_len :]
                    self.tokens_cache = []
                    self.print_len = 0
                elif len(text) >= 3 and text[-3:] == chr(65533):
                    pass
                elif len(text) > self.print_len:
                    word = text[self.print_len :]
                    self.print_len = len(text)
                self.put_word(word)

                if self.get_stop_flag():
                    self.end()
                    return True
                else:
                    return False

            def end(self) -> None:
                """
                Flushes residual tokens from the buffer
                and puts a None value in the queue to signal the end.
                """
                text = self.tokenizer.decode(
                    self.tokens_cache, skip_special_tokens=True
                )
                if len(text) > self.print_len:
                    word = text[self.print_len :]
                    self.put_word(word)
                    self.tokens_cache = []
                    self.print_len = 0
                self.put_word(None)

            def reset(self) -> None:
                """
                Resets the state.
                """
                self.tokens_cache = []
                self.text_queue = queue.Queue()
                self.print_len = 0

        class ChunkStreamer(IterableStreamer):
            def __init__(self, tokenizer: Any, tokens_len: int = 4) -> None:
                super().__init__(tokenizer)
                self.tokens_len = tokens_len

            def put(self, token_id: int) -> bool:
                if (len(self.tokens_cache) + 1) % self.tokens_len != 0:
                    self.tokens_cache.append(token_id)
                    return False
                return super().put(token_id)

        """Initialize params."""
        pipe = openvino_genai.LLMPipeline(model_path, device, config, **kwargs)

        config = pipe.get_generation_config()

        tokenizer = tokenizer or pipe.get_tokenizer()
        streamer = ChunkStreamer(tokenizer)

        if isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = PromptTemplate(query_wrapper_prompt)

        messages_to_prompt = messages_to_prompt or self._tokenizer_messages_to_prompt

        super().__init__(
            tokenizer=tokenizer,
            model_path=model_path,
            device=device,
            query_wrapper_prompt=query_wrapper_prompt,
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
        )

        self._pipe = pipe
        self._tokenizer = tokenizer
        self._streamer = streamer
        self.config = config

    @classmethod
    def class_name(cls) -> str:
        return "OpenVINO_GenAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            model_name=self.model_path,
            is_chat_model=self.is_chat_model,
        )

    def _tokenizer_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Use the tokenizer to convert messages to prompt. Fallback to generic."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages_dict = [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ]
            return (
                self._tokenizer.apply_chat_template(
                    messages_dict, add_generation_prompt=True
                )
                if isinstance(self._tokenizer, openvino_genai.Tokenizer)
                else self._tokenizer.apply_chat_template(
                    messages_dict, tokenize=False, add_generation_prompt=True
                )
            )

        return generic_messages_to_prompt(messages)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint."""
        full_prompt = prompt
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.completion_to_prompt:
                full_prompt = self.completion_to_prompt(full_prompt)
            elif self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        if not isinstance(self._tokenizer, openvino_genai.Tokenizer):
            inputs = self._tokenizer(
                full_prompt, add_special_tokens=False, return_tensors="np"
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            full_prompt = openvino_genai.TokenizedInputs(
                ov.Tensor(input_ids), ov.Tensor(attention_mask)
            )

        tokens = self._pipe.generate(full_prompt, self.config, **kwargs)
        if not isinstance(self._tokenizer, openvino_genai.Tokenizer):
            completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
            completion = self._tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            )
        else:
            completion = tokens
        return CompletionResponse(text=completion, raw={"model_output": tokens})

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion endpoint."""
        full_prompt = prompt
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        if not isinstance(self._tokenizer, openvino_genai.Tokenizer):
            inputs = self._tokenizer(
                full_prompt, add_special_tokens=False, return_tensors="np"
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            full_prompt = openvino_genai.TokenizedInputs(
                ov.Tensor(input_ids), ov.Tensor(attention_mask)
            )

        stream_complete = Event()

        def generate_and_signal_complete() -> None:
            """
            Generation function for single thread.
            """
            self._streamer.reset()
            self._pipe.generate(full_prompt, self.config, self._streamer, **kwargs)
            stream_complete.set()
            self._streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        # create generator based off of streamer
        def gen() -> CompletionResponseGen:
            text = ""
            for x in self._streamer:
                text += x
                yield CompletionResponse(text=text, delta=x)

        return gen()

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
