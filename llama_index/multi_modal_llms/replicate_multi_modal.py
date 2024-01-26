import logging
from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.schema import ImageDocument

_logger = logging.getLogger(__name__)

REPLICATE_MULTI_MODAL_LLM_MODELS = {
    "llava-13b": "yorickvp/llava-13b:e272157381e2a3bf12df3a8edd1f38d1dbd736bbb7437277c8b34175f8fce358",
    "fuyu-8b": "lucataco/fuyu-8b:42f23bc876570a46f5a90737086fbc4c3f79dd11753a28eaa39544dd391815e9",
    "minigpt-4": "daanelson/minigpt-4:b96a2f33cc8e4b0aa23eacfce731b9c41a7d9466d9ed4e167375587b54db9423",
    "cogvlm": "naklecha/cogvlm:ec3886f9ea85dd0aee216585be5e6d07b04c9650f7b8b08363a14eb89e207eb2",
}


class ReplicateMultiModal(MultiModalLLM):
    model: str = Field(description="The Multi-Modal model to use from Replicate.")
    temperature: float = Field(
        description="The temperature to use for sampling. Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic."
    )
    max_new_tokens: int = Field(
        description=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt"
    )
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
    prompt_key: str = Field(description="The key to use for the prompt in API calls.")
    image_key: str = Field(description="The key to use for the image in API calls.")
    top_p: float = Field(
        description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens."
    )
    num_beams: int = Field(description="Number of beams for beam search decoding.")
    repetition_penalty: float = Field(
        description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Replicate API."
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()

    def __init__(
        self,
        model: str = REPLICATE_MULTI_MODAL_LLM_MODELS["fuyu-8b"],
        temperature: float = 0.75,
        max_new_tokens: int = 512,
        num_input_files: int = 1,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        prompt_key: str = "prompt",
        image_key: str = "image",
        repetition_penalty: Optional[float] = 1.0,
        num_beams: Optional[int] = 1,
        top_p: Optional[float] = 0.9,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)

        super().__init__(
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_input_files=num_input_files,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            top_p=top_p,
            additional_kwargs=additional_kwargs or {},
            context_window=context_window,
            prompt_key=prompt_key,
            image_key=image_key,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "replicate_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_length": self.context_window,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "top_p": self.top_p,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_multi_modal_chat_messages(
        self, prompt: str, image_document: ImageDocument, **kwargs: Any
    ) -> Dict[str, Any]:
        if image_document.image_path:
            # load local image file and pass file handler to replicate
            try:
                return {
                    self.prompt_key: prompt,
                    self.image_key: open(image_document.image_path, "rb"),
                    **self._model_kwargs,
                    **kwargs,
                }
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Could not load local image file. Please check whether the file exists"
                )
        elif image_document.image_url:
            # load remote image url and pass file url to replicate
            return {
                self.prompt_key: prompt,
                self.image_key: image_document.image_url,
                **self._model_kwargs,
                **kwargs,
            }
        else:
            raise FileNotFoundError(
                "Could not load image file. Please check whether the file exists"
            )

    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        response_gen = self.stream_complete(prompt, image_documents, **kwargs)
        response_list = list(response_gen)
        final_response = response_list[-1]
        final_response.delta = None
        return final_response

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseGen:
        try:
            import replicate
        except ImportError:
            raise ImportError(
                "Could not import replicate library."
                "Please install replicate with `pip install replicate`"
            )

        # TODO: at the current moment, only support uploading one image document
        if len(image_documents) > 1:
            _logger.warning(
                "ReplicateMultiModal currently only supports uploading one image document"
                "we are using the first image document for completion."
            )

        prompt = self._completion_to_prompt(prompt)
        input_dict = self._get_multi_modal_chat_messages(
            # using the first image for single image completion
            prompt,
            image_documents[0],
            **kwargs,
        )
        if self.model not in REPLICATE_MULTI_MODAL_LLM_MODELS.values():
            raise ValueError(
                f"Unknown model {self.model!r}. Please provide a valid Replicate Multi-Modal model name in:"
                f" {', '.join(REPLICATE_MULTI_MODAL_LLM_MODELS.values())}"
            )

        response_iter = replicate.run(self.model, input=input_dict)

        def gen() -> CompletionResponseGen:
            text = ""
            for delta in response_iter:
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                )

        return gen()

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        raise NotImplementedError

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        raise NotImplementedError

    # ===== Async Endpoints =====

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        response_gen = self.stream_complete(prompt, image_documents, **kwargs)
        response_list = list(response_gen)
        final_response = response_list[-1]
        final_response.delta = None
        return final_response

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        try:
            import replicate
        except ImportError:
            raise ImportError(
                "Could not import replicate library."
                "Please install replicate with `pip install replicate`"
            )

        # TODO: at the current moment, only support uploading one image document
        if len(image_documents) > 1:
            _logger.warning(
                "ReplicateMultiModal currently only supports uploading one image document"
                "we are using the first image document for completion."
            )

        prompt = self._completion_to_prompt(prompt)
        input_dict = self._get_multi_modal_chat_messages(
            # using the first image for single image completion
            prompt,
            image_documents[0],
            **kwargs,
        )
        if self.model not in REPLICATE_MULTI_MODAL_LLM_MODELS.values():
            raise ValueError(
                f"Unknown model {self.model!r}. Please provide a valid Replicate Multi-Modal model name in:"
                f" {', '.join(REPLICATE_MULTI_MODAL_LLM_MODELS.values())}"
            )

        response_iter = replicate.run(self.model, input=input_dict)

        async def gen() -> CompletionResponseAsyncGen:
            text = ""
            for delta in response_iter:
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                )

        return gen()

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        raise NotImplementedError

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError
