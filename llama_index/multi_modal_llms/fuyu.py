from typing import Any, Callable, Dict, Optional

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.multi_modal_llms import (
    MultiModalCompletionResponse,
    MultiModalCompletionResponseGen,
    MultiModalLLMMetadata,
)
from llama_index.schema import ImageDocument


class Fuyu(MultiModalLLMMetadata):
    model: str = Field(description="The Fuyu model to use.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: int = Field(
        description=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt"
    )
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
    prompt_key: str = Field(description="The key to use for the prompt in API calls.")
    image_key: str = Field(description="The key to use for the image in API calls.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Replicate API."
    )
    is_chat_model: bool = Field(
        default=False, description="Whether the model is a chat model."
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()

    def __init__(
        self,
        model: str = "lucataco/fuyu-8b:42f23bc876570a46f5a90737086fbc4c3f79dd11753a28eaa39544dd391815e9",
        temperature: float = 0.75,
        max_new_tokens: int = 512,
        num_input_files: int = 1,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        prompt_key: str = "prompt",
        image_key: str = "image",
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        is_chat_model: bool = False,
    ) -> None:
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)

        super().__init__(
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_input_files=num_input_files,
            additional_kwargs=additional_kwargs or {},
            context_window=context_window,
            prompt_key=prompt_key,
            image_key=image_key,
            callback_manager=callback_manager,
            is_chat_model=is_chat_model,
        )

    @classmethod
    def class_name(cls) -> str:
        return "fuyu_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=self.is_chat_model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_length": self.context_window,
            "max_new_tokens": self.max_new_tokens,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def load_image_documents(
        self, image_paths: list[str], **kwargs: Any
    ) -> list[ImageDocument]:
        image_documents = []
        for i in range(min(self.num_input_files, len(image_paths))):
            new_image_document = ImageDocument()
            if "http" in image_paths[i] or "https" in image_paths[i]:
                # remote image file with url
                new_image_document.image_url = image_paths[i]
            else:
                # local image file with path
                new_image_document.image_local_file_path = image_paths[i]

            image_documents.append(new_image_document)
        return image_documents

    def _get_multi_modal_input_dict(
        self, prompt: str, image_document: ImageDocument, **kwargs: Any
    ) -> Dict[str, Any]:
        if image_document and image_document.image_local_file_path:
            # load local image file and pass file handler to replicate
            try:
                return {
                    self.prompt_key: prompt,
                    self.image_key: open(image_document.image_local_file_path, "rb"),
                    **self._model_kwargs,
                    **kwargs,
                }
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Could not load local image file. Please check whether the file exists"
                )
        # load remote image url and pass file url to replicate
        return {
            self.prompt_key: prompt,
            self.image_key: image_document.image_url,
            **self._model_kwargs,
            **kwargs,
        }

    def complete(
        self,
        prompt: str,
        image_documents: list[ImageDocument],
        image_idx: int,
        **kwargs: Any
    ) -> MultiModalCompletionResponse:
        try:
            import replicate
        except ImportError:
            raise ImportError(
                "Could not import replicate library."
                "Please install replicate with `pip install replicate`"
            )

        prompt = self._completion_to_prompt(prompt)
        # load one of image from image document list for understanding
        input_dict = self._get_multi_modal_input_dict(
            prompt, image_documents[image_idx], **kwargs
        )

        return replicate.run(self.model, input=input_dict)

    def stream_complete(
        self,
        prompt: str,
        image_documents: list[ImageDocument],
        image_idx: int,
        **kwargs: Any
    ) -> MultiModalCompletionResponseGen:
        raise NotImplementedError(
            "stream_complete is not supported for Fuyu 8B model Replicate API atm."
        )
