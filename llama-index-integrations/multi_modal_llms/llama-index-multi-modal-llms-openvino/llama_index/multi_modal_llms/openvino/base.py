from typing import Any, Dict, Sequence, cast
from deprecated import deprecated
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    ImageBlock,
)
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.core.base.llms.generic_utils import image_node_to_image_block
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.multi_modal_llms import MultiModalLLMMetadata
from llama_index.core.schema import ImageDocument, ImageNode
from llama_index.core.types import Thread
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer
from optimum.intel.openvino import OVModelForVisualCausalLM

DEFAULT_MULTIMODAL_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"


@deprecated(
    reason="This package has been deprecated and thus will no longer be maintained. Please feel free to contribute to multi modal support in llama-index-llms-openvino instead. See Multi Modal LLMs documentation for a complete guide on migration: https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/#multi-modal-llms",
    version="0.3.1",
)
class OpenVINOMultiModal(OpenVINOLLM):
    """
    This class provides a base implementation for interacting with OpenVINO multi-modal models.
    It handles model initialization, input preparation, and text/image-based interaction.
    """

    model_id_or_path: str = Field(
        default=DEFAULT_MULTIMODAL_MODEL,
        description="The model id or local path of the Hugging Face multi-modal model to use.",
    )
    device: str = Field(
        default="auto",
        description="The device to run the model on.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code when loading the model.",
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of new tokens to generate.",
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="generation kwargs for model generation.",
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model kwargs for model initialization.",
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _processor: Any = PrivateAttr()
    _messages_to_prompt: Any = PrivateAttr()

    def __init__(self, messages_to_prompt=None, **kwargs: Any) -> None:
        """
        Initializes the HuggingFace multi-modal model and processor based on the provided configuration.
        """
        super().__init__(**kwargs)
        try:
            # use local model
            self._model = OVModelForVisualCausalLM.from_pretrained(
                self.model_id_or_path,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
                **self.model_kwargs,
            )
        except Exception:
            # use remote model
            self._model = OVModelForVisualCausalLM.from_pretrained(
                self.model_id_or_path,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
                export=True,
                **self.model_kwargs,
            )
        # Load the processor (for handling text and image inputs)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id_or_path, trust_remote_code=self.trust_remote_code
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)

        self._messages_to_prompt = messages_to_prompt or self._prepare_messages

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name for the model."""
        return "OpenVINO_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_id_or_path,
        )

    # each unique model will override it
    def _prepare_messages(
        self, messages: Sequence[ChatMessage], image_documents: Sequence[ImageDocument]
    ) -> Dict[str, Any]:
        """
        Prepares the input messages and images.
        """
        if all(isinstance(doc, ImageNode) for doc in image_documents):
            image_docs: Sequence[ImageBlock] = [
                image_node_to_image_block(doc) for doc in image_documents
            ]
        else:
            image_docs = cast(Sequence[ImageBlock], image_documents)
        conversation = []
        images = []
        conversation.append(
            {"type": "text", "text": messages[0].content}
        )  # Add user text message
        for img_doc in image_docs:
            images.append(Image.open(img_doc.path))
            conversation.append({"type": "image"})
        messages = [
            {"role": "user", "content": conversation}
        ]  # Wrap conversation in a user role

        # Apply a chat template to format the message with the processor
        text_prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # Prepare the model inputs (text + images) and convert to tensor
        return self._processor(text=text_prompt, images=images, return_tensors="pt")

    # each unique model will override it
    def _generate(self, prepared_inputs: Dict[str, Any]) -> str:
        """
        Generates text based on prepared inputs. The text is decoded from token IDs generated by the model.
        """
        output_ids = self._model.generate(
            **prepared_inputs,
            max_new_tokens=self.max_new_tokens,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(prepared_inputs["input_ids"], output_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text[0]

    # some models will override it, some won't
    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        """
        Completes a task based on a text prompt and optional images.
        The method prepares inputs and generates the corresponding text.
        """
        prepared_inputs = self._messages_to_prompt(
            [ChatMessage(role="user", content=prompt)], image_documents
        )
        generated_text = self._generate(prepared_inputs)
        return CompletionResponse(text=generated_text)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion endpoint."""
        from transformers import TextIteratorStreamer

        prepared_inputs = self._messages_to_prompt(
            [ChatMessage(role="user", content=prompt)], image_documents
        )

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            prepared_inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            **self.generate_kwargs,
        )

        # generate in background thread
        # NOTE/TODO: token counting doesn't work with streaming
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        # create generator based off of streamer
        def gen() -> CompletionResponseGen:
            text = ""
            for x in streamer:
                text += x
                yield CompletionResponse(text=text, delta=x)

        return gen()

    # some models will override it, some won't
    def chat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Sequence[ImageDocument],
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Engages in a chat-style interaction by processing a sequence of messages and optional images.
        """
        prepared_inputs = self._prepare_messages(messages, image_documents)
        generated_text = self._generate(prepared_inputs)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=generated_text),
            raw={"model_output": generated_text},
        )
