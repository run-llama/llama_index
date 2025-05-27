from typing import Any, Dict, Sequence, Union
from typing_extensions import override
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.multi_modal_llms import MultiModalLLM, MultiModalLLMMetadata
from llama_index.core.schema import ImageDocument, ImageNode
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoConfig,
    Qwen2VLForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    MllamaForConditionalGeneration,
)
from qwen_vl_utils import (
    process_vision_info,
)  # We will need that in order to work with different image shapes

DEFAULT_MULTIMODAL_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_REQUEST_TIMEOUT = 120.0
SUPPORTED_VLMS = [
    "Phi3VForCausalLM",
    "Florence2ForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "MllamaForConditionalGeneration",
]


class HuggingFaceMultiModal(MultiModalLLM):
    """
    This class provides a base implementation for interacting with HuggingFace multi-modal models.
    It handles model initialization, input preparation, and text/image-based interaction.
    """

    model_name: str = Field(
        description="The name of the Hugging Face multi-modal model to use."
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="The device to run the model on.",
    )
    device_map: Union[Dict[str, Any], str] = Field(
        default="auto",
        description="Tell HF accelerate where to put each layer of the model. In auto mode, HF accelerate determines this on it's own",
    )
    torch_dtype: Any = Field(
        default=torch.float16 if torch.cuda.is_available() else torch.float32,
        description="The torch dtype to use.",
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
    temperature: float = Field(
        default=0.0, description="The temperature to use for sampling."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for model initialization and generation.",
    )

    _model: Any = PrivateAttr()
    _processor: Any = PrivateAttr()
    _config: Any = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the HuggingFace multi-modal model and processor based on the provided configuration.
        """
        super().__init__(**kwargs)
        try:
            # Load model configuration
            self._config = AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            architecture = self._config.architectures[0]
            AutoModelClass = AutoModelForCausalLM  # Default model class

            # Special cases for specific model architectures
            if "Qwen2VLForConditionalGeneration" in architecture:
                AutoModelClass = Qwen2VLForConditionalGeneration
            if "PaliGemmaForConditionalGeneration" in architecture:
                AutoModelClass = PaliGemmaForConditionalGeneration
            if "MllamaForConditionalGeneration" in architecture:
                AutoModelClass = MllamaForConditionalGeneration

            # Load the model based on the architecture
            self._model = AutoModelClass.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.trust_remote_code,
                **self.additional_kwargs,
            )
            # Load the processor (for handling text and image inputs)
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize the model and processor: {e!s}")

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name for the model."""
        return "HuggingFace_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
        )

    # each unique model will override it
    def _prepare_messages(
        self, messages: Sequence[ChatMessage], image_documents: Sequence[ImageDocument]
    ) -> Dict[str, Any]:
        """
        Abstract method: Prepares input messages and image documents for the model.
        This must be overridden by subclasses.
        """
        raise NotImplementedError

    # each unique model will override it
    def _generate(self, prepared_inputs: Dict[str, Any]) -> str:
        """
        Abstract method: Generates text based on the prepared inputs.
        This must be overridden by subclasses.
        """
        raise NotImplementedError

    # some models will override it, some won't
    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        """
        Completes a task based on a text prompt and optional images.
        The method prepares inputs and generates the corresponding text.
        """
        prepared_inputs = self._prepare_messages(
            [ChatMessage(role="user", content=prompt)], image_documents
        )
        generated_text = self._generate(prepared_inputs)
        return CompletionResponse(text=generated_text)

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

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError(
            "HuggingFaceMultiModal does not support async streaming chat yet."
        )

    async def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError(
            "HuggingFaceMultiModal does not support streaming chat yet."
        )

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError(
            "HuggingFaceMultiModal does not support async streaming completion yet."
        )

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError(
            "HuggingFaceMultiModal does not support async completion yet."
        )

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError(
            "HuggingFaceMultiModal does not support async chat yet."
        )

    async def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError(
            "HuggingFaceMultiModal does not support async completion yet."
        )

    # we check the model architecture here
    @classmethod
    def from_model_name(cls, model_name: str, **kwargs: Any) -> "HuggingFaceMultiModal":
        """Checks the model architecture and initializes the model."""
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # we check the architecture because users would want to use their own finetuned versions of VLMs
        architecture = config.architectures[0]

        if "Phi3VForCausalLM" in architecture:
            return Phi35VisionMultiModal(model_name=model_name, **kwargs)
        elif "Florence2ForConditionalGeneration" in architecture:
            return Florence2MultiModal(model_name=model_name, **kwargs)
        elif "Qwen2VLForConditionalGeneration" in architecture:
            return Qwen2VisionMultiModal(model_name=model_name, **kwargs)
        elif "PaliGemmaForConditionalGeneration" in architecture:
            return PaliGemmaMultiModal(model_name=model_name, **kwargs)
        elif "MllamaForConditionalGeneration" in architecture:
            return LlamaMultiModal(model_name=model_name, **kwargs)
        else:
            raise ValueError(
                f"Unsupported model architecture: {architecture}. "
                f"We currently support: {', '.join(SUPPORTED_VLMS)}"
            )


class Qwen2VisionMultiModal(HuggingFaceMultiModal):
    """
    A specific implementation for the Qwen2 multi-modal model.
    Handles chat-style interactions that involve both text and images.
    """

    def _prepare_messages(
        self, messages: Sequence[ChatMessage], image_documents: Sequence[ImageDocument]
    ) -> Dict[str, Any]:
        """
        Prepares the input messages and images for Qwen2 models. Images are appended in a custom format.
        """
        conversation = []
        for img_doc in image_documents:
            conversation.append(
                {"type": "image", "image": img_doc.image_path}
            )  # Append images to conversation
        conversation.append(
            {"type": "text", "text": messages[0].content}
        )  # Add user text message

        messages = [
            {"role": "user", "content": conversation}
        ]  # Wrap conversation in a user role

        # Apply a chat template to format the message with the processor
        text_prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        # Prepare the model inputs (text + images) and convert to tensor
        inputs = self._processor(
            text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt"
        )
        return inputs.to(self.device)

    def _generate(self, prepared_inputs: Dict[str, Any]) -> str:
        """
        Generates text based on prepared inputs. The text is decoded from token IDs generated by the model.
        """
        output_ids = self._model.generate(
            **prepared_inputs, max_new_tokens=self.max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(prepared_inputs["input_ids"], output_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text[0]

    async def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError(
            "Qwen2VisionMultiModal does not support async streaming chat yet."
        )


class Florence2MultiModal(HuggingFaceMultiModal):
    """
    A specific implementation for the Florence2 multi-modal model.
    Handles chat-style interactions that involve both text and images.
    """

    @override
    def complete(
        self, task: str, image_documents: ImageDocument, **kwargs: Any
    ) -> CompletionResponse:
        if isinstance(image_documents, list):
            print(
                f"{self.model_name} can handle only one image. Will continue with the first image."
            )
            image_documents = image_documents[0]

        prepared_inputs = self._prepare_messages(task, image_documents)
        generated_text = self._generate(prepared_inputs)
        return CompletionResponse(text=generated_text)

    @override
    def chat(
        self, task: str, image_documents: ImageDocument, **kwargs: Any
    ) -> ChatResponse:
        if isinstance(image_documents, list):
            print(
                f"{self.model_name} can handleo only one image. Will continue with the first image."
            )
            image_documents = image_documents[0]

        prepared_inputs = self._prepare_messages(task, image_documents)
        generated_text = self._generate(prepared_inputs)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=generated_text),
            raw={"model_output": generated_text},
        )

    # TODO: Florence2 works with task_prompts, not user prompts
    # Task prompts are: '<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>'
    def _prepare_messages(
        self, task: str, image_documents: ImageDocument
    ) -> Dict[str, Any]:
        """
        Prepares the input messages and images for Qwen2 models. Images are appended in a custom format.
        """
        if isinstance(image_documents, list):
            print(
                f"{self.model_name} can handleo only one image. Will continue with the first image."
            )
            image_documents = image_documents[0]
        prompt = (
            task.upper()
            if task.upper()
            in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
            else "<DETAILED_CAPTION>"
        )
        images = Image.open(image_documents.image_path)
        inputs = self._processor(text=prompt, images=images, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        return {
            "prompt": prompt,
            "inputs": inputs,
            "image_size": (images.width, images.height),
        }

    def _generate(self, prepared_inputs: Dict[str, Any]) -> str:
        """
        Generates text based on prepared inputs. The text is decoded from token IDs generated by the model.
        """
        inputs = prepared_inputs["inputs"]
        image_size = prepared_inputs["image_size"]
        task = prepared_inputs["prompt"]

        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.max_new_tokens,
            num_beams=3,
            do_sample=False,
        )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Use image_size from prepared_inputs to avoid storing self.image
        parsed_answer = self._processor.post_process_generation(
            generated_text, task=task, image_size=image_size
        )
        return parsed_answer[task]

    async def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError(
            "Florence2MultiModal do not support async streaming chat yet."
        )


class Phi35VisionMultiModal(HuggingFaceMultiModal):
    """
    A specific implementation for the Phi3.5 multi-modal model.
    Handles chat-style interactions that involve both text and images.
    """

    def _prepare_messages(
        self, message: ChatMessage, image_documents: Sequence[ImageDocument]
    ) -> Dict[str, Any]:
        """
        Prepares the input messages and images for Phi3.5 models. Images are appended in a custom format.
        """
        images = [Image.open(img_doc.image_path) for img_doc in image_documents]
        placeholder = "".join(f"<|image_{i + 1}|>\n" for i in range(len(images)))

        chat_messages = [{"role": message.role, "content": message.content}]
        if images:
            chat_messages[-1]["content"] = placeholder + chat_messages[-1]["content"]

        prompt = self._processor.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        return self._processor(prompt, images, return_tensors="pt").to(self.device)

    def _generate(self, prepared_inputs: Dict[str, Any]) -> str:
        """
        Generates text based on prepared inputs. The text is decoded from token IDs generated by the model.
        """
        generate_ids = self._model.generate(
            **prepared_inputs,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=False,
        )
        generate_ids = generate_ids[:, prepared_inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    async def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError(
            "Phi35VisionMultiModal does not support async streaming chat yet."
        )


class PaliGemmaMultiModal(HuggingFaceMultiModal):
    """
    A specific implementation for the PaliGemma multi-modal model.
    Handles chat-style interactions that involve both text and images.
    """

    @override
    def complete(
        self, task: str, image_documents: ImageDocument, **kwargs: Any
    ) -> CompletionResponse:
        if isinstance(image_documents, list):
            print(
                f"{self.model_name} can handle only one image. Will continue with the first image."
            )
            image_documents = image_documents[0]

        prepared_inputs = self._prepare_messages(task, image_documents)
        generated_text = self._generate(prepared_inputs)
        return CompletionResponse(text=generated_text)

    @override
    def chat(
        self, task: str, image_documents: ImageDocument, **kwargs: Any
    ) -> ChatResponse:
        if isinstance(image_documents, list):
            print(
                f"{self.model_name} can handle only one image. Will continue with the first image."
            )
            image_documents = image_documents[0]

        prepared_inputs = self._prepare_messages(task, image_documents)
        generated_text = self._generate(prepared_inputs)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=generated_text),
            raw={"model_output": generated_text},
        )

    def _prepare_messages(
        self, messages: ChatMessage, image_documents: ImageDocument
    ) -> Dict[str, Any]:
        """
        Prepares the input messages and images for PaliGemma models. Images are appended in a custom format.
        """
        if isinstance(image_documents, list):
            print(
                f"{self.model_name} can handleo only one image. Will continue with the first image."
            )
            image_documents = image_documents[0]
        images = Image.open(image_documents.image_path)
        inputs = self._processor(text=messages, images=images, return_tensors="pt").to(
            self.device
        )
        input_len = inputs["input_ids"].shape[-1]
        return {"inputs": inputs, "input_len": input_len}

    def _generate(self, prepared_inputs: Dict[str, Any]) -> str:
        """
        Generates text based on prepared inputs. The text is decoded from token IDs generated by the model.
        """
        input_len = prepared_inputs["input_len"]
        inputs = prepared_inputs["inputs"]
        generation = self._model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        return self._processor.decode(generation, skip_special_tokens=True)

    async def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError(
            "PaliGemmaMultiModal does not support async streaming chat yet."
        )


class LlamaMultiModal(HuggingFaceMultiModal):
    """
    A specific implementation for the Llama3.2 multi-modal model.
    Handles chat-style interactions that involve both text and images.
    """

    def _prepare_messages(
        self, messages: Sequence[ChatMessage], image_documents: Sequence[ImageDocument]
    ) -> Dict[str, Any]:
        """
        Prepares the input messages and images for Llama3.2 models. Images are appended in a custom format.
        """
        prompt = messages[0].content
        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]
        images = []

        for img_doc in image_documents:
            messages[0]["content"].append({"type": "image"})
            images.append(Image.open(img_doc.image_path))

        messages[0]["content"].append({"type": "text", "text": prompt})

        # Apply a chat template to format the message with the processor
        input_text = self._processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # If no images are present then we should pass None to deactivate image processing in the processor
        if len(images) == 0:
            images = None

        # Prepare the model inputs (text + images) and convert to tensor
        inputs = self._processor(images, input_text, return_tensors="pt")
        return inputs.to(self.device)

    def _generate(self, prepared_inputs: Dict[str, Any]) -> str:
        """
        Generates text based on prepared inputs. The text is decoded from token IDs generated by the model.
        """
        output_ids = self._model.generate(
            **prepared_inputs, max_new_tokens=self.max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(prepared_inputs["input_ids"], output_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text[0]
