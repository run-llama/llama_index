from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import urllib.parse
import logging
import re

import httpx
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.core.schema import ImageNode
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)
import warnings
import base64
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import requests

import httpx
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.schema import ImageDocument
import os
from llama_index.multi_modal.nvidia.utils import (
    BASE_URL,
    KNOWN_URLS,
    NVIDIA_MULTI_MODAL_MODELS,
)

logger = logging.getLogger(__name__)


def infer_image_mimetype_from_file_path(image_file_path: str) -> str:
    # Get the file extension
    file_extension = image_file_path.split(".")[-1].lower()

    # Map file extensions to mimetypes
    # Claude 3 support the base64 source type for images, and the image/jpeg, image/png, image/gif, and image/webp media types.\
    if file_extension in ["jpg", "jpeg", "png"]:
        return file_extension
    return "jpeg"


def _process_for_vlm(
    inputs: List[Dict[str, Any]],
    model: Optional[str],  # not optional, Optional for type alignment
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Process inputs for NVIDIA VLM models.

    This function processes the input messages for NVIDIA VLM models.
    It extracts asset IDs from the input messages and adds them to the
    headers for the NVIDIA VLM API.
    """
    if not model:
        return inputs, {}

    extra_headers = {}
    asset_ids = []
    for input in inputs:
        if "content" in input:
            asset_ids.extend(_nv_vlm_get_asset_ids(input["content"]))
    if asset_ids:
        extra_headers["NVCF-INPUT-ASSET-REFERENCES"] = ",".join(asset_ids)
    # inputs = [_nv_vlm_adjust_input(message, model.model_type) for message in inputs]
    return inputs, extra_headers


def generate_nvidia_multi_modal_chat_message(
    prompt: str,
    role: str,
    image_documents: Optional[Sequence[ImageDocument]] = None,
) -> List[Dict[str, Any]]:
    # If image_documents is None, return a text-only chat message
    completion_content = []

    # Process each image document
    for image_document in image_documents:
        image_content = create_image_content(image_document)
        if image_content:
            completion_content.append(image_content)

    # Append the text prompt to the completion content
    completion_content.append({"type": "text", "text": prompt})

    return [{"role": role, "content": completion_content}]


def create_image_content(image_document) -> Optional[Dict[str, Any]]:
    """
    Create the image content based on the provided image document.
    """
    # if image_document.asset_id:
    #     return _nv_vlm_get_asset_ids(image_document.image_path)

    # if image_document.image_path:
    #     return create_image_from_path(image_document.image_path)

    # if "file_path" in image_document.metadata:
    #     return create_image_from_path(image_document.metadata["file_path"])

    if image_document.image_url and image_document.image_url != "":
        mimetype = infer_image_mimetype_from_file_path(image_document.image_url)
        data = base64.b64encode(httpx.get(image_document.image_url).content).decode(
            "utf-8"
        )
        return {
            "type": "text",
            "text": f'<img src="data:image/{mimetype};base64,{data}" />',
        }

    # if image_document.image:
    #     return _is_url(image_document.image)

    return None


def _is_url(s: str) -> bool:
    try:
        result = urllib.parse.urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug(f"Unable to parse URL: {e}")
        return False


def _url_to_b64_string(image_source: str) -> str:
    try:
        if _is_url(image_source):
            return image_source
            # import sys
            # import io
            # try:
            #     import PIL.Image
            #     has_pillow = True
            # except ImportError:
            #     has_pillow = False
            # def _resize_image(img_data: bytes, max_dim: int = 1024) -> str:
            #     if not has_pillow:
            #         print(
            #             "Pillow is required to resize images down to reasonable scale."
            #             " Please install it using `pip install pillow`."
            #             " For now, not resizing; may cause NVIDIA API to fail."
            #         )
            #         return base64.b64encode(img_data).decode("utf-8")
            #     image = PIL.Image.open(io.BytesIO(img_data))
            #     max_dim_size = max(image.size)
            #     aspect_ratio = max_dim / max_dim_size
            #     new_h = int(image.size[1] * aspect_ratio)
            #     new_w = int(image.size[0] * aspect_ratio)
            #     resized_image = image.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            #     output_buffer = io.BytesIO()
            #     resized_image.save(output_buffer, format="JPEG")
            #     output_buffer.seek(0)
            #     resized_b64_string = base64.b64encode(output_buffer.read()).decode("utf-8")
            #     return resized_b64_string
            # b64_template = "data:image/png;base64,{b64_string}"
            # response = requests.get(
            #     image_source, headers={"User-Agent": "langchain-nvidia-ai-endpoints"}
            # )
            # response.raise_for_status()
            # encoded = base64.b64encode(response.content).decode("utf-8")
            # if sys.getsizeof(encoded) > 200000:
            #     ## (VK) Temporary fix. NVIDIA API has a limit of 250KB for the input.
            #     encoded = _resize_image(response.content)
            # return b64_template.format(b64_string=encoded)
        elif image_source.startswith("data:image"):
            return image_source
        elif os.path.exists(image_source):
            with open(image_source, "rb") as f:
                image_data = f.read()
                import imghdr

                image_type = imghdr.what(None, image_data)
                encoded = base64.b64encode(image_data).decode("utf-8")
                return f"data:image/{image_type};base64,{encoded}"
        else:
            raise ValueError(
                "The provided string is not a valid URL, base64, or file path."
            )
    except Exception as e:
        raise ValueError(f"Unable to process the provided image source: {e}")


def _nv_vlm_get_asset_ids(
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> List[str]:
    """
    VLM APIs accept asset IDs as input in two forms:
     - content = [{"image_url": {"url": "data:image/{type};asset_id,{asset_id}"}}*]
     - content = .*<img src="data:image/{type};asset_id,{asset_id}"/>.*
    This function extracts asset IDs from the message content.
    """

    def extract_asset_id(data: str) -> List[str]:
        pattern = re.compile(r'data:image/[^;]+;asset_id,([^"\'\s]+)')
        return pattern.findall(data)

    asset_ids = []
    if isinstance(content, str):
        asset_ids.extend(extract_asset_id(content))
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                asset_ids.extend(extract_asset_id(part))
            elif isinstance(part, dict) and "image_url" in part:
                image_url = part["image_url"]
                if isinstance(image_url, dict) and "url" in image_url:
                    asset_ids.extend(extract_asset_id(image_url["url"]))

    return asset_ids


class NVIDIAClient:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = BASE_URL,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = 10,
    ):
        self.api_key = api_key
        self.api_base = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_model_details(self, model_name: str) -> requests.Response:
        """
        Get model details from DeepInfra API.
        If the model does not exist, a 404 response is returned.

        Returns:
            requests.Response: The API response.
        """
        request_url = self.get_url(f"models/{model_name}")
        return requests.get(request_url, headers=self._get_headers())

    def request(
        self, endpoint: str, messages: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Perform a synchronous request to the DeepInfra API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            messages (Dict[str, Any]): The request payload.

        Returns:
            Dict[str, Any]: The API response.
        """

        def perform_request():
            url = f"https://ai.api.nvidia.com/v1/vlm/{endpoint}"
            payload = {
                "messages": messages,
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "stream": False,
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

        return perform_request()


class NVIDIAMultiModal(MultiModalLLM):
    model: str = Field(description="The Multi-Modal model to use from NVIDIA.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: Optional[int] = Field(
        description=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt",
        gt=0,
    )
    context_window: Optional[int] = Field(
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries.",
        ge=0,
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout, in seconds, for API requests.",
        ge=0,
    )
    api_key: str = Field(default=None, description="The NVIDIA API key.", exclude=True)
    system_prompt: str = Field(default="", description="System Prompt.")
    api_base: str = Field(default=None, description="The base URL for NVIDIA API.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the NVIDIA API."
    )
    default_headers: Optional[Dict[str, str]] = Field(
        default=None, description="The default headers for API requests."
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()
    _client: Optional[NVIDIAClient] = PrivateAttr()
    _is_hosted: bool = PrivateAttr(True)
    _mode: str = PrivateAttr(default="nvidia")

    def __init__(
        self,
        model: str = "microsoft/phi-3-vision-128k-instruct",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 300,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: Optional[int] = DEFAULT_CONTEXT_WINDOW,
        max_retries: int = 3,
        timeout: float = 60.0,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = BASE_URL,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        system_prompt: Optional[str] = "",
        **kwargs: Any,
    ) -> None:
        api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        is_hosted = base_url in KNOWN_URLS
        if base_url not in KNOWN_URLS:
            base_url = self._validate_url(base_url)

        if is_hosted and api_key == "NO_API_KEY_PROVIDED":
            warnings.warn(
                "An API key is required for the hosted NIM. This will become an error in 0.2.0.",
            )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs or {},
            context_window=context_window,
            max_retries=max_retries,
            timeout=timeout,
            api_key=api_key,
            api_base=base_url,
            callback_manager=callback_manager,
            default_headers=default_headers,
            system_promt=system_prompt,
            **kwargs,
        )
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)
        self._client = self._get_clients(**kwargs)

    def _get_clients(self, **kwargs: Any) -> NVIDIAClient:
        return NVIDIAClient(**self._get_credential_kwargs())

    @classmethod
    def class_name(cls) -> str:
        return "nvidia_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            num_output=self.max_tokens or DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    def _get_credential_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        credential_kwargs = {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            **kwargs,
        }

        if self.default_headers:
            credential_kwargs["default_headers"] = self.default_headers

        return credential_kwargs

    def _get_multi_modal_chat_messages(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[ImageNode],
        **kwargs: Any,
    ) -> List[Dict]:
        return generate_nvidia_multi_modal_chat_message(
            prompt=prompt,
            role=role,
            image_documents=image_documents,
        )

    # Model Params for NVIDIA Multi Modal model.
    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        if self.model not in NVIDIA_MULTI_MODAL_MODELS:
            raise ValueError(
                f"Invalid model {self.model}. "
                f"Available models are: {list(NVIDIA_MULTI_MODAL_MODELS.keys())}"
            )
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        if self.max_tokens is not None:
            base_kwargs["max_tokens"] = self.max_tokens
        return {**base_kwargs, **self.additional_kwargs}

    def _get_response_token_counts(self, raw_response: Any) -> dict:
        """Get the token usage reported by the response."""
        if not isinstance(raw_response, dict):
            return {}

        usage = raw_response.get("usage", {})
        # NOTE: other model providers that use the NVIDIA client may not report usage
        if usage is None:
            return {}

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def _complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        response = self._client.request(
            endpoint=self.model,
            messages=message_dict,
            **all_kwargs,
        )
        text = response["choices"][0]["message"]["content"]
        return CompletionResponse(
            text=text,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        def gen() -> CompletionResponseGen:
            text = ""

            for response in self._client.messages.create(
                messages=message_dict,
                stream=True,
                system=self.system_prompt,
                **all_kwargs,
            ):
                if isinstance(response, ContentBlockDeltaEvent):
                    # update using deltas
                    content_delta = response.delta.text or ""
                    text += content_delta

                    yield CompletionResponse(
                        delta=content_delta,
                        text=text,
                        raw=response,
                        additional_kwargs=self._get_response_token_counts(response),
                    )

        return gen()

    def complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt, image_documents, **kwargs)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        return self._stream_complete(prompt, image_documents, **kwargs)

    def chat(
        self,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("This function is not yet implemented.")

    def stream_chat(
        self,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("This function is not yet implemented.")

    # ===== Async Endpoints =====

    async def _acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )
        response = await self._aclient.messages.create(
            messages=message_dict,
            stream=False,
            system=self.system_prompt,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.content[0].text,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        return await self._acomplete(prompt, image_documents, **kwargs)

    async def _astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        async def gen() -> CompletionResponseAsyncGen:
            text = ""

            async for response in await self._aclient.messages.create(
                messages=message_dict,
                stream=True,
                system=self.system_prompt,
                **all_kwargs,
            ):
                if isinstance(response, ContentBlockDeltaEvent):
                    # update using deltas
                    content_delta = response.delta.text or ""
                    text += content_delta

                    yield CompletionResponse(
                        delta=content_delta,
                        text=text,
                        raw=response,
                        additional_kwargs=self._get_response_token_counts(response),
                    )

        return gen()

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await self._astream_complete(prompt, image_documents, **kwargs)

    async def achat(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This function is not yet implemented.")

    async def astream_chat(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This function is not yet implemented.")
