from __future__ import annotations

import base64
import logging
from abc import ABC

from enum import Enum
from io import IOBase, BytesIO
from pathlib import Path
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Union,
    cast,
)

import filetype
from tinytag import TinyTag, UnsupportedFormatError
from typing_extensions import Self

from llama_index.core.async_utils import asyncio_run
from llama_index.core.bridge.pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    field_serializer,
    field_validator,
    model_validator,
)
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.schema import ImageDocument
from llama_index.core.utils import get_tokenizer, resolve_binary

_logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"
    MODEL = "model"


class BaseContentBlock(ABC, BaseModel):
    @classmethod
    async def amerge(
        cls, splits: List[Self], chunk_size: int, tokenizer: Any | None = None
    ) -> list[Self]:
        """
        Async merge smaller content blocks into larger blocks up to chunk_size tokens.
        Default implementation returns splits without merging, should be overridden by subclasses that support merging.
        """
        return splits

    @classmethod
    def merge(
        cls, splits: List[Self], chunk_size: int, tokenizer: Any | None = None
    ) -> list[Self]:
        """Merge smaller content blocks into larger blocks up to chunk_size tokens."""
        return asyncio_run(
            cls.amerge(splits=splits, chunk_size=chunk_size, tokenizer=tokenizer)
        )

    async def aestimate_tokens(self, tokenizer: Any | None = None) -> int:
        """
        Async estimate the number of tokens in this content block.

        Default implementation returns 0, should be overridden by subclasses to provide meaningful estimates.
        """
        return 0

    def estimate_tokens(self, tokenizer: Any | None = None) -> int:
        """Estimate the number of tokens in this content block."""
        return asyncio_run(self.aestimate_tokens(tokenizer=tokenizer))

    async def asplit(
        self, max_tokens: int, overlap: int = 0, tokenizer: Any | None = None
    ) -> List[Self]:
        """
        Async split the content block into smaller blocks with up to max_tokens tokens each.

        Default implementation returns self in a list, should be overridden by subclasses that support splitting.
        """
        return [self]

    def split(
        self, max_tokens: int, overlap: int = 0, tokenizer: Any | None = None
    ) -> List[Self]:
        """Split the content block into smaller blocks with up to max_tokens tokens each."""
        return asyncio_run(
            self.asplit(max_tokens=max_tokens, overlap=overlap, tokenizer=tokenizer)
        )

    async def atruncate(
        self, max_tokens: int, tokenizer: Any | None = None, reverse: bool = False
    ) -> Self:
        """Async truncate the content block to up to max_tokens tokens."""
        tknizer = tokenizer or get_tokenizer()
        estimated_tokens = await self.aestimate_tokens(tokenizer=tknizer)
        if estimated_tokens <= max_tokens:
            return self

        split_blocks = await self.asplit(max_tokens=max_tokens, tokenizer=tknizer)
        return split_blocks[0] if not reverse else split_blocks[-1]

    def truncate(
        self, max_tokens: int, tokenizer: Any | None = None, reverse: bool = False
    ) -> Self:
        """Truncate the content block to up to max_tokens tokens."""
        return asyncio_run(
            self.atruncate(max_tokens=max_tokens, tokenizer=tokenizer, reverse=reverse)
        )

    @property
    def templatable_attributes(self) -> List[str]:
        """
        List of attributes that can be templated.

        Can be overridden by subclasses.
        """
        return []

    @staticmethod
    def _get_template_str_from_attribute(attribute: Any) -> str | None:
        """
        Helper function to get template string from attribute.

        It primarily enables cases of template_vars in binary strings for non text types such as:
            - ImageBlock(image=b'{image_bytes}')
            - AudioBlock(audio=b'{audio_bytes}')
            - VideoBlock(video=b'{video_bytes}')
            - DocumentBlock(data=b'{document_bytes}')

        However, it could in theory also work with other attributes like:
            - ImageBlock(path=b'{image_path}')
            - AudioBlock(url=b'{audio_url}')

        For that to work, the validation on those fields would need to be updated though.
        """
        if attribute is None:
            return None
        if isinstance(attribute, str):
            return attribute
        elif isinstance(attribute, bytes):
            try:
                return resolve_binary(attribute).read().decode("utf-8")
            except UnicodeDecodeError:
                return None
        else:
            return str(attribute)

    def get_template_vars(self) -> list[str]:
        """
        Get template variables from the content block.
        """
        from llama_index.core.prompts.utils import get_template_vars

        for attribute_name in self.templatable_attributes:
            attribute = getattr(self, attribute_name, None)
            template_str = self._get_template_str_from_attribute(attribute)
            if template_str:
                return get_template_vars(template_str)
        return []

    def format_vars(self, **kwargs: Any) -> "BaseContentBlock":
        """
        Format the content block with the given keyword arguments.

        This function primarily enables formatting of template_vars in Textblocks and binary strings for non text:
            - ImageBlock(image=b'{image_bytes}')
            - AudioBlock(audio=b'{audio_bytes}')
            - VideoBlock(video=b'{video_bytes}')
            - DocumentBlock(data=b'{document_bytes}')

        However, it could in theory also work with other attributes like:
            - ImageBlock(path=b'{image_path}')
            - AudioBlock(url=b'{audio_url}')

        For that to work, the validation on those fields would need to be updated though.
        """
        from llama_index.core.prompts.utils import format_string

        formatted_attrs: Dict[str, Any] = {}
        for attribute_name in self.templatable_attributes:
            attribute = getattr(self, attribute_name, None)
            att_type = type(attribute)
            template_str = self._get_template_str_from_attribute(attribute)
            # If the attribute is a binary string, we need to coerce to string for formatting,
            # but then we need to re-encode to bytes after formatting, which is what the code below does.
            formatted_kwargs = {
                k: resolve_binary(v, as_base64=True).read().decode()
                if isinstance(v, bytes)
                else v
                for k, v in kwargs.items()
            }
            if template_str:
                formatted_str = format_string(template_str, **formatted_kwargs)
                if att_type is str:
                    formatted_attrs[attribute_name] = formatted_str
                elif att_type is bytes:
                    formatted_attrs[attribute_name] = formatted_str.encode()
                else:
                    try:
                        formatted_attrs[attribute_name] = att_type(formatted_str)  # type: ignore
                    except Exception:
                        raise ValueError(
                            "Could not format attribute {attribute_name} with value {template_str} to type {att_type}"
                        )
        return type(self).model_validate(self.model_copy(update=formatted_attrs))

    @staticmethod
    def mimetype_from_inline_url(url: str) -> filetype.Type | None:
        if url.startswith("data:"):
            try:
                mimetype = url.split(";base64,")[0].split("data:")[1]
                return filetype.get_type(mime=mimetype)
            except Exception:
                try:
                    data = url.split(";base64,")[1]
                    decoded_data = base64.b64decode(data)
                    return filetype.guess(decoded_data)
                except Exception:
                    return None
        return None


class TextBlock(BaseContentBlock):
    """A representation of text data to directly pass to/from the LLM."""

    block_type: Literal["text"] = "text"
    text: str

    @classmethod
    async def amerge(
        cls, splits: List["TextBlock"], chunk_size: int, tokenizer: Any | None = None
    ) -> list["TextBlock"]:
        merged_blocks = []
        current_block_texts = []
        current_block_tokens = 0

        # TODO: Think about separators when merging, since correctly joining them requires us to understand how they
        #  were previously split. For now, we just universally join with spaces.
        for split in splits:
            split_tokens = await split.aestimate_tokens(tokenizer=tokenizer)

            if current_block_tokens + split_tokens <= chunk_size:
                current_block_texts.append(split.text)
                current_block_tokens += split_tokens
            else:
                merged_blocks.append(TextBlock(text=" ".join(current_block_texts)))
                current_block_texts = [split.text]
                current_block_tokens = split_tokens

        if current_block_texts:
            merged_blocks.append(TextBlock(text=" ".join(current_block_texts)))

        return merged_blocks

    async def aestimate_tokens(self, tokenizer: Any | None = None) -> int:
        tknizer = tokenizer or get_tokenizer()
        return len(tknizer(self.text))

    async def asplit(
        self, max_tokens: int, overlap: int = 0, tokenizer: Any | None = None
    ) -> List["TextBlock"]:
        from llama_index.core.node_parser import TokenTextSplitter

        text_splitter = TokenTextSplitter(
            chunk_size=max_tokens, chunk_overlap=overlap, tokenizer=tokenizer
        )
        chunks = text_splitter.split_text(self.text)
        return [TextBlock(text=chunk) for chunk in chunks]

    @property
    def templatable_attributes(self) -> list[str]:
        return ["text"]


class ImageBlock(BaseContentBlock):
    """A representation of image data to directly pass to/from the LLM."""

    block_type: Literal["image"] = "image"
    image: bytes | IOBase | None = None
    path: FilePath | None = None
    url: AnyUrl | str | None = None
    image_mimetype: str | None = None
    detail: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("url", mode="after")
    @classmethod
    def urlstr_to_anyurl(cls, url: str | AnyUrl | None) -> AnyUrl | None:
        """Store the url as Anyurl."""
        if isinstance(url, AnyUrl):
            return url
        if url is None:
            return None

        return AnyUrl(url=url)

    @field_serializer("image")
    def serialize_image(self, image: bytes | IOBase | None) -> bytes | None:
        """Serialize the image field."""
        if isinstance(image, bytes):
            return image
        if isinstance(image, IOBase):
            image.seek(0)
            return image.read()
        return None

    @model_validator(mode="after")
    def image_to_base64(self) -> Self:
        """
        Store the image as base64 and guess the mimetype when possible.

        In case the model was built passing image data but without a mimetype,
        we try to guess it using the filetype library. To avoid resource-intense
        operations, we won't load the path or the URL to guess the mimetype.
        """
        if not self.image or not isinstance(self.image, bytes):
            if not self.image_mimetype:
                path = self.path or self.url
                if path:
                    suffix = Path(str(path)).suffix.replace(".", "") or None
                    mimetype = filetype.get_type(ext=suffix)
                    if not mimetype or not mimetype.mime:
                        mimetype = self.mimetype_from_inline_url(str(path))
                    if mimetype and str(mimetype.mime).startswith("image/"):
                        self.image_mimetype = str(mimetype.mime)

            return self

        self._guess_mimetype(resolve_binary(self.image).read())
        self.image = resolve_binary(self.image, as_base64=True).read()
        return self

    def _guess_mimetype(self, img_data: bytes) -> None:
        if not self.image_mimetype:
            guess = filetype.guess(img_data)
            self.image_mimetype = guess.mime if guess else None

    def resolve_image(self, as_base64: bool = False) -> IOBase:
        """
        Resolve an image such that PIL can read it.

        Args:
            as_base64 (bool): whether the resolved image should be returned as base64-encoded bytes

        """
        data_buffer = (
            self.image
            if isinstance(self.image, IOBase)
            else resolve_binary(
                raw_bytes=self.image,
                path=self.path,
                url=str(self.url) if self.url else None,
                as_base64=as_base64,
            )
        )

        # Check size by seeking to end and getting position
        data_buffer.seek(0, 2)  # Seek to end
        size = data_buffer.tell()
        data_buffer.seek(0)  # Reset to beginning

        if size == 0:
            raise ValueError("resolve_image returned zero bytes")
        return data_buffer

    def inline_url(self) -> str:
        b64 = self.resolve_image(as_base64=True)
        b64_str = b64.read().decode("utf-8")
        return f"data:{self.image_mimetype};base64,{b64_str}"

    async def aestimate_tokens(self, *args: Any, **kwargs: Any) -> int:
        """
        Many APIs measure images differently. Here, we take a large estimate.

        This is based on a 2048 x 1536 image using OpenAI.

        TODO: In the future, LLMs should be able to count their own tokens.
        """
        try:
            self.resolve_image()
            return 2125
        except ValueError as e:
            # Null case
            if str(e) == "resolve_image returned zero bytes":
                return 0
            raise

    @property
    def templatable_attributes(self) -> list[str]:
        return ["image"]


class AudioBlock(BaseContentBlock):
    """A representation of audio data to directly pass to/from the LLM."""

    block_type: Literal["audio"] = "audio"
    audio: bytes | IOBase | None = None
    path: FilePath | None = None
    url: AnyUrl | str | None = None
    format: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("url", mode="after")
    @classmethod
    def urlstr_to_anyurl(cls, url: str | AnyUrl) -> AnyUrl:
        """Store the url as Anyurl."""
        if isinstance(url, AnyUrl):
            return url
        return AnyUrl(url=url)

    @field_serializer("audio")
    def serialize_audio(self, audio: bytes | IOBase | None) -> bytes | None:
        """Serialize the audio field."""
        if isinstance(audio, bytes):
            return audio
        if isinstance(audio, IOBase):
            audio.seek(0)
            return audio.read()
        return None

    @model_validator(mode="after")
    def audio_to_base64(self) -> Self:
        """
        Store the audio as base64 and guess the mimetype when possible.

        In case the model was built passing audio data but without a format,
        we try to guess it using the filetype library. To avoid resource-intense
        operations, we won't load the path or the URL to guess the format.
        """
        if not self.audio or not isinstance(self.audio, bytes):
            if not self.format:
                path = self.path or self.url
                if path:
                    suffix = Path(str(path)).suffix.replace(".", "") or None
                    mimetype = filetype.get_type(ext=suffix)
                    if not mimetype or not mimetype.mime:
                        mimetype = self.mimetype_from_inline_url(str(path))
                    if mimetype and str(mimetype.mime).startswith("audio/"):
                        self.format = str(mimetype.extension)

            return self

        self._guess_format(resolve_binary(self.audio).read())
        self.audio = resolve_binary(self.audio, as_base64=True).read()
        return self

    def _guess_format(self, audio_data: bytes) -> None:
        if not self.format:
            guess = filetype.guess(audio_data)
            self.format = guess.extension if guess else None

    def resolve_audio(self, as_base64: bool = False) -> IOBase:
        """
        Resolve an audio such that PIL can read it.

        Args:
            as_base64 (bool): whether the resolved audio should be returned as base64-encoded bytes

        """
        data_buffer = (
            self.audio
            if isinstance(self.audio, IOBase)
            else resolve_binary(
                raw_bytes=self.audio,
                path=self.path,
                url=str(self.url) if self.url else None,
                as_base64=as_base64,
            )
        )
        # Check size by seeking to end and getting position
        data_buffer.seek(0, 2)  # Seek to end
        size = data_buffer.tell()
        data_buffer.seek(0)  # Reset to beginning

        if size == 0:
            raise ValueError("resolve_audio returned zero bytes")
        return data_buffer

    def inline_url(self) -> str:
        b64 = self.resolve_audio(as_base64=True)
        b64_str = b64.read().decode("utf-8")
        if self.format:
            mimetype = filetype.get_type(ext=self.format).mime
            if mimetype:
                return f"data:{mimetype};base64,{b64_str}"
        return f"data:audio;base64,{b64_str}"

    async def aestimate_tokens(self, *args: Any, **kwargs: Any) -> int:
        """
        Use TinyTag to estimate the duration of the audio file and convert to tokens.

        Gemini estimates 32 tokens per second of audio
        https://ai.google.dev/gemini-api/docs/tokens?lang=python

        OpenAI estimates 1 token per 0.1 second for user input and 1 token per 0.05 seconds for assistant output
        https://platform.openai.com/docs/guides/realtime-costs
        """
        try:
            # First try tinytag
            try:
                tag = TinyTag.get(file_obj=cast(BytesIO, self.resolve_audio()))
                if duration := tag.duration:
                    # We conservatively return the max estimate
                    return max((int(duration) + 1) * 32, int(duration / 0.05) + 1)
            except UnsupportedFormatError:
                _logger.info(
                    "TinyTag does not support file type for video token estimation."
                )
            return 256  # fallback
        except ValueError as e:
            # Null case
            if str(e) == "resolve_audio returned zero bytes":
                return 0
            raise

    @property
    def templatable_attributes(self) -> list[str]:
        return ["audio"]


class VideoBlock(BaseContentBlock):
    """A representation of video data to directly pass to/from the LLM."""

    block_type: Literal["video"] = "video"
    video: bytes | IOBase | None = None
    path: FilePath | None = None
    url: AnyUrl | str | None = None
    video_mimetype: str | None = None
    detail: str | None = None
    fps: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("url", mode="after")
    @classmethod
    def urlstr_to_anyurl(cls, url: str | AnyUrl | None) -> AnyUrl | None:
        """Store the url as AnyUrl."""
        if isinstance(url, AnyUrl):
            return url
        if url is None:
            return None
        return AnyUrl(url=url)

    @field_serializer("video")
    def serialize_video(self, video: bytes | IOBase | None) -> bytes | None:
        """Serialize the video field."""
        if isinstance(video, bytes):
            return video
        if isinstance(video, IOBase):
            video.seek(0)
            return video.read()
        return None

    @model_validator(mode="after")
    def video_to_base64(self) -> "VideoBlock":
        """
        Store the video as base64 and guess the mimetype when possible.

        If video data is passed but no mimetype is provided, try to infer it.
        """
        if not self.video or not isinstance(self.video, bytes):
            if not self.video_mimetype:
                path = self.path or self.url
                if path:
                    suffix = Path(str(path)).suffix.replace(".", "") or None
                    mimetype = filetype.get_type(ext=suffix)
                    if not mimetype or not mimetype.mime:
                        mimetype = self.mimetype_from_inline_url(str(path))
                    if mimetype and str(mimetype.mime).startswith("video/"):
                        self.video_mimetype = str(mimetype.mime)
            return self

        self._guess_mimetype(resolve_binary(self.video).read())
        self.video = resolve_binary(self.video, as_base64=True).read()
        return self

    def _guess_mimetype(self, vid_data: bytes) -> None:
        if not self.video_mimetype:
            guess = filetype.guess(vid_data)
            if guess and guess.mime.startswith("video/"):
                self.video_mimetype = guess.mime

    def resolve_video(self, as_base64: bool = False) -> IOBase:
        """
        Resolve a video file to a IOBase buffer.

        Args:
            as_base64 (bool): whether to return the video as base64-encoded bytes

        """
        data_buffer = (
            self.video
            if isinstance(self.video, IOBase)
            else resolve_binary(
                raw_bytes=self.video,
                path=self.path,
                url=str(self.url) if self.url else None,
                as_base64=as_base64,
            )
        )

        # Check size by seeking to end and getting position
        data_buffer.seek(0, 2)  # Seek to end
        size = data_buffer.tell()
        data_buffer.seek(0)  # Reset to beginning

        if size == 0:
            raise ValueError("resolve_video returned zero bytes")
        return data_buffer

    def inline_url(self) -> str:
        b64 = self.resolve_video(as_base64=True)
        b64_str = b64.read().decode("utf-8")
        if self.video_mimetype:
            return f"data:{self.video_mimetype};base64,{b64_str}"
        return f"data:video;base64,{b64_str}"

    async def aestimate_tokens(self, *args: Any, **kwargs: Any) -> int:
        """
        Use TinyTag to estimate the duration of the video file and convert to tokens.

        Gemini estimates 263 tokens per second of video
        https://ai.google.dev/gemini-api/docs/tokens?lang=python
        """
        try:
            # First try tinytag
            try:
                tag = TinyTag.get(file_obj=cast(BytesIO, self.resolve_video()))
                if duration := tag.duration:
                    return (int(duration) + 1) * 263
            except UnsupportedFormatError:
                _logger.info(
                    "TinyTag does not support file type for video token estimation."
                )
            # fallback of roughly 8 times the fallback cost of audio (263 // 32; based on gemini pricing per sec)
            return 256 * 8
        except ValueError as e:
            # Null case
            if str(e) == "resolve_video returned zero bytes":
                return 0
            raise

    @property
    def templatable_attributes(self) -> list[str]:
        return ["video"]


class DocumentBlock(BaseContentBlock):
    """A representation of a document to directly pass to the LLM."""

    block_type: Literal["document"] = "document"
    data: bytes | IOBase | None = None
    path: Optional[Union[FilePath | str]] = None
    url: Optional[str] = None
    title: Optional[str] = None
    document_mimetype: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def document_validation(self) -> Self:
        self.document_mimetype = self.document_mimetype or self._guess_mimetype()

        if not self.title:
            self.title = "input_document"

        # skip data validation if no byte is provided
        if not self.data or not isinstance(self.data, bytes):
            return self

        self.data = resolve_binary(self.data, as_base64=True).read()
        return self

    @field_serializer("data")
    def serialize_data(self, data: bytes | IOBase | None) -> bytes | None:
        """Serialize the data field."""
        if isinstance(data, bytes):
            return data
        if isinstance(data, IOBase):
            data.seek(0)
            return data.read()
        return None

    def resolve_document(self) -> IOBase:
        """
        Resolve a document such that it is represented by a BufferIO object.
        """
        data_buffer = (
            self.data
            if isinstance(self.data, IOBase)
            else resolve_binary(
                raw_bytes=self.data,
                path=self.path,
                url=str(self.url) if self.url else None,
                as_base64=False,
            )
        )
        # Check size by seeking to end and getting position
        data_buffer.seek(0, 2)  # Seek to end
        size = data_buffer.tell()
        data_buffer.seek(0)  # Reset to beginning

        if size == 0:
            raise ValueError("resolve_document returned zero bytes")
        return data_buffer

    def _get_b64_bytes(self, data_buffer: IOBase) -> bytes:
        """
        Get base64-encoded bytes from a IOBase buffer.
        """
        return resolve_binary(data_buffer.read(), as_base64=True).read()

    def _get_b64_string(self, data_buffer: IOBase) -> str:
        """
        Get base64-encoded string from a IOBase buffer.
        """
        return self._get_b64_bytes(data_buffer).decode("utf-8")

    def inline_url(self) -> str:
        b64_str = self._get_b64_string(data_buffer=self.resolve_document())
        if self.document_mimetype:
            return f"data:{self.document_mimetype};base64,{b64_str}"
        return f"data:application;base64,{b64_str}"

    def guess_format(self) -> str | None:
        path = self.path or self.url
        if not path:
            return None

        return Path(str(path)).suffix.replace(".", "")

    def _guess_mimetype(self) -> str | None:
        if self.data:
            guess = filetype.guess(self.data)
            return str(guess.mime) if guess else None

        suffix = self.guess_format()
        if not suffix:
            return None

        guess = filetype.get_type(ext=suffix)
        return str(guess.mime) if guess else None

    async def aestimate_tokens(self, *args: Any, **kwargs: Any) -> int:
        try:
            self.resolve_document()
        except ValueError as e:
            # Null case
            if str(e) == "resolve_document returned zero bytes":
                return 0
            raise
        # We currently only use this fallback estimate for documents which are non zero bytes
        return 512

    @property
    def templatable_attributes(self) -> list[str]:
        return ["data"]


class CacheControl(BaseContentBlock):
    type: str
    ttl: str = Field(default="5m")


class CachePoint(BaseContentBlock):
    """Used to set the point to cache up to, if the LLM supports caching."""

    block_type: Literal["cache"] = "cache"
    cache_control: CacheControl


class BaseRecursiveContentBlock(BaseContentBlock):
    """Base class for content blocks that can contain other content blocks."""

    @classmethod
    def nested_blocks_field_name(cls) -> str:
        """
        Return the name of the field that contains nested content blocks.

        By default, this is "content", but subclasses can override this method
        """
        return "content"

    @property
    def nested_blocks(self) -> List[BaseContentBlock]:
        """Return the nested content blocks."""
        blocks = getattr(self, self.nested_blocks_field_name())
        if isinstance(blocks, str):
            blocks = TextBlock(text=blocks)
        return blocks if isinstance(blocks, list) else [blocks]

    def can_merge(self, other: Self) -> bool:
        """Check if this block can be merged with another block of the same type."""
        atts = {
            k: v
            for k, v in self.model_dump().items()
            if k != self.nested_blocks_field_name()
        }
        other_atts = {
            k: v
            for k, v in other.model_dump().items()
            if k != self.nested_blocks_field_name()
        }
        return atts == other_atts

    @staticmethod
    async def amerge_nested(
        nested_blocks: list[BaseContentBlock],
        chunk_size: int,
        tokenizer: Any | None = None,
    ) -> list[BaseContentBlock]:
        # make list of lists out of nested blocks of same type
        nested_blocks_by_type: list[list[BaseContentBlock]] = []
        for nb in nested_blocks:
            if not nested_blocks_by_type or type(
                nested_blocks_by_type[-1][0]
            ) is not type(nb):
                nested_blocks_by_type.append([nb])
            else:
                nested_blocks_by_type[-1].append(nb)

        new_nested_blocks = []
        # merge nested blocks of same type
        for nbs in nested_blocks_by_type:
            new_nested_blocks.extend(
                await type(nbs[0]).amerge(
                    nbs, chunk_size=chunk_size, tokenizer=tokenizer
                )
            )
        return new_nested_blocks

    @classmethod
    async def amerge(
        cls,
        splits: List["BaseRecursiveContentBlock"],
        chunk_size: int,
        tokenizer: Any | None = None,
    ) -> list["BaseRecursiveContentBlock"]:
        """
        First merge nested_blocks of consecutive BaseRecursiveContentBlock types based on token estimates

        Then, merge consecutive nested content blocks of the same type.
        """
        merged_blocks = []
        cur_blocks: list["BaseRecursiveContentBlock"] = []
        cur_block_tokens = 0

        for split in splits:
            split_tokens = await split.aestimate_tokens(tokenizer=tokenizer)
            can_merge = len(cur_blocks) == 0 or cur_blocks[-1].can_merge(split)
            if cur_block_tokens + split_tokens <= chunk_size and can_merge:
                cur_blocks.append(split)
                cur_block_tokens += split_tokens
            else:
                if cur_blocks:
                    attributes = cur_blocks[0].model_dump() | {
                        # Overwrite nested blocks
                        cls.nested_blocks_field_name(): await cls.amerge_nested(
                            nested_blocks=[
                                nested_block
                                for block in cur_blocks
                                for nested_block in block.nested_blocks
                            ],
                            chunk_size=chunk_size,
                            tokenizer=tokenizer,
                        )
                    }
                    merged_blocks.append(cls(**attributes))
                cur_blocks = [split]
                cur_block_tokens = split_tokens

        if cur_blocks:
            attributes = cur_blocks[0].model_dump() | {
                # Overwrite nested blocks attribute and merge nested blocks of the same type
                cls.nested_blocks_field_name(): await cls.amerge_nested(
                    nested_blocks=[
                        nested_block
                        for block in cur_blocks
                        for nested_block in block.nested_blocks
                    ],
                    chunk_size=chunk_size,
                    tokenizer=tokenizer,
                )
            }
            merged_blocks.append(cls(**attributes))

        return merged_blocks

    async def aestimate_tokens(self, *args: Any, **kwargs: Any) -> int:
        """Estimate the number of tokens in this content block."""
        return sum(
            [
                await block.aestimate_tokens(*args, **kwargs)
                for block in self.nested_blocks
            ]
        )

    async def asplit(
        self, max_tokens: int, overlap: int = 0, tokenizer: Any | None = None
    ) -> List["BaseRecursiveContentBlock"]:
        """Split the content block into smaller blocks with up to max_tokens tokens each."""
        splits = []

        cls = type(self)
        for block in self.nested_blocks:
            block_tokens = await block.aestimate_tokens(tokenizer=tokenizer)
            if block_tokens <= max_tokens:
                attributes = self.model_dump() | {
                    # Overwrite nested blocks
                    self.nested_blocks_field_name(): [block]
                }
                splits.append(cls(**attributes))
            else:
                split_blocks = await block.asplit(
                    max_tokens=max_tokens, tokenizer=tokenizer
                )
                for split_block in split_blocks:
                    attributes = self.model_dump() | {
                        # Overwrite nested blocks
                        self.nested_blocks_field_name(): [split_block]
                    }
                    splits.append(cls(**attributes))

        return splits

    async def atruncate(
        self, max_tokens: int, tokenizer: Any | None = None, reverse: bool = False
    ) -> "BaseRecursiveContentBlock":
        """Truncate the content block to have at most max_tokens tokens."""
        tknizer = tokenizer or get_tokenizer()
        current_tokens = 0
        truncated_blocks = []

        cls = type(self)
        for block in (
            self.nested_blocks if not reverse else reversed(self.nested_blocks)
        ):
            block_tokens = await block.aestimate_tokens(tokenizer=tknizer)
            if current_tokens + block_tokens <= max_tokens:
                if not reverse:
                    truncated_blocks.append(block)
                else:
                    truncated_blocks.insert(0, block)
                current_tokens += block_tokens
            else:
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 0:
                    truncated_block = await block.atruncate(
                        max_tokens=remaining_tokens, tokenizer=tknizer, reverse=reverse
                    )
                    # For some block types, truncate may return a block larger than requested
                    # However, we still want to include it if no other truncated blocks were added
                    # We leave it the user to handle cases where even the truncated block exceeds max_tokens
                    if (
                        await truncated_block.aestimate_tokens(tokenizer=tknizer)
                        <= remaining_tokens
                        or not truncated_blocks
                    ):
                        if not reverse:
                            truncated_blocks.append(truncated_block)
                        else:
                            truncated_blocks.insert(0, truncated_block)
                break  # Stop after reaching max_tokens

        attributes = self.model_dump() | {
            # Overwrite nested blocks
            self.nested_blocks_field_name(): truncated_blocks
        }
        return cls(**attributes)

    @property
    def templatable_attributes(self) -> list[str]:
        return [self.nested_blocks_field_name()]

    def get_template_vars(self) -> list[str]:
        vars = []
        for block in self.nested_blocks:
            vars.extend(block.get_template_vars())
        return vars

    def format_vars(self, **kwargs: Any) -> Self:
        formatted_blocks = []
        for block in self.nested_blocks:
            relevant_kwargs = {
                k: v for k, v in kwargs.items() if k in block.get_template_vars()
            }
            formatted_blocks.append(block.format_vars(**relevant_kwargs))
        attributes = self.model_dump() | {
            # Overwrite nested blocks
            self.nested_blocks_field_name(): formatted_blocks
        }
        return type(self)(**attributes)


class CitableBlock(BaseRecursiveContentBlock):
    """Supports providing citable content to LLMs that have built-in citation support."""

    block_type: Literal["citable"] = "citable"
    title: str
    source: str
    # TODO: We could maybe expand the types here,
    # limiting for now to known use cases
    content: List[
        Annotated[
            Union[TextBlock, ImageBlock, DocumentBlock],
            Field(discriminator="block_type"),
        ]
    ]

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [TextBlock(text=v)]
        return v


class CitationBlock(BaseRecursiveContentBlock):
    """A representation of cited content from past messages."""

    block_type: Literal["citation"] = "citation"
    cited_content: Annotated[
        Union[TextBlock, ImageBlock], Field(discriminator="block_type")
    ]
    source: str
    title: str
    additional_location_info: Dict[str, int]

    @field_validator("cited_content", mode="before")
    @classmethod
    def validate_cited_content(cls, v: Any) -> Any:
        if isinstance(v, str):
            return TextBlock(text=v)
        if isinstance(v, list):
            if len(v) != 1:
                raise ValueError(
                    "CitableBlock content must contain exactly one block when provided as a list."
                )
            value = v[0]
            if isinstance(value, str):
                return TextBlock(text=value)
            else:
                return value
        return v

    @classmethod
    def nested_blocks_field_name(self) -> str:
        return "cited_content"

    def can_merge(self, other: Self) -> bool:
        """Check if this block can be merged with another block of the same type."""
        # Only merge if cited_content is of the same type and is a TextBlock
        if type(self.cited_content) is type(other.cited_content) and isinstance(
            self.cited_content, TextBlock
        ):
            atts = {k: v for k, v in self.model_dump().items() if k != "cited_content"}
            other_atts = {
                k: v for k, v in other.model_dump().items() if k != "cited_content"
            }
            return atts == other_atts
        return False


class ThinkingBlock(BaseContentBlock):
    """
    A representation of the content streamed from reasoning/thinking processes by LLMs

    Because of LLM provider's reliance on signatures for Thought Processes,
    we do not support merging/splitting/truncating for this block, as we want to preserve the integrity of the content
    provided by the LLM.

    For the same reason, they are also not templatable.
    """

    block_type: Literal["thinking"] = "thinking"
    content: Optional[str] = Field(
        description="Content of the reasoning/thinking process, if available",
        default=None,
    )
    num_tokens: Optional[int] = Field(
        description="Number of token used for reasoning/thinking, if available",
        default=None,
    )
    additional_information: Dict[str, Any] = Field(
        description="Additional information related to the thinking/reasoning process, if available",
        default_factory=dict,
    )

    async def aestimate_tokens(self, tokenizer: Any | None = None) -> int:
        return self.num_tokens or await TextBlock(
            text=self.content or ""
        ).aestimate_tokens(tokenizer=tokenizer)


class ToolCallBlock(BaseContentBlock):
    block_type: Literal["tool_call"] = "tool_call"
    tool_call_id: Optional[str] = Field(
        default=None, description="ID of the tool call, if provided"
    )
    tool_name: str = Field(description="Name of the called tool")
    tool_kwargs: dict[str, Any] | str = Field(
        default_factory=dict,  # type: ignore
        description="Arguments provided to the tool, if available",
    )

    async def aestimate_tokens(self, *args: Any, **kwargs: Any) -> int:
        return await TextBlock(text=self.model_dump_json()).aestimate_tokens(
            *args, **kwargs
        )


ContentBlock = Annotated[
    Union[
        TextBlock,
        ImageBlock,
        AudioBlock,
        VideoBlock,
        DocumentBlock,
        CachePoint,
        CitableBlock,
        CitationBlock,
        ThinkingBlock,
        ToolCallBlock,
    ],
    Field(discriminator="block_type"),
]


class ChatMessage(BaseRecursiveContentBlock):
    """Chat message."""

    role: MessageRole = MessageRole.USER
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    blocks: list[ContentBlock] = Field(default_factory=list)

    def __init__(self, /, content: Any | None = None, **data: Any) -> None:
        """
        Keeps backward compatibility with the old `content` field.

        If content was passed and contained text, store a single TextBlock.
        If content was passed and it was a list, assume it's a list of content blocks and store it.
        """
        if content is not None:
            if isinstance(content, str):
                data["blocks"] = [TextBlock(text=content)]
            elif isinstance(content, list):
                data["blocks"] = content

        super().__init__(**data)

    @model_validator(mode="after")
    def legacy_additional_kwargs_image(self) -> Self:
        """
        Provided for backward compatibility.

        If `additional_kwargs` contains an `images` key, assume the value is a list
        of ImageDocument and convert them into image blocks.
        """
        if documents := self.additional_kwargs.get("images"):
            documents = cast(list[ImageDocument], documents)
            for doc in documents:
                img_base64_bytes = doc.resolve_image(as_base64=True).read()
                self.blocks.append(ImageBlock(image=img_base64_bytes))
        return self

    @classmethod
    def nested_blocks_field_name(self) -> str:
        return "blocks"

    @property
    def content(self) -> str | None:
        """
        Keeps backward compatibility with the old `content` field.

        Returns:
            The cumulative content of the TextBlock blocks, None if there are none.

        """
        content_strs = []
        for block in self.blocks:
            if isinstance(block, TextBlock):
                content_strs.append(block.text)

        ct = "\n".join(content_strs) or None
        if ct is None and len(content_strs) == 1:
            return ""
        return ct

    @content.setter
    def content(self, content: str) -> None:
        """
        Keeps backward compatibility with the old `content` field.

        Raises:
            ValueError: if blocks contains more than a block, or a block that's not TextBlock.

        """
        if not self.blocks:
            self.blocks = [TextBlock(text=content)]
        elif len(self.blocks) == 1 and isinstance(self.blocks[0], TextBlock):
            self.blocks = [TextBlock(text=content)]
        else:
            raise ValueError(
                "ChatMessage contains multiple blocks, use 'ChatMessage.blocks' instead."
            )

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"

    @classmethod
    def from_str(
        cls,
        content: str,
        role: Union[MessageRole, str] = MessageRole.USER,
        **kwargs: Any,
    ) -> Self:
        if isinstance(role, str):
            role = MessageRole(role)
        return cls(role=role, blocks=[TextBlock(text=content)], **kwargs)

    def _recursive_serialization(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            value.model_rebuild()  # ensures all fields are initialized and serializable
            return value.model_dump()  # type: ignore
        if isinstance(value, dict):
            return {
                key: self._recursive_serialization(value)
                for key, value in value.items()
            }
        if isinstance(value, list):
            return [self._recursive_serialization(item) for item in value]

        if isinstance(value, bytes):
            return base64.b64encode(value).decode("utf-8")

        return value

    @field_serializer("additional_kwargs", check_fields=False)
    def serialize_additional_kwargs(self, value: Any, _info: Any) -> Any:
        return self._recursive_serialization(value)


class LogProb(BaseModel):
    """LogProb of a token."""

    token: str = Field(default_factory=str)
    logprob: float = Field(default_factory=float)
    bytes: List[int] = Field(default_factory=list)


# ===== Generic Model Output - Chat =====
class ChatResponse(BaseModel):
    """Chat response."""

    message: ChatMessage
    raw: Optional[Any] = None
    delta: Optional[str] = None
    logprobs: Optional[List[List[LogProb]]] = None
    additional_kwargs: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return str(self.message)


ChatResponseGen = Generator[ChatResponse, None, None]
ChatResponseAsyncGen = AsyncGenerator[ChatResponse, None]


# ===== Generic Model Output - Completion =====
class CompletionResponse(BaseModel):
    """
    Completion response.

    Fields:
        text: Text content of the response if not streaming, or if streaming,
            the current extent of streamed text.
        additional_kwargs: Additional information on the response(i.e. token
            counts, function calling information).
        raw: Optional raw JSON that was parsed to populate text, if relevant.
        delta: New text that just streamed in (only relevant when streaming).
    """

    text: str
    additional_kwargs: dict = Field(default_factory=dict)
    raw: Optional[Any] = None
    logprobs: Optional[List[List[LogProb]]] = None
    delta: Optional[str] = None

    def __str__(self) -> str:
        return self.text


CompletionResponseGen = Generator[CompletionResponse, None, None]
CompletionResponseAsyncGen = AsyncGenerator[CompletionResponse, None]


class LLMMetadata(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "Total number of tokens the model can be input and output for one response."
        ),
    )
    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of tokens the model can output when generating a response.",
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            "Set True if the model exposes a chat interface (i.e. can be passed a"
            " sequence of messages, rather than text), like OpenAI's"
            " /v1/chat/completions endpoint."
        ),
    )
    is_function_calling_model: bool = Field(
        default=False,
        # SEE: https://openai.com/blog/function-calling-and-other-api-updates
        description=(
            "Set True if the model supports function calling messages, similar to"
            " OpenAI's function calling API. For example, converting 'Email Anya to"
            " see if she wants to get coffee next Friday' to a function call like"
            " `send_email(to: string, body: string)`."
        ),
    )
    model_name: str = Field(
        default="unknown",
        description=(
            "The model's name used for logging, testing, and sanity checking. For some"
            " models this can be automatically discerned. For other models, like"
            " locally loaded models, this must be manually specified."
        ),
    )
    system_role: MessageRole = Field(
        default=MessageRole.SYSTEM,
        description="The role this specific LLM provider"
        "expects for system prompt. E.g. 'SYSTEM' for OpenAI, 'CHATBOT' for Cohere",
    )
