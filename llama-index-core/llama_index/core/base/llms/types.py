from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod

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
from ffmpeg.asyncio import FFmpeg
from PIL import Image
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


class TextBlock(BaseContentBlock):
    """A representation of text data to directly pass to/from the LLM."""

    block_type: Literal["text"] = "text"
    text: str

    @classmethod
    async def amerge(
        cls, splits: List[TextBlock], chunk_size: int, tokenizer: Any | None = None
    ) -> list[TextBlock]:
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
    ) -> List[TextBlock]:
        from llama_index.core.node_parser import TokenTextSplitter

        text_splitter = TokenTextSplitter(
            chunk_size=max_tokens, chunk_overlap=overlap, tokenizer=tokenizer
        )
        chunks = text_splitter.split_text(self.text)
        return [TextBlock(text=chunk) for chunk in chunks]


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

    async def aestimate_tokens(self, *args: Any, **kwargs: Any) -> int:
        """Use PIL to read image size and conservatively estimate tokens."""
        try:
            with Image.open(cast(BytesIO, self.resolve_image())) as im:
                width, height = im.size

                # Calculates image tokens for OpenAI high res (maximum possible number of tokens)
                w_quotient, w_remainder = divmod(width, 512)
                h_quotient, h_remainder = divmod(height, 512)
                w_512factor = w_quotient + (1 if w_remainder > 0 else 0)
                h_512factor = h_quotient + (1 if h_remainder > 0 else 0)
                openai_max_count = 85 + w_512factor * h_512factor * 170

                # Calculates image tokens for Gemini (maximum possible number of tokens)
                w_quotient, w_remainder = divmod(width, 768)
                h_quotient, h_remainder = divmod(height, 768)
                w_768factor = w_quotient + (1 if w_remainder > 0 else 0)
                h_768factor = h_quotient + (1 if h_remainder > 0 else 0)
                gemini_max_count = w_768factor * h_768factor * 258
        except ValueError as e:
            if str(e) == "resolve_image returned zero bytes":
                return 0
            raise

        # We take the larger of the two estimates to be safe
        return max(openai_max_count, gemini_max_count)


class AudioVideoMixIn(ABC):
    """Mixin for audio and video blocks."""

    @property
    @abstractmethod
    def source(self) -> BytesIO:
        """Resolve the audio or video source as bytes."""
        ...

    @property
    @abstractmethod
    def extension(self) -> str | None:
        """The file extension"""
        ...

    @staticmethod
    def has_ffmpeg() -> bool:
        """Check if ffmpeg is installed."""
        return bool(shutil.which("ffmpeg"))

    async def ffprobe(self) -> dict:
        """
        Probe an audio source and return metadata from ffprobe
        """
        if not self.has_ffmpeg():
            _logger.warning(
                "ffmpeg is not installed or not found in PATH. Cannot probe audio/video data."
            )
            return {}

        data = self.source.read()

        # I found that piping in bytes does not work as well and sometimes results in critical metadata missing if
        # ffprobe can't determine format from pipe input. More reliable performance is achieved by writing to a temp file
        # with the proper extension
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / f"input.{self.extension or 'video'}"
                with open(tmp_path, "wb") as f:
                    f.write(data)

                ffmpeg = FFmpeg(executable="ffprobe").input(
                    str(tmp_path),
                    print_format="json",
                    show_streams=None,
                    show_format=None,
                )
                try:
                    return json.loads(await ffmpeg.execute())
                except Exception as e:
                    _logger.warning(
                        f"ffprobe output parsing error: {e}. Returning empty metadata."
                    )
                    return {}
        except Exception as e:
            _logger.warning(
                f"ffprobe failed with error: {e}. Returning empty metadata."
            )
            return {}

    async def can_concatenate(self, other: Self) -> bool:
        """
        Compare two audio/video files for concat-compatibility (no re-encoding).

        For audio files, video streams will be empty.

        Checks:
        - stream layout / order (video/audio/subtitle types must match)
        - per-video-stream keys
        - per-audio-stream keys
        """
        if not self.has_ffmpeg():
            _logger.warning(
                "ffmpeg is not installed or not found in PATH. Audio/Video data cannot be compared"
            )
            return False

        info_a = await self.ffprobe()
        info_b = await other.ffprobe()

        if not info_a or not info_b:
            # Ffprobe failed on one of the files
            return False

        streams_a = info_a.get("streams", []) or []
        streams_b = info_b.get("streams", []) or []

        # Ensure stream layout and order match (codec_type sequence)
        layout_a = [s.get("codec_type") for s in streams_a]
        layout_b = [s.get("codec_type") for s in streams_b]
        if layout_a != layout_b:
            return False

        video_keys = [
            "codec_name",
            "profile",
            "width",
            "height",
            "pix_fmt",
            "codec_tag_string",
        ]
        video_streams_a = [s for s in streams_a if s.get("codec_type") == "video"]
        video_streams_b = [s for s in streams_b if s.get("codec_type") == "video"]
        if len(video_streams_a) != len(video_streams_b):
            return False
        for sa, sb in zip(video_streams_a, video_streams_b):
            for k in video_keys:
                if sa.get(k) != sb.get(k):
                    return False

        # Audio stream checks (if any)
        audio_keys = [
            "codec_name",
            "sample_rate",
            "channels",
            "channel_layout",
            "sample_fmt",
            "codec_tag_string",
        ]
        audio_streams_a = [s for s in streams_a if s.get("codec_type") == "audio"]
        audio_streams_b = [s for s in streams_b if s.get("codec_type") == "audio"]
        if len(audio_streams_a) != len(audio_streams_b):
            return False
        for sa, sb in zip(audio_streams_a, audio_streams_b):
            for k in audio_keys:
                if sa.get(k) != sb.get(k):
                    return False
        return True

    async def ffmpeg_trim(
        self, data: bytes, seconds: float, reverse: bool = False
    ) -> bytes:
        """
        Trim `seconds` from start (reverse=False) or end (reverse=True) of `data`
        using ffmpeg. Returns trimmed bytes or raises on failure.
        """
        if not self.has_ffmpeg():
            _logger.warning(
                "ffmpeg is not installed or not found in PATH. Audio/Video data cannot be segmented"
            )
            return data

        default_extension = "audio" if isinstance(self, AudioBlock) else "video"
        extension = self.extension or default_extension

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / f"input.{extension}"
            out_path = Path(tmpdir) / f"out.{extension}"
            with open(in_path, "wb") as f:
                f.write(data)

            if reverse:
                # If trimming from end, use input-level sseof to seek from EOF
                ff = (
                    FFmpeg(executable="ffmpeg")
                    .input(str(in_path), sseof=f"-{seconds}")
                    .output(str(out_path), t=seconds, c="copy")
                )
            else:
                # trim from start: limit duration with -t
                ff = (
                    FFmpeg(executable="ffmpeg")
                    .input(str(in_path))
                    .output(str(out_path), t=seconds, c="copy")
                )
            try:
                await ff.execute()
                with open(out_path, "rb") as f:
                    return f.read()
            except Exception as e:
                _logger.warning(f"ffmpeg trim failed: {e}")
                return data

    async def ffmpeg_segment(
        self, data: bytes, segment_time: float
    ) -> AsyncGenerator[bytes, None]:
        """
        Segment audio/video using ffmpeg.

        This function is intentionally lightweight. It does not do any sort of forceful re-encoding or attempt to
        perfectly split on segment_time boundaries, but rather just cuts the input into segments of approximately
        segment_time length using ffmpeg's segment muxer with copy codec. For audio data, where it is cut primarily
        depends on packet boundaries, and for video data it depends on keyframe placement.
        This means that segments may be slightly
        longer than segment_time and may not be perfectly split if the input format does not support it,
        but it will be much faster, less resource-intensive and maintain data integrity.
        """
        if not self.has_ffmpeg():
            _logger.warning(
                "ffmpeg is not installed or not found in PATH. Audio/Video data cannot be segmented"
            )
            yield data
            return

        default_extension = "audio" if isinstance(self, AudioBlock) else "video"
        extension = self.extension or default_extension

        duration = (await self.ffprobe()).get("format", {}).get("duration", None)
        if duration:
            num_segments = int(float(duration) / segment_time) + 1
            name_padding = len(str(num_segments)) + 1
        else:
            name_padding = 9

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"input.{extension}"
            with open(tmp_path, "wb") as f:
                f.write(data)

            ffmpeg = (
                FFmpeg(executable="ffmpeg")
                .input(
                    str(tmp_path),
                )
                .output(
                    f"{tmp_path.parent}/output%0{name_padding}d.{extension}",
                    c="copy",
                    map="0",
                    f="segment",
                    segment_time=segment_time,
                    reset_timestamps=1,
                    avoid_negative_ts=1,
                    write_empty_segments=0,
                )
            )
            try:
                await ffmpeg.execute()

                segment_paths = [
                    (f, int(f.split(".")[0].replace("output", "")))
                    for f in os.listdir(tmpdir)
                    if f.startswith("output")
                ]
                segment_paths = [
                    segment for segment, _ in sorted(segment_paths, key=lambda x: x[1])
                ]
                for path in segment_paths:
                    with open(Path(tmpdir) / path, "rb") as f:
                        segment_data = f.read()
                    yield segment_data
            except Exception as e:
                _logger.warning(
                    f"ffmpeg segmentation failed with error: {e}. Returning original data."
                )
                yield data
                return

    @classmethod
    async def ffmpeg_concat(
        cls, data_list: List[bytes], extension: str | None
    ) -> list[bytes]:
        """
        Concatenate audio/video using ffmpeg

        This function is intentionally lightweight. It does not do any sort of forceful re-encoding or attempt to
        perfectly concatenate on segment_time boundaries, but rather just concatenates the inputs using ffmpeg's
        concat demuxer with copy codec. This means that concatenation may fail if the inputs are not perfectly
        compatible (e.g. different codecs or metadata), but it will be much faster, less resource-intensive and
        maintain data integrity when it works.
        """
        if not cls.has_ffmpeg():
            _logger.warning(
                "ffmpeg is not installed or not found in PATH. Audio/Video data cannot be concatenated"
            )
            return data_list

        default_extension = "audio" if cls.__name__ == "AudioBlock" else "video"
        ext = extension or default_extension

        with tempfile.TemporaryDirectory() as tmpdir:
            input_paths = []
            for i, data in enumerate(data_list):
                tmp_path = Path(tmpdir) / f"input_{i}.{ext}"
                with open(tmp_path, "wb") as f:
                    f.write(data)
                input_paths.append(str(tmp_path))
            input_file = Path(tmpdir) / "input.txt"
            with open(input_file, "w") as f:
                for path in input_paths:
                    f.write(f"file '{path}'\n")
            output_path = Path(tmpdir) / f"concatenated_output.{ext}"
            ffmpeg = (
                FFmpeg(executable="ffmpeg")
                .input(str(input_file), f="concat", safe=0)
                .output(str(output_path), c="copy")
            )
            try:
                await ffmpeg.execute()
                with open(output_path, "rb") as f:
                    concatenated_data = f.read()
                return [concatenated_data]
            except Exception as e:
                _logger.warning(
                    f"ffmpeg concatenation failed with error: {e}. Returning original splits."
                )
                return data_list


class AudioBlock(AudioVideoMixIn, BaseContentBlock):
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

        In case the model was built passing audio data but without a mimetype,
        we try to guess it using the filetype library. To avoid resource-intense
        operations, we won't load the path or the URL to guess the mimetype.
        """
        if not self.audio or not isinstance(self.audio, bytes):
            return self

        self._guess_format(resolve_binary(self.audio).read())
        self.audio = resolve_binary(self.audio, as_base64=True).read()
        return self

    @property
    def source(self) -> BytesIO:
        return cast(BytesIO, self.resolve_audio())

    @property
    def extension(self) -> str | None:
        return self.format

    @classmethod
    async def amerge(
        cls, splits: List[Self], chunk_size: int, *args, **kwargs
    ) -> list[Self]:
        """
        Use ffmpeg to merge audio files
        """
        if not cls.has_ffmpeg():
            _logger.warning(
                "ffmpeg is not installed or not found in PATH. Audio data cannot be merged"
            )
            return splits

        merged_blocks = []
        cur_splits = []
        cur_tokens = 0

        for split in splits:
            split_tokens = await split.aestimate_tokens()
            can_concat = len(cur_splits) == 0 or await cur_splits[-1].can_concatenate(
                split
            )
            if cur_tokens + split_tokens <= chunk_size and can_concat:
                cur_splits.append(split)
                cur_tokens += split_tokens
            else:
                if len(cur_splits) == 1:
                    # Just add the single split
                    merged_blocks.append(cur_splits[0])
                    cur_splits = [split]
                    cur_tokens = split_tokens
                elif len(cur_splits) > 0:
                    # Merge current audios
                    data_splits = [split.source.read() for split in cur_splits]
                    merged_blocks.extend(
                        AudioBlock(audio=split, format=splits[0].format)
                        for split in await cls.ffmpeg_concat(
                            data_splits, splits[0].extension
                        )
                    )
                    cur_splits = [split]
                    cur_tokens = split_tokens

        if cur_splits:
            if len(cur_splits) == 1:
                # Just add the single split
                merged_blocks.append(cur_splits[0])
            else:
                # Merge current audios
                data_splits = [split.source.read() for split in cur_splits]
                merged_blocks.extend(
                    AudioBlock(audio=split, format=splits[0].format)
                    for split in await cls.ffmpeg_concat(data_splits, splits[0].format)
                )
        return merged_blocks

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
            # Next try ffprobe
            if self.has_ffmpeg():
                metadata = await self.ffprobe()
                if duration := metadata.get("format", {}).get("duration", None):
                    # We conservatively return the max estimate
                    # First convert string to float
                    duration = float(duration)
                    return max((int(duration) + 1) * 32, int(duration / 0.05) + 1)
                return 256  # fallback
            return 256  # fallback
        except ValueError as e:
            if str(e) == "resolve_audio returned zero bytes":
                return 0
            raise

    async def asplit(self, max_tokens: int, *args, **kwargs) -> List[Self]:
        """
        Split audio into chunks using ffmpeg
        """
        if not self.has_ffmpeg():
            _logger.warning(
                "ffmpeg is not installed or not found in PATH. Audio data cannot be split"
            )
            return [self]

        source = self.resolve_audio()
        data = source.read()
        segment_time = max(
            float(max_tokens // 32.0), 1.0
        )  # default 32 tokens per second, min 1 second segments

        return [
            AudioBlock(audio=audio_segment, format=self.format)
            async for audio_segment in self.ffmpeg_segment(data, segment_time)
        ]

    async def atruncate(
        self, max_tokens: int, tokenizer: Any | None = None, reverse=False
    ) -> Self:
        """Async truncate the audio block to up to max_tokens tokens."""
        estimated_tokens = await self.aestimate_tokens(tokenizer=tokenizer)
        if estimated_tokens <= max_tokens:
            return self

        source = self.resolve_audio()
        data = source.read()
        # default 32 tokens per second
        # Minimum segment time is 1 second to avoid issues with very small trims
        segment_time = max(float(max_tokens // 32.0), 1.0)

        trimmed_data = await self.ffmpeg_trim(
            data, seconds=segment_time, reverse=reverse
        )
        return AudioBlock(audio=trimmed_data, format=self.format)


class VideoBlock(AudioVideoMixIn, BaseContentBlock):
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
                    if mimetype and str(mimetype.mime).startswith("video/"):
                        self.video_mimetype = str(mimetype.mime)
            return self

        self._guess_mimetype(resolve_binary(self.video).read())
        self.video = resolve_binary(self.video, as_base64=True).read()
        return self

    @property
    def source(self) -> BytesIO:
        return cast(BytesIO, self.resolve_video())

    @property
    def extension(self) -> str | None:
        if self.video_mimetype:
            kind = filetype.get_type(mime=self.video_mimetype)
            return kind.extension if kind else None
        return None

    @classmethod
    async def amerge(
        cls, splits: List[Self], chunk_size: int, *args, **kwargs
    ) -> list[Self]:
        """
        Use ffmpeg to merge audio files
        """
        if not cls.has_ffmpeg():
            _logger.warning(
                "ffmpeg is not installed or not found in PATH. Audio data cannot be merged"
            )
            return splits

        merged_blocks = []
        cur_splits = []
        cur_tokens = 0

        for split in splits:
            split_tokens = await split.aestimate_tokens()
            can_concat = len(cur_splits) == 0 or await cur_splits[-1].can_concatenate(
                split
            )
            if cur_tokens + split_tokens <= chunk_size and can_concat:
                cur_splits.append(split)
                cur_tokens += split_tokens
            else:
                if len(cur_splits) == 1:
                    # Just add the single split
                    merged_blocks.append(cur_splits[0])
                    cur_splits = [split]
                    cur_tokens = split_tokens
                elif len(cur_splits) > 0:
                    # Merge current videos
                    data_splits = [split.source.read() for split in cur_splits]
                    merged_blocks.extend(
                        VideoBlock(video=split, video_mimetype=splits[0].video_mimetype)
                        for split in await cls.ffmpeg_concat(
                            data_splits, splits[0].extension
                        )
                    )
                    cur_splits = [split.resolve_audio().read()]
                    cur_tokens = split_tokens

        if cur_splits:
            if len(cur_splits) == 1:
                # Just add the single split
                merged_blocks.append(cur_splits[0])
            else:
                # Merge current videos
                data_splits = [split.source.read() for split in cur_splits]
                merged_blocks.extend(
                    VideoBlock(video=split, video_mimetype=splits[0].video_mimetype)
                    for split in await cls.ffmpeg_concat(
                        data_splits, splits[0].extension
                    )
                )
        return merged_blocks

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

            # Next try ffprobe
            if self.has_ffmpeg():
                metadata = await self.ffprobe()
                if duration := metadata.get("format", {}).get("duration", None):
                    # First convert string to float
                    duration = float(duration)
                    return (int(float(duration)) + 1) * 263
            # fallback of roughly 8 times the fallback cost of audio (263 // 32; based on gemini pricing per sec)
            return 256 * 8
        except ValueError as e:
            if str(e) == "resolve_video returned zero bytes":
                return 0
            raise

    async def asplit(self, max_tokens: int, *args, **kwargs) -> List[Self]:
        """
        Split video into chunks using ffmpeg
        """
        if not self.has_ffmpeg():
            _logger.warning(
                "ffmpeg is not installed or not found in PATH. Video data cannot be split"
            )
            return [self]

        source = self.resolve_video()
        data = source.read()
        # default 263 tokens per second, min 1 second segments
        segment_time = max(float(max_tokens // 263), 1.0)

        return [
            VideoBlock(video=video_segment, video_mimetype=self.video_mimetype)
            async for video_segment in self.ffmpeg_segment(data, segment_time)
        ]

    async def atruncate(
        self, max_tokens: int, tokenizer: Any | None = None, reverse=False
    ) -> Self:
        """Async truncate the video block to up to max_tokens tokens."""
        estimated_tokens = await self.aestimate_tokens(tokenizer=tokenizer)
        if estimated_tokens <= max_tokens:
            return self

        source = self.resolve_video()
        data = source.read()
        # default 263 tokens per second
        # Minimum segment time is 1 second to avoid issues with very small trims
        segment_time = max(float(max_tokens // 263), 1.0)

        trimmed_data = await self.ffmpeg_trim(
            data, seconds=segment_time, reverse=reverse
        )
        return VideoBlock(video=trimmed_data, video_mimetype=self.video_mimetype)


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


class CacheControl(BaseContentBlock):
    type: str
    ttl: str = Field(default="5m")


class CachePoint(BaseContentBlock):
    """Used to set the point to cache up to, if the LLM supports caching."""

    block_type: Literal["cache"] = "cache"
    cache_control: CacheControl


class CitableBlock(BaseContentBlock):
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


class CitationBlock(BaseContentBlock):
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
        return v


class ThinkingBlock(BaseContentBlock):
    """
    A representation of the content streamed from reasoning/thinking processes by LLMs

    Because of LLM provider's reliance on signatures for Thought Processes,
    we do not support merging/splitting/truncating for this block, as we want to preserve the integrity of the content
    provided by the LLM.
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


class ChatMessage(BaseModel):
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
