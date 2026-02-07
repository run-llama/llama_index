import base64
from io import BytesIO
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, Mock

import pytest
import httpx
from ffmpeg import FFmpegError
from ffmpeg.asyncio import FFmpeg
from tinytag import UnsupportedFormatError, TinyTag

from llama_index.core import get_tokenizer
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ImageBlock,
    MessageRole,
    TextBlock,
    DocumentBlock,
    VideoBlock,
    AudioBlock,
    CachePoint,
    CacheControl,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.bridge.pydantic import ValidationError
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import ImageDocument
from pydantic import AnyUrl


@pytest.fixture()
def empty_bytes() -> bytes:
    return b""


@pytest.fixture()
def png_1px_b64() -> bytes:
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def png_1px(png_1px_b64) -> bytes:
    return base64.b64decode(png_1px_b64)


@pytest.fixture()
def pdf_url() -> str:
    return "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


@pytest.fixture()
def mock_pdf_bytes(pdf_url) -> bytes:
    """
    Returns a byte string representing a very simple, minimal PDF file.
    """
    return httpx.get(pdf_url).content


@pytest.fixture()
def pdf_base64(mock_pdf_bytes) -> bytes:
    return base64.b64encode(mock_pdf_bytes)


@pytest.fixture()
def mp3_bytes(mock_ffprobe_mp3_bytes_output) -> bytes:
    """
    Small mp3 file bytes (0.2 seconds of audio).
    Actually works with ffmpeg to be split into two 0.1 second chunks using
    ffmpeg -i file_path.mp3 -c copy -map 0 -f segment -segment_time 0.1 output%03d.mp3
    """
    return b"ID3\x04\x00\x00\x00\x00\x01\tTXXX\x00\x00\x00\x12\x00\x00\x03major_brand\x00isom\x00TXXX\x00\x00\x00\x13\x00\x00\x03minor_version\x00512\x00TXXX\x00\x00\x00$\x00\x00\x03compatible_brands\x00isomiso2avc1mp41\x00TSSE\x00\x00\x00\x0e\x00\x00\x03Lavf62.3.100\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf3X\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00Info\x00\x00\x00\x0f\x00\x00\x00\x06\x00\x00\x03<\x00YYYYYYYYYYYYYYYYzzzzzzzzzzzzzzzzz\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00Lavf\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$\x00\x00\x00\x00\x00\x00\x00\x00\x03<\xa6\xbc`\x8e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf38\xc4\x00\x00\x00\x03H\x00\x00\x00\x00LAME3.100UUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4_\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"


@pytest.fixture()
def mp3_split1_bytes() -> bytes:
    """
    First half of the small mp3 file bytes (0.1 seconds of audio).
    Split using
    ffmpeg -i file_path.mp3 -c copy -map 0 -f segment -segment_time 0.1 output%03d.mp3
    """
    return b"ID3\x04\x00\x00\x00\x00\x01\tTXXX\x00\x00\x00\x12\x00\x00\x03major_brand\x00isom\x00TXXX\x00\x00\x00\x13\x00\x00\x03minor_version\x00512\x00TXXX\x00\x00\x00$\x00\x00\x03compatible_brands\x00isomiso2avc1mp41\x00TSSE\x00\x00\x00\x0e\x00\x00\x03Lavf62.3.100\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf3X\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00Info\x00\x00\x00\x0f\x00\x00\x00\x03\x00\x00\x01\xf8\x00\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00Lavf\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$\x00\x00\x00\x00\x00\x00\x00\x00\x01\xf8\xbfu\\\xdf\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf38\xc4\x00\x00\x00\x03H\x00\x00\x00\x00LAME3.100UUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4_\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"


@pytest.fixture()
def mp3_split2_bytes() -> bytes:
    """
    Second half of the small mp3 file bytes (0.1 seconds of audio).
    Split using
    ffmpeg -i <file_path>.mp3 -c copy -map 0 -f segment -segment_time 0.1 output%03d.mp3
    """
    return b"ID3\x04\x00\x00\x00\x00\x01\tTXXX\x00\x00\x00\x12\x00\x00\x03major_brand\x00isom\x00TXXX\x00\x00\x00\x13\x00\x00\x03minor_version\x00512\x00TXXX\x00\x00\x00$\x00\x00\x03compatible_brands\x00isomiso2avc1mp41\x00TSSE\x00\x00\x00\x0e\x00\x00\x03Lavf62.3.100\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf3X\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00Info\x00\x00\x00\x0f\x00\x00\x00\x03\x00\x00\x01\xf8\x00\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00Lavf\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\xf8}z\x1e\xf9\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"


@pytest.fixture()
def mp3_split2_sr_8000_bytes() -> bytes:
    """
    Second half of the small mp3 file bytes (0.1 seconds of audio) with sample rate 8000.
    Split using
    ffmpeg -i <file_path>.mp3 -c copy -map 0 -f segment -segment_time 0.1 output%03d.mp3

    resampled using
    ffmpeg -y -hide_banner -loglevel error -i <file_path>.mp3 -ar 8000 -ac 1 -c:a libmp3lame -b:a 64k output001_8000.mp3
    """
    return b"ID3\x04\x00\x00\x00\x00\x01\tTXXX\x00\x00\x00\x12\x00\x00\x03major_brand\x00isom\x00TXXX\x00\x00\x00\x13\x00\x00\x03minor_version\x00512\x00TXXX\x00\x00\x00$\x00\x00\x03compatible_brands\x00isomiso2avc1mp41\x00TSSE\x00\x00\x00\x0e\x00\x00\x03Lavf62.3.100\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xe38\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00Info\x00\x00\x00\x0f\x00\x00\x00\x04\x00\x00\x01\xf8\x00\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00Lavc62.11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$\x04Q\x00\x00\x00\x00\x00\x00\x01\xf8A\x1a\x80\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xe3\x18\xc4\x00\x00\x00\x03H\x00\x00\x00\x00LAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xe3\x18\xc4;\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xe3\x18\xc4v\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xe3\x18\xc4\xb1\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"


@pytest.fixture()
def mp3_base64(mp3_bytes: bytes) -> bytes:
    return base64.b64encode(mp3_bytes)


@pytest.fixture()
def mp3_split1_base64(mp3_split1_bytes: bytes) -> bytes:
    return base64.b64encode(mp3_split1_bytes)


@pytest.fixture()
def mp3_split2_base64(mp3_split2_bytes: bytes) -> bytes:
    return base64.b64encode(mp3_split2_bytes)


@pytest.fixture()
def mp3_split2_sr_8000_base64(mp3_split2_sr_8000_bytes: bytes) -> bytes:
    return base64.b64encode(mp3_split2_sr_8000_bytes)


@pytest.fixture()
def mock_ffprobe_mp3_bytes_output():
    """
    Actual ffprobe output of mp3_bytes.
    """
    return """{
        "streams": [
            {
                "index": 0,
                "codec_name": "mp3",
                "codec_long_name": "MP3 (MPEG audio layer 3)",
                "codec_type": "audio",
                "codec_tag_string": "[0][0][0][0]",
                "codec_tag": "0x0000",
                "sample_fmt": "fltp",
                "sample_rate": "16000",
                "channels": 1,
                "channel_layout": "mono",
                "bits_per_sample": 0,
                "initial_padding": 0,
                "r_frame_rate": "0/0",
                "avg_frame_rate": "0/0",
                "time_base": "1/14112000",
                "start_pts": 974610,
                "start_time": "0.069062",
                "duration_ts": 2540160,
                "duration": "0.180000",
                "bit_rate": "24000",
                "disposition": {
                    "default": 0,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                }
            }
        ],
        "format": {
            "filename": "/var/folders/43/2mzmbvc14xxgrggdtw27nhww0000gp/T/tmpa9y2yz0c/input.mp3",
            "nb_streams": 1,
            "nb_programs": 0,
            "nb_stream_groups": 0,
            "format_name": "mp3",
            "format_long_name": "MP2/3 (MPEG audio layer 2/3)",
            "start_time": "0.069063",
            "duration": "0.180000",
            "size": "975",
            "bit_rate": "43333",
            "probe_score": 51,
            "tags": {
                "major_brand": "isom",
                "minor_version": "512",
                "compatible_brands": "isomiso2avc1mp41",
                "encoder": "Lavf62.3.100"
            }
        }
    }"""


@pytest.fixture()
def mock_ffprobe_mp3_split1_bytes_output():
    """
    Actual ffprobe output of the first split mp3 file.
    """
    return """{
        "streams": [
            {
                "index": 0,
                "codec_name": "mp3",
                "codec_long_name": "MP3 (MPEG audio layer 3)",
                "codec_type": "audio",
                "codec_tag_string": "[0][0][0][0]",
                "codec_tag": "0x0000",
                "sample_fmt": "fltp",
                "sample_rate": "16000",
                "channels": 1,
                "channel_layout": "mono",
                "bits_per_sample": 0,
                "initial_padding": 0,
                "r_frame_rate": "0/0",
                "avg_frame_rate": "0/0",
                "time_base": "1/14112000",
                "start_pts": 974610,
                "start_time": "0.069062",
                "duration_ts": 1016064,
                "duration": "0.072000",
                "bit_rate": "24000",
                "disposition": {
                    "default": 0,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                }
            }
        ],
        "format": {
            "filename": "/var/folders/43/2mzmbvc14xxgrggdtw27nhww0000gp/T/tmpu5oybbgp/input.mp3",
            "nb_streams": 1,
            "nb_programs": 0,
            "nb_stream_groups": 0,
            "format_name": "mp3",
            "format_long_name": "MP2/3 (MPEG audio layer 2/3)",
            "start_time": "0.069063",
            "duration": "0.072000",
            "size": "651",
            "bit_rate": "72333",
            "probe_score": 25,
            "tags": {
                "major_brand": "isom",
                "minor_version": "512",
                "compatible_brands": "isomiso2avc1mp41",
                "encoder": "Lavf62.3.100"
            }
        }
    }"""


@pytest.fixture()
def mock_ffprobe_mp3_split2_bytes_output():
    """
    Actual ffprobe output of the second split mp3 file.
    """
    return """{
        "streams": [
            {
                "index": 0,
                "codec_name": "mp3",
                "codec_long_name": "MP3 (MPEG audio layer 3)",
                "codec_type": "audio",
                "codec_tag_string": "[0][0][0][0]",
                "codec_tag": "0x0000",
                "sample_fmt": "fltp",
                "sample_rate": "16000",
                "channels": 1,
                "channel_layout": "mono",
                "bits_per_sample": 0,
                "initial_padding": 0,
                "r_frame_rate": "0/0",
                "avg_frame_rate": "0/0",
                "time_base": "1/14112000",
                "start_pts": 466578,
                "start_time": "0.033063",
                "duration_ts": 1524096,
                "duration": "0.108000",
                "bit_rate": "24000",
                "disposition": {
                    "default": 0,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                }
            }
        ],
        "format": {
            "filename": "/var/folders/43/2mzmbvc14xxgrggdtw27nhww0000gp/T/tmpwbc3d42f/input.mp3",
            "nb_streams": 1,
            "nb_programs": 0,
            "nb_stream_groups": 0,
            "format_name": "mp3",
            "format_long_name": "MP2/3 (MPEG audio layer 2/3)",
            "start_time": "0.033063",
            "duration": "0.108000",
            "size": "651",
            "bit_rate": "48222",
            "probe_score": 25,
            "tags": {
                "major_brand": "isom",
                "minor_version": "512",
                "compatible_brands": "isomiso2avc1mp41",
                "encoder": "Lavf62.3.100"
            }
        }
    }"""


@pytest.fixture()
def mock_ffprobe_mp3_split2_sr8000_bytes_output():
    """
    Actual ffprobe output of the second split mp3 file with sample rate 8000.
    """
    return """{
        "streams": [
            {
                "index": 0,
                "codec_name": "mp3",
                "codec_long_name": "MP3 (MPEG audio layer 3)",
                "codec_type": "audio",
                "codec_tag_string": "[0][0][0][0]",
                "codec_tag": "0x0000",
                "sample_fmt": "fltp",
                "sample_rate": "8000",
                "channels": 1,
                "channel_layout": "mono",
                "bits_per_sample": 0,
                "initial_padding": 0,
                "r_frame_rate": "0/0",
                "avg_frame_rate": "0/0",
                "time_base": "1/14112000",
                "start_pts": 1949220,
                "start_time": "0.138125",
                "duration_ts": 1098972,
                "duration": "0.077875",
                "bit_rate": "8000",
                "disposition": {
                    "default": 0,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                }
            }
        ],
        "format": {
            "filename": "/var/folders/43/2mzmbvc14xxgrggdtw27nhww0000gp/T/tmp1d00gpc5/input.mp3",
            "nb_streams": 1,
            "nb_programs": 0,
            "nb_stream_groups": 0,
            "format_name": "mp3",
            "format_long_name": "MP2/3 (MPEG audio layer 2/3)",
            "start_time": "0.138125",
            "duration": "0.077875",
            "size": "651",
            "bit_rate": "66876",
            "probe_score": 25,
            "tags": {
                "major_brand": "isom",
                "minor_version": "512",
                "compatible_brands": "isomiso2avc1mp41",
                "encoder": "Lavf62.3.100"
            }
        }
    }"""


@pytest.fixture()
def mp4_bytes() -> bytes:
    """
    Small video file bytes (~3 seconds/2kb that behaves well with ffmpeg)
    """
    return b"""\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41\x00\x00\x00\x08free\x00\x00\x03Gmdat\x00\x00\x00\x18gd\x00\n\xac\xd9B\x8d\xf9!\x00\x00\x03\x00\x01\x00\x00\x03\x00\x04\x0f\x12%\x96\x00\x00\x00\x05h\xef\x82\\\xb0\x00\x00\x01de\x88\x82\x00\x05?\xbc@k\xad-\xccR<o\xadJ\xf7\xd5\xabId\xe6\x98\x86\xde\xaf\xa9\xe7q\xa2\xb3\xf0\x04*5\x84_\xb4\xbc\xb9\xd9;\t\x1d\x8c\xd9\xb5\xd8\xf7\xd0g\xa1\xb5\x166Dh4\xbcfE\xefH\x10:\xf2]E_\xb4\xe4\xec\x99\x8a\x17Sm\xef%\x94\xd8a6\x9aX\xf9\xfe\x81/kv\xa4\x8fo\xc5:\xa4\xc1\xccrS\xf1!F\xf4\x80\xf1~$<\x8fdq\xb2}1h\xa5\xc09(\x18>\xad\xa1\x06$z8R\x1a \xde\xbd\xeaT\x9a*\xb1\x196\xf3qD\xf0\x81\xe3GP\xb5-\x92(\xecXZ\xab\xee=\x9c\x89\x8fY\xa3\x9a\x0f\xda\x8d\xab\x04\x80\xed8\x13v\x1a~\xbc\x00\x97\x84\xdea\x03\xeb\xf9\xc5\x8f\xf1p7A\xcd|\xf7s\xaa\xfc\xe1\xd2\xb1\x1c\xe0~\x9aC/c\xe1\xf6\x93\xd9\x883P\x1b\xd7\xf0\xa4\xc6\x00\xb5E\xa1\x955\xa6$\xcac\xb6`;\xe9\x83\xf8\xa8\x0bKi\x1bg\x8d\x90\xdesxeK\xfcw\xaf\xee\xc4s\x9f\x8a\xdb\xfdw\xd8\xc2\x90E\xa3g\xde\xd1\xf4G\xdc\xdeJ\xdf\x07\x12\xado\xfd\x8fJ\x1b9\xf3\xa5\xde\x1e\x02Q\xed\x8d\x91h\xebn\xe7\xa9\xc6\xfcJN!8\x08r\xd1\x88\xf2R\x92\x0c\xc7\'\x1f\xd4\x93f\x9a1s\x08\xa0`9A|\xdcF\xabg\xf5\xf7\xf1\'\xc7\xac\xa9\xf7\xfc\x0fW\x90\x02\x80\xe0\xcf\xe7\x11\x96$Cy\x10\xfc\x90\xde\x02\x00Lavc62.11.100\x00\x020@\x0e\x00\x00\x00\x18gd\x00\n\xac\xd9B\x8d\xf9!\x00\x00\x03\x00\x01\x00\x00\x03\x00\x04\x0f\x12%\x96\x00\x00\x00\x05h\xef\x82\\\xb0\x00\x00\x01de\x88\x84\x00\x14\xff\xbc@k\xad-\xccR<o\xadJ\xf7\xd5\xabId\xe6\x98\x86\xde\xaf\xa9\xe7p\xcf\xf7\xe4\x02\x84\xcfZ\x8d\x95S\xedR\x81\xecy_\n\xc4PT\xb2ry\xac\x17=c\xf7\x88\x1b\xbe u\xe4\xba\x8a\xc1\xb9\xc5\x94>\xb3*[\x00\xc1\xd5SAl\xf21wh<\x04\x82\xa4\x1f9\xd2n^\xae\xa8\xd7O\xb2\x19J\x99\xd1\x18\rN\x84\xab\xcd\x8c\xbc\xfe\xfc\xa7\xa8u}[\x8bWi\rf \xf9\xae\xb1#\x13\xf1\xacyY\x19\xf5:xG\xc72\xbas\xf2\x97Eg\x0e\xa0\xaa\x1a\x0c\x94\x9e\t\xf9(\xb5\xe5\x9d\x8b\x01#\xac\xe2\x91|\xc6\xfd\xa5D\xdd\x9a\xf5\xb6\xd5L\x12\x03\xfa\xce\x90\x02\xe6\xa9\x18\xc9\xdc}\x88n\xfb\xfb*\x83&KNzJ\x11b\xbbm\xa0\xedS\x8c\x03(\'\x16\x9fc:\xff\xf8"\xa2b\xce]\xfc\xb2\xbe\x9d\xdc\xd4\x94\xa5Y \xe8\x8e \xa5\x87\x98\x07\x12-\xday\xdf}*\xdfh\x96\'\xc6\xbc\x82\x99\xfc\xf2~F\xf1\xef\xfb\x8b\xb6\x96\xc0{\xe9]\x84h\xd7\xbct\x1c\x9a\xaf\x01\xddsP\x84\'\xf6$G\xef\x12\x1e4\x8f1\xea\x0f\xb2\xc0\xed[\xa4\x00\xa9\x83:\xad\xe9\x86\xd1\xbf\xa0\xb6\xacp\xe5\x06\xc3\xf0v\x94\x04\x9e\xdffBWcO\xe9\xa9/M\xec\x8c\x14\x00\xa1\xa0\xc2\xa4bY\xd0\x030b\x02\x00\x1b}\xc3\xea\xa1s\x8b\xb3\x889\x82\xc5A\x01\x18 \x07\x01\x18 \x07\x01\x18 \x07\x01\x18 \x07\x00\x00\x05\x91moov\x00\x00\x00lmvhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\nD\x00\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x02_trak\x00\x00\x00\\tkhd\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\nD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\xa0\x00\x00\x00Z\x00\x00\x00\x00\x000edts\x00\x00\x00(elst\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x02s\xff\xff\xff\xff\x00\x01\x00\x00\x00\x00\x07\xd0\x00\x00@\x00\x00\x01\x00\x00\x00\x00\x01\xcbmdia\x00\x00\x00 mdhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x80\x00\x15\xc7\x00\x00\x00\x00\x00-hdlr\x00\x00\x00\x00\x00\x00\x00\x00vide\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00VideoHandler\x00\x00\x00\x01vminf\x00\x00\x00\x14vmhd\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$dinf\x00\x00\x00\x1cdref\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0curl \x00\x00\x00\x01\x00\x00\x016stbl\x00\x00\x00\xaestsd\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x9eavc1\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0\x00Z\x00H\x00\x00\x00H\x00\x00\x00\x00\x00\x00\x00\x01\x15Lavc62.11.100 libx264\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x18\xff\xff\x00\x00\x004avcC\x01d\x00\n\xff\xe1\x00\x18gd\x00\n\xac\xd9B\x8d\xf9!\x00\x00\x03\x00\x01\x00\x00\x03\x00\x04\x0f\x12%\x96\x01\x00\x05h\xef\x82\\\xb0\xfd\xf8\xf8\x00\x00\x00\x00\x14btrt\x00\x00\x00\x00\x00\x00\x16\x80\x00\x00\x0ch\x00\x00\x00 stts\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00`\x00\x00\x00\x00\x01\x00\x00 \x00\x00\x00\x00\x18ctts\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00@\x00\x00\x00\x00\x1cstsc\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x14stsz\x00\x00\x00\x00\x00\x00\x01\x8d\x00\x00\x00\x02\x00\x00\x00\x18stco\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x000\x00\x00\x01\xd2\x00\x00\x02]trak\x00\x00\x00\\tkhd\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x06h\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$edts\x00\x00\x00\x1celst\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x06h\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x01\xd5mdia\x00\x00\x00 mdhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f@\x00\x003@\x15\xc7\x00\x00\x00\x00\x00-hdlr\x00\x00\x00\x00\x00\x00\x00\x00soun\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00SoundHandler\x00\x00\x00\x01\x80minf\x00\x00\x00\x10smhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$dinf\x00\x00\x00\x1cdref\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0curl \x00\x00\x00\x01\x00\x00\x01Dstbl\x00\x00\x00~stsd\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00nmp4a\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x10\x00\x00\x00\x00\x1f@\x00\x00\x00\x00\x006esds\x00\x00\x00\x00\x03\x80\x80\x80%\x00\x02\x00\x04\x80\x80\x80\x17@\x15\x00\x00\x00\x00\x00\x05 \x00\x00\x00\xb4\x05\x80\x80\x80\x05\x15\x88V\xe5\x00\x06\x80\x80\x80\x01\x02\x00\x00\x00\x14btrt\x00\x00\x00\x00\x00\x00\x05 \x00\x00\x00\xb4\x00\x00\x00 stts\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00#@\x00\x00\x00\x04\x00\x00\x04\x00\x00\x00\x00(stsc\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00(stsz\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x15\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x18stco\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x01\xbd\x00\x00\x03_\x00\x00\x00\x1asgpd\x01\x00\x00\x00roll\x00\x00\x00\x02\x00\x00\x00\x01\xff\xff\x00\x00\x00\x1csbgp\x00\x00\x00\x00roll\x00\x00\x00\x01\x00\x00\x00\x05\x00\x00\x00\x01\x00\x00\x00audta\x00\x00\x00Ymeta\x00\x00\x00\x00\x00\x00\x00!hdlr\x00\x00\x00\x00\x00\x00\x00\x00mdirappl\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00,ilst\x00\x00\x00$\xa9too\x00\x00\x00\x1cdata\x00\x00\x00\x01\x00\x00\x00\x00Lavf62.3.100"""


@pytest.fixture()
def mp4_split1_bytes() -> bytes:
    """
    First segment of the small video file bytes.

    Split using
    ffmpeg -i <file_path>.mp4 -c copy -map 0 -f segment -segment_time 1.0 output%03d.mp4
    """
    return b"""\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41\x00\x00\x00\x08free\x00\x00\x01\xaamdat\x00\x00\x00\x18gd\x00\n\xac\xd9B\x8d\xf9!\x00\x00\x03\x00\x01\x00\x00\x03\x00\x04\x0f\x12%\x96\x00\x00\x00\x05h\xef\x82\\\xb0\x00\x00\x01de\x88\x82\x00\x05?\xbc@k\xad-\xccR<o\xadJ\xf7\xd5\xabId\xe6\x98\x86\xde\xaf\xa9\xe7q\xa2\xb3\xf0\x04*5\x84_\xb4\xbc\xb9\xd9;\t\x1d\x8c\xd9\xb5\xd8\xf7\xd0g\xa1\xb5\x166Dh4\xbcfE\xefH\x10:\xf2]E_\xb4\xe4\xec\x99\x8a\x17Sm\xef%\x94\xd8a6\x9aX\xf9\xfe\x81/kv\xa4\x8fo\xc5:\xa4\xc1\xccrS\xf1!F\xf4\x80\xf1~$<\x8fdq\xb2}1h\xa5\xc09(\x18>\xad\xa1\x06$z8R\x1a \xde\xbd\xeaT\x9a*\xb1\x196\xf3qD\xf0\x81\xe3GP\xb5-\x92(\xecXZ\xab\xee=\x9c\x89\x8fY\xa3\x9a\x0f\xda\x8d\xab\x04\x80\xed8\x13v\x1a~\xbc\x00\x97\x84\xdea\x03\xeb\xf9\xc5\x8f\xf1p7A\xcd|\xf7s\xaa\xfc\xe1\xd2\xb1\x1c\xe0~\x9aC/c\xe1\xf6\x93\xd9\x883P\x1b\xd7\xf0\xa4\xc6\x00\xb5E\xa1\x955\xa6$\xcac\xb6`;\xe9\x83\xf8\xa8\x0bKi\x1bg\x8d\x90\xdesxeK\xfcw\xaf\xee\xc4s\x9f\x8a\xdb\xfdw\xd8\xc2\x90E\xa3g\xde\xd1\xf4G\xdc\xdeJ\xdf\x07\x12\xado\xfd\x8fJ\x1b9\xf3\xa5\xde\x1e\x02Q\xed\x8d\x91h\xebn\xe7\xa9\xc6\xfcJN!8\x08r\xd1\x88\xf2R\x92\x0c\xc7'\x1f\xd4\x93f\x9a1s\x08\xa0`9A|\xdcF\xabg\xf5\xf7\xf1'\xc7\xac\xa9\xf7\xfc\x0fW\x90\x02\x80\xe0\xcf\xe7\x11\x96$Cy\x10\xfc\x90\xde\x02\x00Lavc62.11.100\x00\x020@\x0e\x00\x00\x05emoov\x00\x00\x00lmvhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\t\xc4\x00\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x02Strak\x00\x00\x00\\tkhd\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\t\xc4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\xa0\x00\x00\x00Z\x00\x00\x00\x00\x000edts\x00\x00\x00(elst\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x03\xe8\xff\xff\xff\xff\x00\x01\x00\x00\x00\x00\x05\xdc\x00\x00@\x00\x00\x01\x00\x00\x00\x00\x01\xbfmdia\x00\x00\x00 mdhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00`\x00\x15\xc7\x00\x00\x00\x00\x00-hdlr\x00\x00\x00\x00\x00\x00\x00\x00vide\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00VideoHandler\x00\x00\x00\x01jminf\x00\x00\x00\x14vmhd\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$dinf\x00\x00\x00\x1cdref\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0curl \x00\x00\x00\x01\x00\x00\x01*stbl\x00\x00\x00\xaestsd\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x9eavc1\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0\x00Z\x00H\x00\x00\x00H\x00\x00\x00\x00\x00\x00\x00\x01\x15Lavc62.11.100 libx264\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x18\xff\xff\x00\x00\x004avcC\x01d\x00\n\xff\xe1\x00\x18gd\x00\n\xac\xd9B\x8d\xf9!\x00\x00\x03\x00\x01\x00\x00\x03\x00\x04\x0f\x12%\x96\x01\x00\x05h\xef\x82\\\xb0\xfd\xf8\xf8\x00\x00\x00\x00\x14btrt\x00\x00\x00\x00\x00\x00\x0ch\x00\x00\x08E\x00\x00\x00\x18stts\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00`\x00\x00\x00\x00\x18ctts\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00@\x00\x00\x00\x00\x1cstsc\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x14stsz\x00\x00\x00\x00\x00\x00\x01\x8d\x00\x00\x00\x01\x00\x00\x00\x14stco\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x000\x00\x00\x02=trak\x00\x00\x00\\tkhd\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x05\xdd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000edts\x00\x00\x00(elst\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x01u\xff\xff\xff\xff\x00\x01\x00\x00\x00\x00\x04h\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x01\xa9mdia\x00\x00\x00 mdhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f@\x00\x00#@\x15\xc7\x00\x00\x00\x00\x00-hdlr\x00\x00\x00\x00\x00\x00\x00\x00soun\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00SoundHandler\x00\x00\x00\x01Tminf\x00\x00\x00\x10smhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$dinf\x00\x00\x00\x1cdref\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0curl \x00\x00\x00\x01\x00\x00\x01\x18stbl\x00\x00\x00~stsd\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00nmp4a\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x10\x00\x00\x00\x00\x1f@\x00\x00\x00\x00\x006esds\x00\x00\x00\x00\x03\x80\x80\x80%\x00\x02\x00\x04\x80\x80\x80\x17@\x15\x00\x00\x00\x00\x00\x00\xb4\x00\x00\x00\x94\x05\x80\x80\x80\x05\x15\x88V\xe5\x00\x06\x80\x80\x80\x01\x02\x00\x00\x00\x14btrt\x00\x00\x00\x00\x00\x00\x00\xb4\x00\x00\x00\x94\x00\x00\x00\x18stts\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00#@\x00\x00\x00\x1cstsc\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x14stsz\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x01\x00\x00\x00\x14stco\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x01\xbd\x00\x00\x00\x1asgpd\x01\x00\x00\x00roll\x00\x00\x00\x02\x00\x00\x00\x01\xff\xff\x00\x00\x00\x1csbgp\x00\x00\x00\x00roll\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00audta\x00\x00\x00Ymeta\x00\x00\x00\x00\x00\x00\x00!hdlr\x00\x00\x00\x00\x00\x00\x00\x00mdirappl\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00,ilst\x00\x00\x00$\xa9too\x00\x00\x00\x1cdata\x00\x00\x00\x01\x00\x00\x00\x00Lavf62.3.100"""


@pytest.fixture()
def mp4_split2_bytes() -> bytes:
    """
    Second segment of the small video file bytes.

    Split using
    ffmpeg -i <file_path>.mp4 -c copy -map 0 -f segment -segment_time 1.0 output%03d.mp4
    """
    return b"""\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41\x00\x00\x00\x08free\x00\x00\x01\xa5mdat\x00\x00\x00\x18gd\x00\n\xac\xd9B\x8d\xf9!\x00\x00\x03\x00\x01\x00\x00\x03\x00\x04\x0f\x12%\x96\x00\x00\x00\x05h\xef\x82\\\xb0\x00\x00\x01de\x88\x84\x00\x14\xff\xbc@k\xad-\xccR<o\xadJ\xf7\xd5\xabId\xe6\x98\x86\xde\xaf\xa9\xe7p\xcf\xf7\xe4\x02\x84\xcfZ\x8d\x95S\xedR\x81\xecy_\n\xc4PT\xb2ry\xac\x17=c\xf7\x88\x1b\xbe u\xe4\xba\x8a\xc1\xb9\xc5\x94>\xb3*[\x00\xc1\xd5SAl\xf21wh<\x04\x82\xa4\x1f9\xd2n^\xae\xa8\xd7O\xb2\x19J\x99\xd1\x18\rN\x84\xab\xcd\x8c\xbc\xfe\xfc\xa7\xa8u}[\x8bWi\rf \xf9\xae\xb1#\x13\xf1\xacyY\x19\xf5:xG\xc72\xbas\xf2\x97Eg\x0e\xa0\xaa\x1a\x0c\x94\x9e\t\xf9(\xb5\xe5\x9d\x8b\x01#\xac\xe2\x91|\xc6\xfd\xa5D\xdd\x9a\xf5\xb6\xd5L\x12\x03\xfa\xce\x90\x02\xe6\xa9\x18\xc9\xdc}\x88n\xfb\xfb*\x83&KNzJ\x11b\xbbm\xa0\xedS\x8c\x03(\'\x16\x9fc:\xff\xf8"\xa2b\xce]\xfc\xb2\xbe\x9d\xdc\xd4\x94\xa5Y \xe8\x8e \xa5\x87\x98\x07\x12-\xday\xdf}*\xdfh\x96\'\xc6\xbc\x82\x99\xfc\xf2~F\xf1\xef\xfb\x8b\xb6\x96\xc0{\xe9]\x84h\xd7\xbct\x1c\x9a\xaf\x01\xddsP\x84\'\xf6$G\xef\x12\x1e4\x8f1\xea\x0f\xb2\xc0\xed[\xa4\x00\xa9\x83:\xad\xe9\x86\xd1\xbf\xa0\xb6\xacp\xe5\x06\xc3\xf0v\x94\x04\x9e\xdffBWcO\xe9\xa9/M\xec\x8c\x14\x00\xa1\xa0\xc2\xa4bY\xd0\x030b\x02\x00\x1b}\xc3\xea\xa1s\x8b\xb3\x889\x82\xc5A\x01\x18 \x07\x01\x18 \x07\x01\x18 \x07\x01\x18 \x07\x00\x00\x05emoov\x00\x00\x00lmvhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x0b\xb8\x00\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x02Strak\x00\x00\x00\\tkhd\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x0b\xb8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\xa0\x00\x00\x00Z\x00\x00\x00\x00\x000edts\x00\x00\x00(elst\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\t\xc4\xff\xff\xff\xff\x00\x01\x00\x00\x00\x00\x01\xf4\x00\x00@\x00\x00\x01\x00\x00\x00\x00\x01\xbfmdia\x00\x00\x00 mdhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00 \x00\x15\xc7\x00\x00\x00\x00\x00-hdlr\x00\x00\x00\x00\x00\x00\x00\x00vide\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00VideoHandler\x00\x00\x00\x01jminf\x00\x00\x00\x14vmhd\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$dinf\x00\x00\x00\x1cdref\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0curl \x00\x00\x00\x01\x00\x00\x01*stbl\x00\x00\x00\xaestsd\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x9eavc1\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0\x00Z\x00H\x00\x00\x00H\x00\x00\x00\x00\x00\x00\x00\x01\x15Lavc62.11.100 libx264\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x18\xff\xff\x00\x00\x004avcC\x01d\x00\n\xff\xe1\x00\x18gd\x00\n\xac\xd9B\x8d\xf9!\x00\x00\x03\x00\x01\x00\x00\x03\x00\x04\x0f\x12%\x96\x01\x00\x05h\xef\x82\\\xb0\xfd\xf8\xf8\x00\x00\x00\x00\x14btrt\x00\x00\x00\x00\x00\x00\x18\xd0\x00\x00\x18\xd0\x00\x00\x00\x18stts\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00 \x00\x00\x00\x00\x18ctts\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00@\x00\x00\x00\x00\x1cstsc\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x14stsz\x00\x00\x00\x00\x00\x00\x01\x8d\x00\x00\x00\x01\x00\x00\x00\x14stco\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x000\x00\x00\x02=trak\x00\x00\x00\\tkhd\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x07\xdd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000edts\x00\x00\x00(elst\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x05\xdd\xff\xff\xff\xff\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x01\xa9mdia\x00\x00\x00 mdhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f@\x00\x00\x10\x00\x15\xc7\x00\x00\x00\x00\x00-hdlr\x00\x00\x00\x00\x00\x00\x00\x00soun\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00SoundHandler\x00\x00\x00\x01Tminf\x00\x00\x00\x10smhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$dinf\x00\x00\x00\x1cdref\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0curl \x00\x00\x00\x01\x00\x00\x01\x18stbl\x00\x00\x00~stsd\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00nmp4a\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x10\x00\x00\x00\x00\x1f@\x00\x00\x00\x00\x006esds\x00\x00\x00\x00\x03\x80\x80\x80%\x00\x02\x00\x04\x80\x80\x80\x17@\x15\x00\x00\x00\x00\x00\x00\xfa\x00\x00\x00\xfa\x05\x80\x80\x80\x05\x15\x88V\xe5\x00\x06\x80\x80\x80\x01\x02\x00\x00\x00\x14btrt\x00\x00\x00\x00\x00\x00\x00\xfa\x00\x00\x00\xfa\x00\x00\x00\x18stts\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x04\x00\x00\x00\x00\x1cstsc\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14stsz\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x14stco\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x01\xbd\x00\x00\x00\x1asgpd\x01\x00\x00\x00roll\x00\x00\x00\x02\x00\x00\x00\x01\xff\xff\x00\x00\x00\x1csbgp\x00\x00\x00\x00roll\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00audta\x00\x00\x00Ymeta\x00\x00\x00\x00\x00\x00\x00!hdlr\x00\x00\x00\x00\x00\x00\x00\x00mdirappl\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00,ilst\x00\x00\x00$\xa9too\x00\x00\x00\x1cdata\x00\x00\x00\x01\x00\x00\x00\x00Lavf62.3.100"""


@pytest.fixture()
def mp4_split2_width_320_bytes() -> bytes:
    """
    Second segment of the small video file bytes.

    Split using
    ffmpeg -i <file_path>.mp4 -c copy -map 0 -f segment -segment_time 1.0 output%03d.mp4

    resampled using
    ffmpeg -i <file_path>.mp4 -vf "fps=2" -c:v libx264  -c:a copy <output>_2fps.mp4
    """
    return b"""\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41\x00\x00\x00\x08free\x00\x00\x08\x97mdat\x01\x18 \x07\x01\x18 \x07\x01\x18 \x07\x01\x18 \x07\x00\x00\x02\xad\x06\x05\xff\xff\xa9\xdcE\xe9\xbd\xe6\xd9H\xb7\x96,\xd8 \xd9#\xee\xefx264 - core 165 r3222 b35605a - H.264/MPEG-4 AVC codec - Copyleft 2003-2025 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=6 lookahead_threads=1 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=2 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00\x00\x80\x00\x00\x05\xcae\x88\x84\x00\x12\xff\xfe\xe8\xc9\xfc\xcb-\x0e\'\xca\xedz\xda\xee\xa1\x1e\x8e\x98uz\x8eQ\xf2\x1bF"\xf2\x9d,\xa5e\xf0mI\x8d!\x12`\xc1"\x80.\xa4\xb8\xfc\xc5\xfe\x7fd=_!\x08GC9\xf2\xba;\xb1\xe4B\xff\x0e@\n\x12\xf4G\xa8 N\xc7\x91\xcd\xe2<sVE\xa4L\x94\x98\xb8d-\x82&\r\xd0"\x90\x1e\xcf\x1e\x10\xed\xd0X.\xbe|\x88\xfa\x1c\x0f\nU\xbf\xd1\xb4\xf3\xf1g\x08&Rw\x8d\xaabx\xa4\xe5~\x96\xf4\xc5\xaa\x81\x92!.\x82SjE\x1d\xb6E~\xeb\xc4\xd8\xa2\xa8bI\xeeNH\xc6\x0f\xb0E\xfa\x05\xaaF:c\x10i\xbf>\xd8\x1e\x8f\x1ff\xabI\xe4\xef\x89\xe0n\x81\xf1\xd4/\x9cK#\x1c\x80\x00\x03\xb2\xd3~\\\xa5\\\xa6\xe2\xe3\xc3|E\xea\x91\x95\x7f\x00\x01&\xd1a3H0\x1d\x1b5\x15\xb6\xa5q\xbf\xe7\xf9\xb8\x1b\xeai\xb7$\xcc_\xf0\xb8_\\\x87C\x80\x85\xf2\x14V\xb35|,2b\n}\x1a\x1co\xa3G\xdc,d\x88\x19/3fY\xe7\r\x9d?\t!\xf6\xbe\xd2\xbdt\xf1\xd5\x8a\x8b\xbe\xa3^\x92r\x9a\x80\xe7\xc5\x83\x88Q\x11F\xaa\xc0{\xb9~\x87\xdd\x9c\xd2\xfb}\xd5\x14\xc1\xad\x91t@\xb2Nb\xd4\xbdb\xd9\xc3,\xf7\xde\x93\xf9\x08\xa0\x98\xc5\xdf@\xfa-+n}?\x86\xe2{\xe8\x9e\xfe\x82ja\x8e\x10\xe2\x04zu}\xa6\xab\x87\x94\x11\xbe\xca\xc6(\x9e\xcb"\x9b\xaf\x15\n\x97F\x0f\xc5!z\xff\xe7\xd7\xc5s\xb5\xe3\xe97\x9f\xc6bG\xa6S_\x90\xf4\xca\x8d]\x97F\xe2\xe5\xc5Ej\xb7\x08p\xd0\xe9\xd0N\x1d&\xdc\xd6\x93]\xfc\xe0q\x85[\x10\xa2\x06O\xf6B\xa8\xef\xab\xf2\x92d\xf3\xbe\x9e\x98N\xbb\xbe\xe4\x96\xff\x1e\xd7\x19\x95fb\xb1\x12\xc4\xd7Ha\xe0_\xd6\x026\x05de\n\xe5\x9f1\x930qD\xaa9\xbb\x12\xccD\xb6\x9a\xaa\x96\x86[\xf7\xfa\x05N\xc5\x81uAz\x86|\x1b\xbeo\x91-{\x84k\xf50\xc6\xdcSF\x95\x1dS~-Q\xfc\x11\x06\xc9u\xf3\x05J\xde\x98|/R\xd4\x90(+\xc2\x9aB\x7fb\xa3\x86\xd7\xbf\xf7m%\xc2\x02+6\xbf\xba\xd7\x8bvw\x7f\x14\xd3f\x91v\xbe\xa1\x02\xce|\xbd:\x9a-Yik\xb7X\x02\xec\xff\xda\x13\xa1u\xd1\xa6\x06x\x90\xed\xad\xee\x13\x1b\xe5z\xdevv\xa7\x0c\xdc\xc4B\xdb\x81b\xbd\x1d\x83\xe2~\x02e\xcb\xcf\xf3\xf08\x91\'\xceQ\xd8\x98 /\x14\xa0\xe7\xd4.\xae:\xb6}\xba4r\x81\xe9\x1a\x96\x0f\xa6t)>\xae\x1d:E\x84\xc1)\x93zG\xc8\xe8\xebP\xc8N\x1btsw\xc7\xa1\x9d\xec\x1by\xfa\xc1\x18\xd3k\xa5\x06\xad\xae\x11\x9e:\xb2\x83\xd5M\x80\x9a\x13\x1ag$\x1c=\x90e\xc5lY`\x1b\x87\xd4W\xb6\x0br\xd1\xfd\x89\x06L\x85\xefA\xfa\'\xc6r\xddg\xa6\x1f<K\xfcEA\xd4}2\x9b\xe2d\x1d#\xe8\x04\x05R\xb9\xfa\xa7W~\x99%\xcc=\xf0U\xd6\xce\xa7\x02\x043.\xd0\x89\xae\xa2\xaenW\xc8p}<\xd9\xe5\r\x1c\x06\xed\x13\x05\xf3\xb2\x06/\x82X&o\xa4\x9ct\x8d$\xbe\xda\xa6?g\x96\xa2\xbcn&\xb6\x1d\x0f\x9f\x14\xc2\x01\'\x03\xfe\xa7-\x1e\xb3~JH\x9cI\xe4v\xea*\xfdL?}\xf9\xf7\xb7wbrt\x97\x8f\x0f\xbe\xb89C\xee\x08\x1d\x97\xacD!L\x8a\x8eE\xd2S\xe5\xb5i\x1a\xf8\xe7$\xeb\xcc@+\xf9D\xf6\x12R<\xd2\x18\x8b\xc4)\xe1\x7f\x10@y\xac\xfc\xcd,\xc8aKX\x0eQ\xac\x8b|GY\x8f\xeeWU\x9a6\xd5\xf3\xa1\'Y\xf0\xa4\xc4\xb7\xb2\xa9\x9c\xa8V<\nK\xf6\x97^\x8b6$!G\xff.\xa4ok/H\xe5\x9e|D!0\x16/\x8c\x9f\xc0\\\xdf\'M[oT\x85\xbe\x0c\x13\xe3\xcf\x8e}\x8am\xfb~\x9f\x03\xef_8\xdcH\xee\xce\x9d\xc61\xc1\xf6\xd7\'Vk\x89\x12}\xe6\x08\xeb\xc9\x95n\x15\xbb\xd0\x0cX\xe4\x08\xe9G\xe0{+T\xa3\t\xb4$\xf7\xc5\xf1\xdd\x08*\x07\xa8^\xed+f\x8dY\x17\xd1\x94\xeb\x88J\x97\xb1\xff\xbf\x12R\xffA\xaf[D\x96\x0c%\xd2m)\x12\n\x89\xd1\x88\x05+\xa0\x1d\xa0&\xb18\x02SG\xc7\x97&\x1c8s9\xaaI\xeduj\xd2\x01\xf2\xc5\xb5\xe2\xa4\xf0&\xaf\xac\xb2\xd7\xf8\x95\xc8Y\x05+\x1d\x88\xd5^\x95\x05nUR\xe4\xdf\r\x8e\xc1\xffeQ\xda\xe1\x89Yh\x84B\xe9\xd3DO\xc8{\xf8\xfe\xcd\x1dCiu\x8el\x87S\xb8&|>\x9agz/\xf9\x06&\xa2\xc7r\xbc\xb6\xfe\x80\xc7\x12\x9f\xae\xcf\xf3\x1ao\xc6=\xeb0\xcdR\xec%Z\xf57ZKv\x17\xb6\x9f!\xad\x84\\\xc0u1\x86\xe0\x93\xf9\xcfC8"\x87\xdc\xc2\xc4\xe0,\xec\x9e\xfb\xd7\x89z\xed2V8>z\xd8\'s\xd9\x16\x12\xc0\x1eX\'O(\xe8\x10\xaf\'&\x83Ak\x1fT\xfa\xc6y\xfbm\xd9\xbd;H\xf0\xfa1\xeck\'\xe4yO\xac\x0f\x7f\xba\xea\xd6\xb2\xbbtt\xac\x95\xbf\x0b3\x147\xee\xa8\x80xr&\xfb\xc2\xe6\xec\xad\xc3\x94\x83 \x95\xd4f,ov\x9a\xc0+Sp\xac\x82\xf4\xec3\x8f\xd2\x81\xc2zX\xce\xbcH\x0c\x94r\x84\xd0\x96\x9bZ\x1b\x7fO\xec\xd1\xf8%E0\x9c\xc9\xa4\xe0V\xe2\xacb\xca\xe1\xf2\tJ\xa6\x95\x9d\xc0\xb2\x9di\x9a\x9e\xf2.O\x88XX6\x1e\xe7\xaf\xef\x99\x84\xf8l\xe0\xe8\x15\xb1q\x9f\x89\x83\x11YG\x14mpD"\xb37Z\x81\xedS\x9b.rk\xf3d\x827\xc4L\xf0\x18\xdc-\x9c\x9dK/ \xce\x9eA\x89"\xc0\xc8v\xec\xc8\xf68\x9e\xf0\x87\xf2\xfd]DU9o\xa3_\xb1s\x9d\x9dH\x17H=U\xdd\xed\xb2Fok\xdcYf\x83\xca\x19\x8f/\xf9^\t\x15-\n~\xa1P\xd7\xc7M\xb0Y\xc7\xc2\xfa\xaa"\x8a\x1e\xb2\x06\xc0F\x94\x00H\x11\x83\x00\x00\x05Cmoov\x00\x00\x00lmvhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x05\xdc\x00\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x02=trak\x00\x00\x00\\tkhd\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x05\xdc\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x01@\x00\x00\x00\xb4\x00\x00\x00\x00\x000edts\x00\x00\x00(elst\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x03\xe8\xff\xff\xff\xff\x00\x01\x00\x00\x00\x00\x01\xf4\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x01\xa9mdia\x00\x00\x00 mdhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00 \x00\x15\xc7\x00\x00\x00\x00\x00-hdlr\x00\x00\x00\x00\x00\x00\x00\x00vide\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00VideoHandler\x00\x00\x00\x01Tminf\x00\x00\x00\x14vmhd\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$dinf\x00\x00\x00\x1cdref\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0curl \x00\x00\x00\x01\x00\x00\x01\x14stbl\x00\x00\x00\xb0stsd\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xa0avc1\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01@\x00\xb4\x00H\x00\x00\x00H\x00\x00\x00\x00\x00\x00\x00\x01\x15Lavc62.11.100 libx264\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x18\xff\xff\x00\x00\x006avcC\x01d\x00\x0c\xff\xe1\x00\x19gd\x00\x0c\xac\xd9AA\x9f\x9e\x10\x00\x00\x03\x00\x10\x00\x00\x03\x00@\xf1B\x99`\x01\x00\x06h\xeb\xe3\xcb"\xc0\xfd\xf8\xf8\x00\x00\x00\x00\x14btrt\x00\x00\x00\x00\x00\x00\x87\xf0\x00\x00\x00\x00\x00\x00\x00\x18stts\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00 \x00\x00\x00\x00\x1cstsc\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x14stsz\x00\x00\x00\x00\x00\x00\x08\x7f\x00\x00\x00\x01\x00\x00\x00\x14stco\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00@\x00\x00\x021trak\x00\x00\x00\\tkhd\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$edts\x00\x00\x00\x1celst\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x01\xa9mdia\x00\x00\x00 mdhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f@\x00\x00\x10\x00\x15\xc7\x00\x00\x00\x00\x00-hdlr\x00\x00\x00\x00\x00\x00\x00\x00soun\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00SoundHandler\x00\x00\x00\x01Tminf\x00\x00\x00\x10smhd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$dinf\x00\x00\x00\x1cdref\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0curl \x00\x00\x00\x01\x00\x00\x01\x18stbl\x00\x00\x00~stsd\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00nmp4a\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x10\x00\x00\x00\x00\x1f@\x00\x00\x00\x00\x006esds\x00\x00\x00\x00\x03\x80\x80\x80%\x00\x02\x00\x04\x80\x80\x80\x17@\x15\x00\x00\x00\x00\x00\x00\xfa\x00\x00\x00\xfa\x05\x80\x80\x80\x05\x15\x88V\xe5\x00\x06\x80\x80\x80\x01\x02\x00\x00\x00\x14btrt\x00\x00\x00\x00\x00\x00\x00\xfa\x00\x00\x00\xfa\x00\x00\x00\x18stts\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x04\x00\x00\x00\x00\x1cstsc\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14stsz\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x14stco\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x000\x00\x00\x00\x1asgpd\x01\x00\x00\x00roll\x00\x00\x00\x02\x00\x00\x00\x01\xff\xff\x00\x00\x00\x1csbgp\x00\x00\x00\x00roll\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00audta\x00\x00\x00Ymeta\x00\x00\x00\x00\x00\x00\x00!hdlr\x00\x00\x00\x00\x00\x00\x00\x00mdirappl\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00,ilst\x00\x00\x00$\xa9too\x00\x00\x00\x1cdata\x00\x00\x00\x01\x00\x00\x00\x00Lavf62.3.100"""


@pytest.fixture()
def mp4_base64(mp4_bytes: bytes) -> bytes:
    return base64.b64encode(mp4_bytes)


@pytest.fixture()
def mp4_split1_base64(mp4_split1_bytes: bytes) -> bytes:
    return base64.b64encode(mp4_split1_bytes)


@pytest.fixture()
def mp4_split2_base64(mp4_split2_bytes: bytes) -> bytes:
    return base64.b64encode(mp4_split2_bytes)


@pytest.fixture()
def mp4_split2_width_320_base64(mp4_split2_width_320_bytes: bytes) -> bytes:
    return base64.b64encode(mp4_split2_width_320_bytes)


@pytest.fixture()
def mock_ffprobe_mp4_bytes_output():
    return """{
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_long_name": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                "profile": "High",
                "codec_type": "video",
                "codec_tag_string": "avc1",
                "codec_tag": "0x31637661",
                "width": 160,
                "height": 90,
                "coded_width": 160,
                "coded_height": 90,
                "has_b_frames": 2,
                "pix_fmt": "yuv420p",
                "level": 10,
                "chroma_location": "left",
                "field_order": "progressive",
                "refs": 1,
                "is_avc": "true",
                "nal_length_size": "4",
                "id": "0x1",
                "r_frame_rate": "2/3",
                "avg_frame_rate": "1/1",
                "time_base": "1/16384",
                "start_pts": 10273,
                "start_time": "0.627014",
                "duration_ts": 32768,
                "duration": "2.000000",
                "bit_rate": "3176",
                "bits_per_raw_sample": "8",
                "nb_frames": "2",
                "extradata_size": 44,
                "disposition": {
                    "default": 1,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                },
                "tags": {
                    "language": "eng",
                    "handler_name": "VideoHandler",
                    "vendor_id": "[0][0][0][0]",
                    "encoder": "Lavc62.11.100 libx264"
                }
            },
            {
                "index": 1,
                "codec_name": "aac",
                "codec_long_name": "AAC (Advanced Audio Coding)",
                "profile": "LC",
                "codec_type": "audio",
                "codec_tag_string": "mp4a",
                "codec_tag": "0x6134706d",
                "sample_fmt": "fltp",
                "sample_rate": "8000",
                "channels": 1,
                "channel_layout": "mono",
                "bits_per_sample": 0,
                "initial_padding": 0,
                "id": "0x2",
                "r_frame_rate": "0/0",
                "avg_frame_rate": "0/0",
                "time_base": "1/8000",
                "start_pts": 0,
                "start_time": "0.000000",
                "duration_ts": 13120,
                "duration": "1.640000",
                "bit_rate": "180",
                "nb_frames": "5",
                "extradata_size": 5,
                "disposition": {
                    "default": 1,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                },
                "tags": {
                    "language": "eng",
                    "handler_name": "SoundHandler",
                    "vendor_id": "[0][0][0][0]"
                }
            }
        ],
        "format": {
            "filename": "/var/folders/43/2mzmbvc14xxgrggdtw27nhww0000gp/T/tmpekmfsmg2/input.mp4",
            "nb_streams": 2,
            "nb_programs": 0,
            "nb_stream_groups": 0,
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "format_long_name": "QuickTime / MOV",
            "start_time": "0.000000",
            "duration": "2.627014",
            "size": "2304",
            "bit_rate": "7016",
            "probe_score": 100,
            "tags": {
                "major_brand": "isom",
                "minor_version": "512",
                "compatible_brands": "isomiso2avc1mp41",
                "encoder": "Lavf62.3.100"
            }
        }
    }"""


@pytest.fixture()
def mock_ffprobe_mp4_split1_bytes_output():
    return """{
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_long_name": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                "profile": "High",
                "codec_type": "video",
                "codec_tag_string": "avc1",
                "codec_tag": "0x31637661",
                "width": 160,
                "height": 90,
                "coded_width": 160,
                "coded_height": 90,
                "has_b_frames": 2,
                "pix_fmt": "yuv420p",
                "level": 10,
                "chroma_location": "left",
                "field_order": "progressive",
                "refs": 1,
                "is_avc": "true",
                "nal_length_size": "4",
                "id": "0x1",
                "r_frame_rate": "2/3",
                "avg_frame_rate": "2/3",
                "time_base": "1/16384",
                "start_pts": 16384,
                "start_time": "1.000000",
                "duration_ts": 24576,
                "duration": "1.500000",
                "bit_rate": "2117",
                "bits_per_raw_sample": "8",
                "nb_frames": "1",
                "extradata_size": 44,
                "disposition": {
                    "default": 1,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                },
                "tags": {
                    "language": "eng",
                    "handler_name": "VideoHandler",
                    "vendor_id": "[0][0][0][0]",
                    "encoder": "Lavc62.11.100 libx264"
                }
            },
            {
                "index": 1,
                "codec_name": "aac",
                "codec_long_name": "AAC (Advanced Audio Coding)",
                "profile": "LC",
                "codec_type": "audio",
                "codec_tag_string": "mp4a",
                "codec_tag": "0x6134706d",
                "sample_fmt": "fltp",
                "sample_rate": "8000",
                "channels": 1,
                "channel_layout": "mono",
                "bits_per_sample": 0,
                "initial_padding": 0,
                "id": "0x2",
                "r_frame_rate": "0/0",
                "avg_frame_rate": "0/0",
                "time_base": "1/8000",
                "start_pts": 2984,
                "start_time": "0.373000",
                "duration_ts": 9024,
                "duration": "1.128000",
                "bit_rate": "148",
                "nb_frames": "1",
                "extradata_size": 5,
                "disposition": {
                    "default": 1,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                },
                "tags": {
                    "language": "eng",
                    "handler_name": "SoundHandler",
                    "vendor_id": "[0][0][0][0]"
                }
            }
        ],
        "format": {
            "filename": "/var/folders/43/2mzmbvc14xxgrggdtw27nhww0000gp/T/tmpsg6g722g/input.mp4",
            "nb_streams": 2,
            "nb_programs": 0,
            "nb_stream_groups": 0,
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "format_long_name": "QuickTime / MOV",
            "start_time": "0.373000",
            "duration": "2.127000",
            "size": "1847",
            "bit_rate": "6946",
            "probe_score": 100,
            "tags": {
                "major_brand": "isom",
                "minor_version": "512",
                "compatible_brands": "isomiso2avc1mp41",
                "encoder": "Lavf62.3.100"
            }
        }
    }"""


@pytest.fixture()
def mock_ffprobe_mp4_split2_bytes_output():
    return """{
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_long_name": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                "profile": "High",
                "codec_type": "video",
                "codec_tag_string": "avc1",
                "codec_tag": "0x31637661",
                "width": 160,
                "height": 90,
                "coded_width": 160,
                "coded_height": 90,
                "has_b_frames": 2,
                "pix_fmt": "yuv420p",
                "level": 10,
                "chroma_location": "left",
                "field_order": "progressive",
                "refs": 1,
                "is_avc": "true",
                "nal_length_size": "4",
                "id": "0x1",
                "r_frame_rate": "2/1",
                "avg_frame_rate": "2/1",
                "time_base": "1/16384",
                "start_pts": 40960,
                "start_time": "2.500000",
                "duration_ts": 8192,
                "duration": "0.500000",
                "bit_rate": "6352",
                "bits_per_raw_sample": "8",
                "nb_frames": "1",
                "extradata_size": 44,
                "disposition": {
                    "default": 1,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                },
                "tags": {
                    "language": "eng",
                    "handler_name": "VideoHandler",
                    "vendor_id": "[0][0][0][0]",
                    "encoder": "Lavc62.11.100 libx264"
                }
            },
            {
                "index": 1,
                "codec_name": "aac",
                "codec_long_name": "AAC (Advanced Audio Coding)",
                "profile": "LC",
                "codec_type": "audio",
                "codec_tag_string": "mp4a",
                "codec_tag": "0x6134706d",
                "sample_fmt": "fltp",
                "sample_rate": "8000",
                "channels": 1,
                "channel_layout": "mono",
                "bits_per_sample": 0,
                "initial_padding": 0,
                "id": "0x2",
                "r_frame_rate": "0/0",
                "avg_frame_rate": "0/0",
                "time_base": "1/8000",
                "start_pts": 12008,
                "start_time": "1.501000",
                "duration_ts": 4096,
                "duration": "0.512000",
                "bit_rate": "250",
                "nb_frames": "4",
                "extradata_size": 5,
                "disposition": {
                    "default": 1,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                },
                "tags": {
                    "language": "eng",
                    "handler_name": "SoundHandler",
                    "vendor_id": "[0][0][0][0]"
                }
            }
        ],
        "format": {
            "filename": "/var/folders/43/2mzmbvc14xxgrggdtw27nhww0000gp/T/tmpnbfct8h1/input.mp4",
            "nb_streams": 2,
            "nb_programs": 0,
            "nb_stream_groups": 0,
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "format_long_name": "QuickTime / MOV",
            "start_time": "1.501000",
            "duration": "1.499000",
            "size": "1842",
            "bit_rate": "9830",
            "probe_score": 100,
            "tags": {
                "major_brand": "isom",
                "minor_version": "512",
                "compatible_brands": "isomiso2avc1mp41",
                "encoder": "Lavf62.3.100"
            }
        }
    }"""


@pytest.fixture()
def mock_ffprobe_mp4_split2_width_320_bytes_output():
    return """{
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_long_name": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                "profile": "High",
                "codec_type": "video",
                "codec_tag_string": "avc1",
                "codec_tag": "0x31637661",
                "width": 320,
                "height": 180,
                "coded_width": 320,
                "coded_height": 180,
                "has_b_frames": 2,
                "pix_fmt": "yuv420p",
                "level": 12,
                "chroma_location": "left",
                "field_order": "progressive",
                "refs": 1,
                "is_avc": "true",
                "nal_length_size": "4",
                "id": "0x1",
                "r_frame_rate": "2/1",
                "avg_frame_rate": "2/1",
                "time_base": "1/16384",
                "start_pts": 16384,
                "start_time": "1.000000",
                "duration_ts": 8192,
                "duration": "0.500000",
                "bit_rate": "34800",
                "bits_per_raw_sample": "8",
                "nb_frames": "1",
                "extradata_size": 46,
                "disposition": {
                    "default": 1,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                },
                "tags": {
                    "language": "eng",
                    "handler_name": "VideoHandler",
                    "vendor_id": "[0][0][0][0]",
                    "encoder": "Lavc62.11.100 libx264"
                }
            },
            {
                "index": 1,
                "codec_name": "aac",
                "codec_long_name": "AAC (Advanced Audio Coding)",
                "profile": "LC",
                "codec_type": "audio",
                "codec_tag_string": "mp4a",
                "codec_tag": "0x6134706d",
                "sample_fmt": "fltp",
                "sample_rate": "8000",
                "channels": 1,
                "channel_layout": "mono",
                "bits_per_sample": 0,
                "initial_padding": 0,
                "id": "0x2",
                "r_frame_rate": "0/0",
                "avg_frame_rate": "0/0",
                "time_base": "1/8000",
                "start_pts": 0,
                "start_time": "0.000000",
                "duration_ts": 4096,
                "duration": "0.512000",
                "bit_rate": "250",
                "nb_frames": "4",
                "extradata_size": 5,
                "disposition": {
                    "default": 1,
                    "dub": 0,
                    "original": 0,
                    "comment": 0,
                    "lyrics": 0,
                    "karaoke": 0,
                    "forced": 0,
                    "hearing_impaired": 0,
                    "visual_impaired": 0,
                    "clean_effects": 0,
                    "attached_pic": 0,
                    "timed_thumbnails": 0,
                    "non_diegetic": 0,
                    "captions": 0,
                    "descriptions": 0,
                    "metadata": 0,
                    "dependent": 0,
                    "still_image": 0,
                    "multilayer": 0
                },
                "tags": {
                    "language": "eng",
                    "handler_name": "SoundHandler",
                    "vendor_id": "[0][0][0][0]"
                }
            }
        ],
        "format": {
            "filename": "/var/folders/43/2mzmbvc14xxgrggdtw27nhww0000gp/T/tmpycd5dnv4/input.mp4",
            "nb_streams": 2,
            "nb_programs": 0,
            "nb_stream_groups": 0,
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "format_long_name": "QuickTime / MOV",
            "start_time": "0.000000",
            "duration": "1.500000",
            "size": "3586",
            "bit_rate": "19125",
            "probe_score": 100,
            "tags": {
                "major_brand": "isom",
                "minor_version": "512",
                "compatible_brands": "isomiso2avc1mp41",
                "encoder": "Lavf62.3.100"
            }
        }
    }"""


@pytest.fixture()
def mock_ffmpeg_error_side_effect():
    async def _side_effect(*args, **kwargs):
        raise FFmpegError("Mock error")

    return _side_effect


@pytest.fixture()
def mock_ffprobe(
    mp3_bytes,
    mp3_split1_bytes,
    mp3_split2_bytes,
    mp3_split2_sr_8000_bytes,
    mp4_bytes,
    mp4_split1_bytes,
    mp4_split2_bytes,
    mp4_split2_width_320_bytes,
    mock_ffprobe_mp3_bytes_output,
    mock_ffprobe_mp3_split1_bytes_output,
    mock_ffprobe_mp3_split2_bytes_output,
    mock_ffprobe_mp3_split2_sr8000_bytes_output,
    mock_ffprobe_mp4_bytes_output,
    mock_ffprobe_mp4_split1_bytes_output,
    mock_ffprobe_mp4_split2_bytes_output,
    mock_ffprobe_mp4_split2_width_320_bytes_output,
    mock_ffmpeg_error_side_effect,
):
    def _mock(tmp_path, error: bool = False):
        with open(tmp_path, "rb") as f:
            bytes_str = f.read()
        if bytes_str == mp3_bytes:
            execute_out = mock_ffprobe_mp3_bytes_output
        elif bytes_str == mp3_split1_bytes:
            execute_out = mock_ffprobe_mp3_split1_bytes_output
        elif bytes_str == mp3_split2_bytes:
            execute_out = mock_ffprobe_mp3_split2_bytes_output
        elif bytes_str == mp3_split2_sr_8000_bytes:
            execute_out = mock_ffprobe_mp3_split2_sr8000_bytes_output
        elif bytes_str == mp4_bytes:
            execute_out = mock_ffprobe_mp4_bytes_output
        elif bytes_str == mp4_split1_bytes:
            execute_out = mock_ffprobe_mp4_split1_bytes_output
        elif bytes_str == mp4_split2_bytes:
            execute_out = mock_ffprobe_mp4_split2_bytes_output
        elif bytes_str == mp4_split2_width_320_bytes:
            execute_out = mock_ffprobe_mp4_split2_width_320_bytes_output
        else:
            raise ValueError("Unrecognized bytes input for ffprobe mock.")
        mock = Mock(spec=FFmpeg)
        if not error:
            mock.execute = AsyncMock(return_value=execute_out)
        else:
            mock.execute = AsyncMock(side_effect=mock_ffmpeg_error_side_effect)
        return mock

    return _mock


@pytest.fixture()
def mock_ffmpeg_segment(
    mp3_bytes,
    mp3_split1_bytes,
    mp3_split2_bytes,
    mp4_bytes,
    mp4_split1_bytes,
    mp4_split2_bytes,
    mock_ffmpeg_error_side_effect,
):
    def _mock(out_pattern: str, error: bool = False):
        out_dir = Path(out_pattern).parent
        out_padding = int(
            Path(out_pattern)
            .stem.replace("output", "")
            .replace("%", "")
            .replace("d", "")
        )
        ext = Path(out_pattern).suffix

        input_file = out_dir / f"input{ext}"
        with open(input_file, "rb") as f:
            in_bytes = f.read()

        if in_bytes == mp3_bytes:
            for i, out_bytes_str in enumerate([mp3_split1_bytes, mp3_split2_bytes]):
                out_path = f"{out_dir}/output{str(i).zfill(out_padding)}{ext}"
                with open(out_path, "wb") as f:
                    f.write(out_bytes_str)
        elif in_bytes == mp4_bytes:
            for i, out_bytes_str in enumerate([mp4_split1_bytes, mp4_split2_bytes]):
                out_path = f"{out_dir}/output{str(i).zfill(out_padding)}{ext}"
                with open(out_path, "wb") as f:
                    f.write(out_bytes_str)
        else:
            raise ValueError("Unrecognized bytes input for ffmpeg segment mock.")
        mock = Mock(spec=FFmpeg)
        if not error:
            mock.execute = AsyncMock(return_value=None)
        else:
            mock.execute = AsyncMock(side_effect=mock_ffmpeg_error_side_effect)
        return mock

    return _mock


@pytest.fixture()
def mock_ffmpeg_trim(
    mp3_bytes,
    mp3_split1_bytes,
    mp3_split2_bytes,
    mp4_bytes,
    mp4_split1_bytes,
    mp4_split2_bytes,
    mock_ffmpeg_error_side_effect,
):
    def _mock(out_path: str, reverse: bool, error: bool = False):
        out_dir = Path(out_path).parent
        ext = Path(out_path).suffix
        input_file = out_dir / f"input{ext}"
        with open(input_file, "rb") as f:
            in_bytes = f.read()

        if in_bytes == mp3_bytes:
            if not reverse:
                out_bytes = mp3_split1_bytes
            else:
                out_bytes = mp3_split2_bytes
        elif in_bytes == mp4_bytes:
            if not reverse:
                out_bytes = mp4_split1_bytes
            else:
                out_bytes = mp4_split2_bytes
        else:
            raise ValueError("Unrecognized bytes input for ffmpeg trim mock.")
        with open(out_path, "wb") as f:
            f.write(out_bytes)
        mock = Mock(spec=FFmpeg)
        if not error:
            mock.execute = AsyncMock(return_value=None)
        else:
            mock.execute = AsyncMock(side_effect=mock_ffmpeg_error_side_effect)
        return mock

    return _mock


@pytest.fixture()
def mock_ffmpeg_concat(mp3_bytes, mp4_bytes, mock_ffmpeg_error_side_effect):
    def _mock(out_path: str, error: bool = False):
        if out_path.endswith(".mp4"):
            with open(out_path, "wb") as f:
                f.write(mp4_bytes)
        elif out_path.endswith(".mp3"):
            with open(out_path, "wb") as f:
                f.write(mp3_bytes)
        else:
            raise ValueError("Unrecognized output path for ffmpeg concat mock.")
        mock = Mock(spec=FFmpeg)
        if not error:
            mock.execute = AsyncMock(return_value=None)
        else:
            mock.execute = AsyncMock(side_effect=mock_ffmpeg_error_side_effect)
        return mock

    return _mock


@pytest.fixture()
def _mock_ffmpeg(
    mock_ffprobe, mock_ffmpeg_segment, mock_ffmpeg_trim, mock_ffmpeg_concat
):
    def _mock(error: str | None = None):
        def mock_input(*in_args, **in_kwargs):
            def mock_output(*out_args, **out_kwargs):
                # This parameter is only supplied when ffmpeg segment is called
                if out_kwargs.get("f") == "segment":
                    return mock_ffmpeg_segment(*out_args, error=error == "segment")
                elif out_kwargs.get("t") is not None:
                    reverse = bool(in_kwargs.get("sseof"))
                    return mock_ffmpeg_trim(
                        *out_args, reverse=reverse, error=error == "trim"
                    )
                # Otherwise we mock the concat output
                return mock_ffmpeg_concat(*out_args, error=error == "concat")

            # This parameter is only supplied when ffprobe is called
            if "print_format" in in_kwargs:
                return mock_ffprobe(*in_args, error=error == "ffprobe")

            # Otherwise we mock the output
            mock = AsyncMock(spec=FFmpeg)
            mock.output.side_effect = mock_output
            return mock

        with (
            mock.patch(
                "llama_index.core.base.llms.types.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            mock.patch(
                "llama_index.core.base.llms.types.FFmpeg", spec=FFmpeg
            ) as mock_ffmpeg,
        ):
            mock_ffmpeg.return_value.input.side_effect = mock_input
            yield mock_ffmpeg

    return _mock


@pytest.fixture()
def mock_ffmpeg(_mock_ffmpeg):
    yield from _mock_ffmpeg(error=None)


@pytest.fixture()
def mock_ffmpeg_ffprobe_error(_mock_ffmpeg):
    yield from _mock_ffmpeg(error="ffprobe")


@pytest.fixture()
def mock_ffmpeg_segment_error(_mock_ffmpeg):
    yield from _mock_ffmpeg(error="segment")


@pytest.fixture()
def mock_ffmpeg_trim_error(_mock_ffmpeg):
    yield from _mock_ffmpeg(error="trim")


@pytest.fixture()
def mock_ffmpeg_concat_error(_mock_ffmpeg):
    yield from _mock_ffmpeg(error="concat")


@pytest.fixture()
def mock_no_ffmpeg():
    with mock.patch(
        "llama_index.core.base.llms.types.shutil.which", return_value=None
    ) as which_ffmpeg:
        yield which_ffmpeg


@pytest.fixture()
def mock_tiny_tag_error():
    def raise_tiny_tag_error(*args, **kwargs):
        raise UnsupportedFormatError

    with mock.patch.object(
        TinyTag, "get", side_effect=raise_tiny_tag_error
    ) as mock_tinytag:
        yield mock_tinytag


def test_chat_message_from_str():
    m = ChatMessage.from_str(content="test content")
    assert m.content == "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"


def test_chat_message_content_legacy_get():
    m = ChatMessage(content="test content")
    assert m.content == "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(role="user", content="test content")
    assert m.role == "user"
    assert m.content == "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(
        content=[TextBlock(text="test content 1"), TextBlock(text="test content 2")]
    )
    assert m.content == "test content 1\ntest content 2"
    assert len(m.blocks) == 2
    assert all(type(block) is TextBlock for block in m.blocks)


def test_chat_message_content_legacy_set():
    m = ChatMessage()
    m.content = "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(content="some original content")
    m.content = "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(content=[TextBlock(text="test content"), ImageBlock()])
    with pytest.raises(ValueError):
        m.content = "test content"


def test_chat_message_content_returns_empty_string():
    m = ChatMessage(content=[TextBlock(text="test content"), ImageBlock()])
    assert m.content == "test content"
    m = ChatMessage()
    assert m.content is None


def test_chat_message__str__():
    assert str(ChatMessage(content="test content")) == "user: test content"


def test_chat_message_serializer():
    class SimpleModel(BaseModel):
        some_field: str = ""

    m = ChatMessage(
        content="test content",
        additional_kwargs={"some_list": ["a", "b", "c"], "some_object": SimpleModel()},
    )
    assert m.model_dump() == {
        "role": MessageRole.USER,
        "additional_kwargs": {
            "some_list": ["a", "b", "c"],
            "some_object": {"some_field": ""},
        },
        "blocks": [{"block_type": "text", "text": "test content"}],
    }


def test_chat_message_legacy_roundtrip():
    legacy_message = {
        "role": MessageRole.USER,
        "content": "foo",
        "additional_kwargs": {},
    }
    m = ChatMessage(**legacy_message)
    assert m.model_dump() == {
        "additional_kwargs": {},
        "blocks": [{"block_type": "text", "text": "foo"}],
        "role": MessageRole.USER,
    }


def test_chat_response():
    message = ChatMessage("some content")
    cr = ChatResponse(message=message)
    assert str(cr) == str(message)


def test_completion_response():
    cr = CompletionResponse(text="some text")
    assert str(cr) == "some text"


@pytest.mark.asyncio
async def test_text_block_aestimate_tokens_default_tokenizer():
    tb = TextBlock(text="Hello world!")

    tknzr = get_tokenizer()
    assert await tb.aestimate_tokens() == len(tknzr(tb.text))


@pytest.mark.asyncio
async def test_text_block_aestimate_tokens_custom_tokenizer():
    tb = TextBlock(text="Hello world!")

    mock_tknzer = Mock(spec=type(get_tokenizer()))
    mock_tknzer.return_value = list(range(100))
    assert await tb.aestimate_tokens(tokenizer=mock_tknzer) == 100


@pytest.mark.asyncio
async def test_text_block_asplit_no_overlap():
    tb = TextBlock(text="Hello world! This is a test.")

    chunks = await tb.asplit(max_tokens=3)
    splitter = TokenTextSplitter(chunk_size=3, chunk_overlap=0)
    assert len(chunks) == len(splitter.split_text(tb.text))


@pytest.mark.asyncio
async def test_text_block_atruncate():
    tb = TextBlock(text="Hello world! This is a test.")
    truncated_tb = await tb.atruncate(max_tokens=4)
    truncated_tb_reverse = await tb.atruncate(max_tokens=4, reverse=True)
    assert await tb.aestimate_tokens() > 4
    assert await truncated_tb.aestimate_tokens() <= 4
    assert await truncated_tb_reverse.aestimate_tokens() <= 4
    assert truncated_tb.text == "Hello world! This"
    assert truncated_tb_reverse.text == "is a test."


@pytest.mark.asyncio
async def test_text_block_amerge():
    tb1 = TextBlock(text="Hello world!")
    tb2 = TextBlock(text="This is a test.")
    merged_tb = await TextBlock.amerge([tb1, tb2], chunk_size=100)
    assert len(merged_tb) == 1
    assert merged_tb[0].text == "Hello world! This is a test."


def test_image_block_resolve_image(png_1px: bytes, png_1px_b64: bytes):
    b = ImageBlock(image=png_1px)

    img = b.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px

    img = b.resolve_image(as_base64=True)
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px_b64


def test_image_block_resolve_image_buffer(png_1px: bytes):
    buffer = BytesIO(png_1px)
    b = ImageBlock(image=buffer)

    img = b.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px


def test_image_block_resolve_image_path(
    tmp_path: Path, png_1px_b64: bytes, png_1px: bytes
):
    png_path = tmp_path / "test.png"
    png_path.write_bytes(png_1px)

    b = ImageBlock(path=png_path)
    img = b.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px

    img = b.resolve_image(as_base64=True)
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px_b64


def test_image_block_resolve_image_url(png_1px_b64: bytes, png_1px: bytes):
    with mock.patch("llama_index.core.utils.requests") as mocked_req:
        url_str = "http://example.com"
        mocked_req.get.return_value = mock.MagicMock(content=png_1px)
        b = ImageBlock(url=AnyUrl(url=url_str))
        img = b.resolve_image()
        assert isinstance(img, BytesIO)
        assert img.read() == png_1px

        img = b.resolve_image(as_base64=True)
        assert isinstance(img, BytesIO)
        assert img.read() == png_1px_b64


def test_image_block_resolve_image_data_url_base64(png_1px_b64: bytes, png_1px: bytes):
    # Test data URL with base64 encoding
    data_url = f"data:image/png;base64,{png_1px_b64.decode('utf-8')}"
    b = ImageBlock(url=AnyUrl(url=data_url))

    img = b.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px

    img = b.resolve_image(as_base64=True)
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px_b64


def test_image_block_resolve_image_data_url_plain_text():
    # Test data URL with plain text (no base64)
    test_text = "Hello, World!"
    data_url = f"data:text/plain,{test_text}"
    b = ImageBlock(url=AnyUrl(url=data_url))

    img = b.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == test_text.encode("utf-8")

    img = b.resolve_image(as_base64=True)
    assert isinstance(img, BytesIO)
    assert img.read() == base64.b64encode(test_text.encode("utf-8"))


def test_image_block_resolve_image_data_url_invalid():
    # Test invalid data URL format (missing comma)
    invalid_data_url = "data:image/png;base64"
    b = ImageBlock(url=AnyUrl(url=invalid_data_url))

    with pytest.raises(
        ValueError, match="Invalid data URL format: missing comma separator"
    ):
        b.resolve_image()


def test_image_block_resolve_error():
    with pytest.raises(
        ValueError, match="No valid source provided to resolve binary data!"
    ):
        b = ImageBlock()
        b.resolve_image()


def test_image_block_store_as_anyurl():
    url_str = "http://example.com"
    b = ImageBlock(url=url_str)
    assert b.url == AnyUrl(url=url_str)


def test_image_block_store_as_base64(png_1px_b64: bytes, png_1px: bytes):
    # Store regular bytes
    assert ImageBlock(image=png_1px).image == png_1px_b64
    # Store already encoded data
    assert ImageBlock(image=png_1px_b64).image == png_1px_b64


def test_legacy_image_additional_kwargs(png_1px_b64: bytes):
    image_doc = ImageDocument(image=png_1px_b64)
    msg = ChatMessage(additional_kwargs={"images": [image_doc]})
    assert len(msg.blocks) == 1
    assert msg.blocks[0].image == png_1px_b64


@pytest.mark.asyncio
async def test_image_block_aestimate_tokens(png_1px_b64: bytes):
    ib = ImageBlock(image=png_1px_b64)
    assert await ib.aestimate_tokens() == 258


@pytest.mark.asyncio
async def test_image_block_asplit(png_1px_b64: bytes):
    ib = ImageBlock(image=png_1px_b64)
    chunks = await ib.asplit(max_tokens=2)

    # Images are not splittable
    assert chunks == [ib]


@pytest.mark.asyncio
async def test_image_block_atruncate(png_1px_b64: bytes):
    ib = ImageBlock(image=png_1px_b64)
    truncated_ib = await ib.atruncate(max_tokens=2)
    truncated_ib_reverse = await ib.atruncate(max_tokens=2, reverse=True)
    # Images are not truncatable
    assert truncated_ib == ib
    assert truncated_ib_reverse == ib


@pytest.mark.asyncio
async def test_image_block_amerge(png_1px_b64: bytes):
    ib1 = ImageBlock(image=png_1px_b64)
    ib2 = ImageBlock(image=png_1px_b64)
    merged = await ImageBlock.amerge([ib1, ib2], chunk_size=1000)

    # Images are not mergeable
    assert merged == [ib1, ib2]


@pytest.mark.asyncio
async def test_audio_block_aestimate_tokens(mp3_bytes: bytes):
    ab = AudioBlock(audio=mp3_bytes)
    assert await ab.aestimate_tokens() == 32  # based on 1 token per 4 bytes


@pytest.mark.asyncio
async def test_audio_block_aestimate_tokens_no_ffmpeg(
    mp3_base64: bytes, mock_no_ffmpeg
):
    """TinyTag is able to read mp3 metadata without ffmpeg installed."""
    ab = AudioBlock(audio=mp3_base64)
    assert await ab.aestimate_tokens() == 32  # based on 1 token per 4 bytes


@pytest.mark.asyncio
async def test_audio_block_aestimate_tokens_tinytag_error(
    mp3_bytes: bytes, mock_tiny_tag_error, mock_ffmpeg
):
    """If tinytag fails to read duration, falls back to ffmpeg."""
    ab = AudioBlock(audio=mp3_bytes)
    assert await ab.aestimate_tokens() == 32


@pytest.mark.asyncio
async def test_audio_block_aestimate_tokens_ffmpeg_error(
    mp3_bytes: bytes, mock_tiny_tag_error, mock_ffmpeg_ffprobe_error
):
    """If ffmpeg fails to read duration, use static fallback estimation."""
    ab = AudioBlock(audio=mp3_bytes)
    assert await ab.aestimate_tokens() == 256  # Fallback


@pytest.mark.asyncio
async def test_audio_block_asplit(
    mp3_bytes: bytes, mp3_split1_base64, mp3_split2_base64, mock_ffmpeg
):
    ab = AudioBlock(audio=mp3_bytes)
    chunks = await ab.asplit(max_tokens=2)

    assert len(chunks) == 2
    assert chunks[0].audio == mp3_split1_base64
    assert chunks[1].audio == mp3_split2_base64


@pytest.mark.asyncio
async def test_audio_block_asplit_no_ffmpeg(
    mp3_bytes: bytes, mp3_base64: bytes, mock_no_ffmpeg
):
    ab = AudioBlock(audio=mp3_bytes)
    chunks = await ab.asplit(max_tokens=2)

    # No splitting occurs
    assert chunks == [ab]


@pytest.mark.asyncio
async def test_audio_block_asplit_ffmpeg_error(
    mp3_bytes: bytes, mp3_base64: bytes, mock_ffmpeg_segment_error
):
    ab = AudioBlock(audio=mp3_bytes)
    chunks = await ab.asplit(max_tokens=2)

    # If ffmpeg fails, no splitting occurs
    assert chunks == [ab]


@pytest.mark.asyncio
async def test_audio_block_atruncate(
    mp3_bytes: bytes, mp3_split1_base64: bytes, mp3_split2_base64: bytes, mock_ffmpeg
):
    ab = AudioBlock(audio=mp3_bytes)
    truncated_ab = await ab.atruncate(max_tokens=16)
    truncated_ab_reverse = await ab.atruncate(max_tokens=16, reverse=True)
    # Returns the first chunk from calling split with max_tokens = 16
    assert truncated_ab.audio == mp3_split1_base64
    assert truncated_ab_reverse.audio == mp3_split2_base64


@pytest.mark.asyncio
async def test_audio_block_atruncate_no_ffmpeg(
    mp3_bytes: bytes, mp3_base64: bytes, mock_no_ffmpeg
):
    ab = AudioBlock(audio=mp3_bytes)
    truncated_ab = await ab.atruncate(max_tokens=16)
    truncated_ab_reverse = await ab.atruncate(max_tokens=16, reverse=True)
    # If no ffmpeg, no truncation occurs
    assert await truncated_ab.aestimate_tokens() == 32
    assert await truncated_ab_reverse.aestimate_tokens() == 32
    assert truncated_ab == ab
    assert truncated_ab_reverse == ab


@pytest.mark.asyncio
async def test_audio_block_atruncate_ffmpeg_error(
    mp3_bytes: bytes, mp3_base64: bytes, mock_ffmpeg_trim_error
):
    ab = AudioBlock(audio=mp3_bytes)
    truncated_ab = await ab.atruncate(max_tokens=16)
    truncated_ab_reverse = await ab.atruncate(max_tokens=16, reverse=True)
    # If ffmpeg fails, no truncation occurs
    assert await truncated_ab.aestimate_tokens() == 32
    assert await truncated_ab_reverse.aestimate_tokens() == 32
    assert truncated_ab == ab
    assert truncated_ab_reverse == ab


@pytest.mark.asyncio
async def test_audio_block_can_concatenate(
    mp3_split1_bytes: bytes, mp3_split2_bytes: bytes, mock_ffmpeg
):
    ab1 = AudioBlock(audio=mp3_split1_bytes)
    ab2 = AudioBlock(audio=mp3_split2_bytes)
    assert await ab1.can_concatenate(ab2) is True


@pytest.mark.asyncio
async def test_audio_block_can_concatenate_false(
    mp3_split1_bytes: bytes, mp3_split2_sr_8000_bytes: bytes, mock_ffmpeg
):
    """
    Only support concatenation when files can be losslessly merged (no resampling).

    Presumably, this should prevent merging audio blocks from different sources except for in cases when
    both audio blocks are exactly the same in terms of sample rate, channels, etc.
    """
    ab1 = AudioBlock(audio=mp3_split1_bytes)
    ab2 = AudioBlock(audio=mp3_split2_sr_8000_bytes)
    assert await ab1.can_concatenate(ab2) is False


@pytest.mark.asyncio
async def test_audio_block_can_concatenate_no_ffmpeg(
    mp3_split1_bytes: bytes, mp3_split2_bytes: bytes, mock_no_ffmpeg
):
    ab1 = AudioBlock(audio=mp3_split1_bytes)
    ab2 = AudioBlock(audio=mp3_split2_bytes)
    assert await ab1.can_concatenate(ab2) is False


@pytest.mark.asyncio
async def test_audio_block_can_concatenate_ffmpeg_error(
    mp3_split1_bytes: bytes,
    mp3_split2_bytes: bytes,
    mock_tiny_tag_error,
    mock_ffmpeg_ffprobe_error,
):
    ab1 = AudioBlock(audio=mp3_split1_bytes)
    ab2 = AudioBlock(audio=mp3_split2_bytes)
    assert await ab1.can_concatenate(ab2) is False


@pytest.mark.asyncio
async def test_audio_block_amerge(
    mp3_split1_bytes: bytes, mp3_split2_bytes: bytes, mp3_base64: bytes, mock_ffmpeg
):
    ab1 = AudioBlock(audio=mp3_split1_bytes)
    ab2 = AudioBlock(audio=mp3_split2_bytes)
    merged_abs = await AudioBlock.amerge([ab1, ab2], chunk_size=1000)

    assert len(merged_abs) == 1
    assert merged_abs[0].audio == mp3_base64


@pytest.mark.asyncio
async def test_audio_block_amerge_cannot_concatenate(
    mp3_split1_bytes: bytes, mp3_split2_sr_8000_bytes: bytes, mock_ffmpeg
):
    ab1 = AudioBlock(audio=mp3_split1_bytes)
    ab2 = AudioBlock(audio=mp3_split2_sr_8000_bytes)
    merged_abs = await AudioBlock.amerge([ab1, ab2], chunk_size=1000)

    # Cannot concatenate due to different sample rates, so returns both blocks as is
    assert merged_abs == [ab1, ab2]


@pytest.mark.asyncio
async def test_audio_block_amerge_no_ffmpeg(
    mp3_split1_bytes: bytes, mp3_split2_bytes: bytes, mock_no_ffmpeg
):
    ab1 = AudioBlock(audio=mp3_split1_bytes)
    ab2 = AudioBlock(audio=mp3_split2_bytes)
    merged_abs = await AudioBlock.amerge([ab1, ab2], chunk_size=1000)

    # If no ffmpeg, no merging occurs
    assert merged_abs == [ab1, ab2]


@pytest.mark.asyncio
async def test_audio_block_amerge_ffmpeg_error(
    mp3_split1_bytes: bytes, mp3_split2_bytes: bytes, mock_ffmpeg_concat_error
):
    ab1 = AudioBlock(audio=mp3_split1_bytes)
    ab2 = AudioBlock(audio=mp3_split2_bytes)
    merged_abs = await AudioBlock.amerge([ab1, ab2], chunk_size=1000)

    # If ffmpeg fails, no merging occurs
    assert merged_abs == [ab1, ab2]


def test_video_block_resolve_video_bytes(mp4_bytes: bytes, mp4_base64: bytes):
    b = VideoBlock(video=mp4_bytes)

    vid = b.resolve_video()
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_bytes

    vid = b.resolve_video(as_base64=True)
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_base64


def test_video_block_resolve_video_buffer(mp4_bytes: bytes):
    buffer = BytesIO(mp4_bytes)
    b = VideoBlock(video=buffer)

    vid = b.resolve_video()
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_bytes


def test_video_block_resolve_video_path(
    tmp_path: Path, mp4_bytes: bytes, mp4_base64: bytes
):
    mp4_path = tmp_path / "test.mp4"
    mp4_path.write_bytes(mp4_bytes)

    b = VideoBlock(path=mp4_path)
    vid = b.resolve_video()
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_bytes

    vid = b.resolve_video(as_base64=True)
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_base64


def test_video_block_resolve_video_url(mp4_bytes: bytes, mp4_base64: bytes):
    with mock.patch("llama_index.core.utils.requests") as mocked_req:
        url_str = "http://example.com/video.mp4"
        mocked_req.get.return_value = mock.MagicMock(content=mp4_bytes)
        b = VideoBlock(url=AnyUrl(url=url_str))
        vid = b.resolve_video()
        assert isinstance(vid, BytesIO)
        assert vid.read() == mp4_bytes

        vid = b.resolve_video(as_base64=True)
        assert isinstance(vid, BytesIO)
        assert vid.read() == mp4_base64


def test_video_block_resolve_video_data_url_base64(mp4_bytes: bytes, mp4_base64: bytes):
    # Test data URL with base64 encoding
    data_url = f"data:video/mp4;base64,{mp4_base64.decode('utf-8')}"
    b = VideoBlock(url=AnyUrl(url=data_url))

    vid = b.resolve_video()
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_bytes

    vid = b.resolve_video(as_base64=True)
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_base64


def test_video_block_resolve_error():
    b = VideoBlock()
    with pytest.raises(ValueError, match="No valid source provided"):
        b.resolve_video()


def test_video_block_store_as_anyurl():
    url_str = "http://example.com/video.mp4"
    b = VideoBlock(url=url_str)
    assert isinstance(b.url, AnyUrl)
    assert str(b.url) == url_str


def test_video_block_store_as_base64(mp4_bytes: bytes, mp4_base64: bytes):
    # Store regular bytes
    assert VideoBlock(video=mp4_bytes).video == mp4_base64
    # Store already encoded data
    assert VideoBlock(video=mp4_base64).video == mp4_base64


@pytest.mark.asyncio
async def test_video_block_aestimate_tokens(mp4_base64: bytes):
    vb = VideoBlock(video=mp4_base64)
    assert (
        await vb.aestimate_tokens() == 3 * 263
    )  # 263 tokens per second (rounded up to 3 seconds)


@pytest.mark.asyncio
async def test_video_block_aestimate_tokens_no_ffmpeg(
    mp4_base64: bytes, mock_no_ffmpeg
):
    """TinyTag fails for most video types, including this mp4 type."""
    vb = VideoBlock(video=mp4_base64)
    assert await vb.aestimate_tokens() == 256 * 8  # Fallback


@pytest.mark.asyncio
async def test_video_block_aestimate_tokens_ffmpeg_error(
    mp4_bytes: bytes, mock_ffmpeg_ffprobe_error
):
    """If ffmpeg fails to read duration, use static fallback estimation."""
    vb = VideoBlock(video=mp4_bytes)
    assert await vb.aestimate_tokens() == 256 * 8  # Fallback


@pytest.mark.asyncio
async def test_video_block_asplit(
    mp4_bytes: bytes, mp4_split1_base64: bytes, mp4_split2_base64: bytes, mock_ffmpeg
):
    vb = VideoBlock(video=mp4_bytes)
    chunks = await vb.asplit(max_tokens=500)

    assert len(chunks) == 2
    assert chunks[0].video == mp4_split1_base64
    assert chunks[1].video == mp4_split2_base64


@pytest.mark.asyncio
async def test_video_block_asplit_no_ffmpeg(
    mp4_bytes: bytes, mp4_base64: bytes, mock_no_ffmpeg
):
    vb = VideoBlock(video=mp4_bytes)
    chunks = await vb.asplit(max_tokens=500)

    # If no ffmpeg, no splitting occurs
    assert chunks == [vb]


@pytest.mark.asyncio
async def test_video_block_asplit_ffmpeg_error(
    mp4_bytes: bytes, mp4_base64: bytes, mock_ffmpeg_segment_error
):
    vb = VideoBlock(video=mp4_bytes)
    chunks = await vb.asplit(max_tokens=500)

    # If ffmpeg fails, no splitting occurs
    assert chunks == [vb]


@pytest.mark.asyncio
async def test_video_block_atruncate(
    mp4_bytes: bytes, mp4_split1_base64: bytes, mp4_split2_base64: bytes, mock_ffmpeg
):
    vb = VideoBlock(video=mp4_bytes)
    truncated_vb = await vb.atruncate(max_tokens=500)
    truncated_vb_reverse = await vb.atruncate(max_tokens=500, reverse=True)
    assert truncated_vb.video == mp4_split1_base64
    assert truncated_vb_reverse.video == mp4_split2_base64


@pytest.mark.asyncio
async def test_video_block_atruncate_no_ffmpeg(
    mp4_bytes: bytes, mp4_base64: bytes, mock_no_ffmpeg
):
    vb = VideoBlock(video=mp4_bytes)
    truncated_vb = await vb.atruncate(max_tokens=500)
    truncated_vb_reverse = await vb.atruncate(max_tokens=500, reverse=True)
    # If no ffmpeg, no truncation occurs
    assert truncated_vb == vb
    assert truncated_vb_reverse == vb


@pytest.mark.asyncio
async def test_video_block_atruncate_ffmpeg_error(
    mp4_bytes: bytes, mp4_base64: bytes, mock_ffmpeg_trim_error
):
    vb = VideoBlock(video=mp4_bytes)
    truncated_vb = await vb.atruncate(max_tokens=500)
    truncated_vb_reverse = await vb.atruncate(max_tokens=500, reverse=True)
    # If ffmpeg fails, no truncation occurs
    assert truncated_vb == vb
    assert truncated_vb_reverse == vb


@pytest.mark.asyncio
async def test_video_block_can_concatenate(
    mp4_split1_bytes: bytes, mp4_split2_bytes: bytes, mock_ffmpeg
):
    vb1 = VideoBlock(video=mp4_split1_bytes)
    vb2 = VideoBlock(video=mp4_split2_bytes)
    assert await vb1.can_concatenate(vb2) is True


@pytest.mark.asyncio
async def test_video_block_can_concatenate_false(
    mp4_split1_bytes: bytes, mp4_split2_width_320_bytes: bytes, mock_ffmpeg
):
    """
    Only support concatenation when files can be losslessly merged (no resampling).

    Presumably, this should prevent merging video blocks from different sources except for in case when
    both video blocks are exactly the same in terms of resolution, pixel format, etc.
    """
    vb1 = VideoBlock(video=mp4_split1_bytes)
    vb2 = VideoBlock(video=mp4_split2_width_320_bytes)
    assert await vb1.can_concatenate(vb2) is False


@pytest.mark.asyncio
async def test_video_block_can_concatenate_no_ffmpeg(
    mp4_split1_bytes: bytes, mp4_split2_bytes: bytes, mock_no_ffmpeg
):
    vb1 = VideoBlock(video=mp4_split1_bytes)
    vb2 = VideoBlock(video=mp4_split2_bytes)
    assert await vb1.can_concatenate(vb2) is False


@pytest.mark.asyncio
async def test_video_block_can_concatenate_ffmpeg_error(
    mp4_split1_bytes: bytes, mp4_split2_bytes: bytes, mock_ffmpeg_ffprobe_error
):
    vb1 = VideoBlock(video=mp4_split1_bytes)
    vb2 = VideoBlock(video=mp4_split2_bytes)
    assert await vb1.can_concatenate(vb2) is False


@pytest.mark.asyncio
async def test_video_block_amerge(
    mp4_split1_bytes: bytes, mp4_split2_bytes: bytes, mp4_base64: bytes, mock_ffmpeg
):
    vb1 = VideoBlock(video=mp4_split1_bytes)
    vb2 = VideoBlock(video=mp4_split2_bytes)
    merged_vbs = await VideoBlock.amerge([vb1, vb2], chunk_size=2000)

    assert len(merged_vbs) == 1
    assert merged_vbs[0].video == mp4_base64


@pytest.mark.asyncio
async def test_video_block_amerge_cannot_concatenate(
    mp4_split1_bytes: bytes, mp4_split2_width_320_bytes: bytes, mock_ffmpeg
):
    vb1 = VideoBlock(video=mp4_split1_bytes)
    vb2 = VideoBlock(video=mp4_split2_width_320_bytes)
    merged_vbs = await VideoBlock.amerge([vb1, vb2], chunk_size=2000)

    # Cannot concatenate due to different resolutions, so returns both blocks as is
    assert merged_vbs == [vb1, vb2]


@pytest.mark.asyncio
async def test_video_block_amerge_no_ffmpeg(
    mp4_split1_bytes: bytes, mp4_split2_bytes: bytes, mock_no_ffmpeg
):
    vb1 = VideoBlock(video=mp4_split1_bytes)
    vb2 = VideoBlock(video=mp4_split2_bytes)
    merged_vbs = await VideoBlock.amerge([vb1, vb2], chunk_size=2000)

    # If no ffmpeg, no merging occurs
    assert merged_vbs == [vb1, vb2]


@pytest.mark.asyncio
async def test_video_block_amerge_ffmpeg_error(
    mp4_split1_bytes: bytes, mp4_split2_bytes: bytes, mock_ffmpeg_concat_error
):
    vb1 = VideoBlock(video=mp4_split1_bytes)
    vb2 = VideoBlock(video=mp4_split2_bytes)
    merged_vbs = await VideoBlock.amerge([vb1, vb2], chunk_size=2000)

    # If ffmpeg fails, no merging occurs
    assert merged_vbs == [vb1, vb2]


def test_document_block_from_bytes(mock_pdf_bytes: bytes, pdf_base64: bytes):
    document = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    assert document.title == "input_document"
    assert document.document_mimetype == "application/pdf"
    assert pdf_base64 == document.data


def test_document_block_from_b64(pdf_base64: bytes):
    document = DocumentBlock(data=pdf_base64)
    assert document.title == "input_document"
    assert pdf_base64 == document.data


def test_document_block_from_path(tmp_path: Path, pdf_url: str):
    pdf_path = tmp_path / "test.pdf"
    pdf_content = httpx.get(pdf_url).content
    pdf_path.write_bytes(pdf_content)
    document = DocumentBlock(path=pdf_path.__str__())
    file_buffer = document.resolve_document()
    assert isinstance(file_buffer, BytesIO)
    file_bytes = file_buffer.read()
    document._guess_mimetype()
    assert document.document_mimetype == "application/pdf"
    fm = document.guess_format()
    assert fm == "pdf"
    b64_string = document._get_b64_string(file_buffer)
    try:
        base64.b64decode(b64_string, validate=True)
        string_base64_encoded = True
    except Exception:
        string_base64_encoded = False
    assert string_base64_encoded
    b64_bytes = document._get_b64_bytes(file_buffer)
    try:
        base64.b64decode(b64_bytes, validate=True)
        bytes_base64_encoded = True
    except Exception:
        bytes_base64_encoded = False
    assert bytes_base64_encoded
    assert document.title == "input_document"


def test_document_block_from_url(pdf_url: str):
    document = DocumentBlock(url=pdf_url, title="dummy_pdf")
    file_buffer = document.resolve_document()
    assert isinstance(file_buffer, BytesIO)
    file_bytes = file_buffer.read()
    document._guess_mimetype()
    assert document.document_mimetype == "application/pdf"
    fm = document.guess_format()
    assert fm == "pdf"
    b64_string = document._get_b64_string(file_buffer)
    try:
        base64.b64decode(b64_string, validate=True)
        string_base64_encoded = True
    except Exception as e:
        string_base64_encoded = False
    assert string_base64_encoded
    b64_bytes = document._get_b64_bytes(file_buffer)
    try:
        base64.b64decode(b64_bytes, validate=True)
        bytes_base64_encoded = True
    except Exception:
        bytes_base64_encoded = False
    assert bytes_base64_encoded
    assert document.title == "dummy_pdf"


def test_document_block_empty_bytes(empty_bytes: bytes, png_1px: bytes):
    errors = []
    try:
        DocumentBlock(data=empty_bytes).resolve_document()
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        AudioBlock(audio=empty_bytes).resolve_audio()
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        ImageBlock(image=empty_bytes).resolve_image()
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        ImageBlock(image=png_1px).resolve_image()
        errors.append(0)
    except ValueError:
        errors.append(1)
    assert sum(errors) == 3


@pytest.mark.asyncio
async def test_document_block_aestimate_tokens(mock_pdf_bytes: bytes):
    document = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    # Fallback: we currently don't estimate tokens for documents since it's too complicated to handle
    # all the different document types. Essentially kicking the can here.
    assert await document.aestimate_tokens() == 512


@pytest.mark.asyncio
async def test_document_block_asplit(mock_pdf_bytes: bytes):
    document = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    chunks = await document.asplit(max_tokens=100)
    # We dont split documents currently
    assert chunks == [document]


@pytest.mark.asyncio
async def test_document_block_atruncate(mock_pdf_bytes: bytes):
    document = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    truncated_document = await document.atruncate(max_tokens=100)
    truncated_document_reverse = await truncated_document.atruncate(
        max_tokens=100, reverse=True
    )
    # We dont truncate documents currently
    assert truncated_document == document
    assert truncated_document_reverse == document


@pytest.mark.asyncio
async def test_document_block_amerge(mock_pdf_bytes: bytes):
    document1 = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    document2 = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    merged = await DocumentBlock.amerge([document1, document2], chunk_size=1000)
    # We dont merge documents currently
    assert merged == [document1, document2]


def test_cache_control() -> None:
    cp = CachePoint(cache_control=CacheControl(type="ephemeral"))
    assert isinstance(cp.model_dump()["cache_control"], dict)
    assert cp.model_dump()["cache_control"]["type"] == "ephemeral"
    with pytest.raises(ValidationError):
        CachePoint.model_validate({"cache_control": "default"})


@pytest.mark.asyncio
async def test_cache_control_aestimate_tokens():
    cp = CachePoint(cache_control=CacheControl(type="ephemeral"))
    # No content length for ephemeral cache control
    assert await cp.aestimate_tokens() == 0


@pytest.mark.asyncio
async def test_cache_control_asplit():
    cp = CachePoint(cache_control=CacheControl(type="ephemeral"))
    chunks = await cp.asplit(max_tokens=10)
    # Cache control points are not splittable
    assert chunks == [cp]


@pytest.mark.asyncio
async def test_cache_control_atruncate():
    cp = CachePoint(cache_control=CacheControl(type="ephemeral"))
    truncated_cp = await cp.atruncate(max_tokens=10)
    truncated_cp_reverse = await cp.atruncate(max_tokens=10, reverse=True)
    # Cache control points are not truncatable
    assert truncated_cp == cp
    assert truncated_cp_reverse == cp


@pytest.mark.asyncio
async def test_cache_control_amerge():
    cp1 = CachePoint(cache_control=CacheControl(type="ephemeral"))
    cp2 = CachePoint(cache_control=CacheControl(type="ephemeral"))
    merged = await CachePoint.amerge([cp1, cp2], chunk_size=100)
    # Cache control points are not mergeable
    assert merged == [cp1, cp2]


def test_thinking_block():
    block = ThinkingBlock()
    assert block.block_type == "thinking"
    assert block.additional_information == {}
    assert block.content is None
    assert block.num_tokens is None
    block = ThinkingBlock(
        content="hello world",
        num_tokens=100,
        additional_information={"total_thinking_tokens": 1000},
    )
    assert block.block_type == "thinking"
    assert block.additional_information == {"total_thinking_tokens": 1000}
    assert block.content == "hello world"
    assert block.num_tokens == 100


@pytest.mark.asyncio
async def test_thinking_block_aestimate_tokens():
    block1 = ThinkingBlock(content="Some Content", num_tokens=150)
    block2 = ThinkingBlock(content="Some Content")

    assert await block1.aestimate_tokens() == block1.num_tokens
    assert (
        await block2.aestimate_tokens()
        == await TextBlock(text=block2.content).aestimate_tokens()
    )


@pytest.mark.asyncio
async def test_thinking_block_asplit():
    block = ThinkingBlock(content="This is a test of the ThinkingBlock split method.")
    chunks = await block.asplit(max_tokens=5)

    # Thinking blocks are not splittable
    assert chunks == [block]


@pytest.mark.asyncio
async def test_thinking_block_atruncate():
    block = ThinkingBlock(
        content="This is a test of the ThinkingBlock truncate method."
    )
    truncated_block = await block.atruncate(max_tokens=5)
    truncated_block_reverse = await block.atruncate(max_tokens=5, reverse=True)
    # Thinking blocks are not truncatable
    assert truncated_block == block
    assert truncated_block_reverse == block


@pytest.mark.asyncio
async def test_thinking_block_amerge():
    block1 = ThinkingBlock(content="This is the first ThinkingBlock.")
    block2 = ThinkingBlock(content="This is the second ThinkingBlock.")
    merged = await ThinkingBlock.amerge([block1, block2], chunk_size=100)

    # Thinking blocks are not mergeable
    assert merged == [block1, block2]


def test_tool_call_block():
    default_block = ToolCallBlock(tool_name="hello_world")
    assert default_block.block_type == "tool_call"
    assert default_block.tool_call_id is None
    assert default_block.tool_name == "hello_world"
    assert default_block.tool_kwargs == {}
    custom_block = ToolCallBlock(
        tool_name="hello_world",
        tool_call_id="1",
        tool_kwargs={"test": 1},
    )
    assert custom_block.tool_call_id == "1"
    assert custom_block.tool_name == "hello_world"
    assert custom_block.tool_kwargs == {"test": 1}


@pytest.mark.asyncio
async def test_tool_call_block_aestimate_tokens():
    block = ToolCallBlock(
        tool_name="example_tool", tool_kwargs={"param1": "value1", "param2": 42}
    )
    assert (
        await block.aestimate_tokens()
        == await TextBlock(text=block.model_dump_json()).aestimate_tokens()
    )


@pytest.mark.asyncio
async def test_tool_call_block_asplit():
    block = ToolCallBlock(
        tool_name="example_tool", tool_kwargs={"param1": "value1", "param2": 42}
    )

    # ToolCallBlocks are not splittable
    assert await block.asplit(max_tokens=5) == [block]


@pytest.mark.asyncio
async def test_tool_call_block_atruncate():
    block = ToolCallBlock(
        tool_name="example_tool", tool_kwargs={"param1": "value1", "param2": 42}
    )
    truncated_block = await block.atruncate(max_tokens=5)
    truncated_block_reverse = await block.atruncate(max_tokens=5, reverse=True)
    # ToolCallBlocks are not truncatable
    assert truncated_block == block
    assert truncated_block_reverse == block


@pytest.mark.asyncio
async def test_tool_call_block_amerge():
    block1 = ToolCallBlock(tool_name="example_tool_1", tool_kwargs={"param": "value1"})
    block2 = ToolCallBlock(tool_name="example_tool_2", tool_kwargs={"param": "value2"})
    merged = await ToolCallBlock.amerge([block1, block2], chunk_size=100)

    # ToolCallBlocks are not mergeable
    assert merged == [block1, block2]
