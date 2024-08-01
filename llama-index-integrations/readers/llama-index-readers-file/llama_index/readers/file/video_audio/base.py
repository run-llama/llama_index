"""Video audio parser.

Contains parsers for mp3, mp4 files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional, cast
import logging
from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class VideoAudioReader(BaseReader):
    """Video audio parser.

    Extract text from transcript of video/audio files.

    """

    def __init__(self, *args: Any, model_version: str = "base", **kwargs: Any) -> None:
        """Init parser."""
        super().__init__(*args, **kwargs)
        self._model_version = model_version

        try:
            import whisper
        except ImportError:
            raise ImportError(
                "Please install OpenAI whisper model "
                "'pip install git+https://github.com/openai/whisper.git' "
                "to use the model"
            )

        model = whisper.load_model(self._model_version)

        self.parser_config = {"model": model}

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        import whisper

        if file.name.endswith("mp4"):
            try:
                from pydub import AudioSegment
            except ImportError:
                raise ImportError("Please install pydub 'pip install pydub' ")
            if fs:
                with fs.open(file, "rb") as f:
                    video = AudioSegment.from_file(f, format="mp4")
            else:
                # open file
                video = AudioSegment.from_file(file, format="mp4")

            # Extract audio from video
            audio = video.split_to_mono()[0]

            file_str = str(file)[:-4] + ".mp3"
            # export file
            audio.export(file_str, format="mp3")

        model = cast(whisper.Whisper, self.parser_config["model"])
        result = model.transcribe(str(file))

        transcript = result["text"]

        return [Document(text=transcript, metadata=extra_info or {})]
