"""Video audio parser.

Contains parsers for mp3, mp4 files.

"""
from pathlib import Path
from typing import Any, Dict, cast

from gpt_index.readers.file.base_parser import BaseParser


class VideoAudioParser(BaseParser):
    """Video audio parser.

    Extract text from transcript of video/audio files.

    """

    def __init__(self, *args: Any, model_version: str = "base", **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._model_version = model_version

    def _init_parser(self) -> Dict:
        """Init parser."""
        try:
            import whisper
        except ImportError:
            raise ValueError(
                "Please install OpenAI whisper model "
                "'pip install git+https://github.com/openai/whisper.git' "
                "to use the model"
            )

        model = whisper.load_model(self._model_version)

        return {"model": model}

    def parse_file(self, file: Path, errors: str = "ignore") -> str:
        """Parse file."""
        import whisper

        if file.name.endswith("mp4"):
            try:
                from pydub import AudioSegment  # noqa: F401
            except ImportError:
                raise ValueError("Please install pydub 'pip install pydub' ")
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

        return transcript
