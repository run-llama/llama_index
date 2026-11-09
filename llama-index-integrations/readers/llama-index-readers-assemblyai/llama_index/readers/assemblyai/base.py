from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

import assemblyai


class TranscriptFormat(Enum):
    """Transcript format to use for the document reader."""

    TEXT = "text"
    """One document with the transcription text"""
    SENTENCES = "sentences"
    """Multiple documents, splits the transcription by each sentence"""
    PARAGRAPHS = "paragraphs"
    """Multiple documents, splits the transcription by each paragraph"""
    CHAPTERS = "chapters"
    """Multiple documents, one per chapter with summary and chapter metadata"""
    UTTERANCES = "utterances"
    """Multiple documents, splits the transcription by each speaker utterance"""
    SUBTITLES_SRT = "subtitles_srt"
    """One document with the transcript exported in SRT subtitles format"""
    SUBTITLES_VTT = "subtitles_vtt"
    """One document with the transcript exported in VTT subtitles format"""


class AssemblyAIAudioTranscriptReader(BaseReader):
    """
    Reader for AssemblyAI audio transcripts.

    It uses the AssemblyAI API to transcribe audio files (or load an existing
    transcript by ID) and loads the transcribed text into one or more Documents,
    depending on the specified format.

    To use, you should have the ``assemblyai`` python package installed, and the
    environment variable ``ASSEMBLYAI_API_KEY`` set with your API key.
    Alternatively, the API key can also be passed as an argument.

    Audio files can be specified via a URL or a local file path.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        *,
        transcript_id: Optional[str] = None,
        transcript_format: TranscriptFormat = TranscriptFormat.TEXT,
        config: Optional[assemblyai.TranscriptionConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the AssemblyAI AudioTranscriptReader.

        Args:
            file_path: A URL or a local file path.
            transcript_id: An ID of an existing AssemblyAI transcript.
            transcript_format: Transcript format to use.
                See class ``TranscriptFormat`` for more info.
            config: Transcription options and features. If ``None`` is given,
                the Transcriber's default configuration will be used.
            api_key: AssemblyAI API key.

        """
        if file_path is None and transcript_id is None:
            raise ValueError("Either file_path or transcript_id must be provided.")
        if file_path is not None and transcript_id is not None:
            raise ValueError(
                "Only one of file_path or transcript_id should be provided."
            )

        if api_key is not None:
            assemblyai.settings.api_key = api_key

        self.file_path = file_path
        self.transcript_id = transcript_id
        self.transcript_format = transcript_format
        self.config = config

        # Instantiating the Transcriber will raise a ValueError if no API key is set.
        self.transcriber = assemblyai.Transcriber(config=config)

    def _get_transcript(self, file_path: Optional[str] = None) -> Any:
        target_path = file_path or self.file_path
        if target_path is not None:
            transcript = self.transcriber.transcribe(target_path)
        elif self.transcript_id is not None:
            transcript = assemblyai.Transcript.get_by_id(self.transcript_id)
        else:
            raise ValueError("No file_path or transcript_id was specified.")

        if getattr(transcript, "error", None):
            raise ValueError(f"Could not transcribe file: {transcript.error}")

        return transcript

    def lazy_load_data(
        self,
        *args: Any,
        extra_info: Optional[Dict[str, Any]] = None,
        **load_kwargs: Any,
    ) -> Iterable[Document]:
        """
        Transcribes the audio file (or loads existing transcript) and lazily yields documents.

        Args:
            *args: Optional positional arguments. The first positional argument can
                be a file path or URL to override ``self.file_path``.
            extra_info: Optional dictionary of metadata to attach to documents.
            **load_kwargs: Additional keyword arguments. Accepts ``file_path`` or ``extra_info``.

        """
        file_path = str(args[0]) if args else load_kwargs.get("file_path")
        merged_extra_info: Dict[str, Any] = {}
        if extra_info:
            merged_extra_info.update(extra_info)
        if "extra_info" in load_kwargs and isinstance(load_kwargs["extra_info"], dict):
            merged_extra_info.update(load_kwargs["extra_info"])

        transcript = self._get_transcript(file_path=file_path)

        if self.transcript_format == TranscriptFormat.TEXT:
            text = transcript.text or ""
            metadata = dict(transcript.json_response or {})
            if merged_extra_info:
                metadata.update(merged_extra_info)
            yield Document(text=text, metadata=metadata)

        elif self.transcript_format == TranscriptFormat.SENTENCES:
            sentences = transcript.get_sentences() or []
            for s in sentences:
                metadata = (
                    s.model_dump(exclude={"text"})
                    if hasattr(s, "model_dump")
                    else s.dict(exclude={"text"})
                )
                if merged_extra_info:
                    metadata.update(merged_extra_info)
                yield Document(text=s.text or "", metadata=metadata)

        elif self.transcript_format == TranscriptFormat.PARAGRAPHS:
            paragraphs = transcript.get_paragraphs() or []
            for p in paragraphs:
                metadata = (
                    p.model_dump(exclude={"text"})
                    if hasattr(p, "model_dump")
                    else p.dict(exclude={"text"})
                )
                if merged_extra_info:
                    metadata.update(merged_extra_info)
                yield Document(text=p.text or "", metadata=metadata)

        elif self.transcript_format == TranscriptFormat.CHAPTERS:
            chapters = getattr(transcript, "chapters", None) or []
            for c in chapters:
                metadata = (
                    c.model_dump(exclude={"summary"})
                    if hasattr(c, "model_dump")
                    else c.dict(exclude={"summary"})
                )
                if merged_extra_info:
                    metadata.update(merged_extra_info)
                yield Document(text=c.summary or "", metadata=metadata)

        elif self.transcript_format == TranscriptFormat.UTTERANCES:
            utterances = getattr(transcript, "utterances", None) or []
            for u in utterances:
                metadata = (
                    u.model_dump(exclude={"text"})
                    if hasattr(u, "model_dump")
                    else u.dict(exclude={"text"})
                )
                if merged_extra_info:
                    metadata.update(merged_extra_info)
                yield Document(text=u.text or "", metadata=metadata)

        elif self.transcript_format == TranscriptFormat.SUBTITLES_SRT:
            text = transcript.export_subtitles_srt()
            metadata = dict(transcript.json_response or {})
            if merged_extra_info:
                metadata.update(merged_extra_info)
            yield Document(text=text, metadata=metadata)

        elif self.transcript_format == TranscriptFormat.SUBTITLES_VTT:
            text = transcript.export_subtitles_vtt()
            metadata = dict(transcript.json_response or {})
            if merged_extra_info:
                metadata.update(merged_extra_info)
            yield Document(text=text, metadata=metadata)

        else:
            raise ValueError(f"Unknown transcript format: {self.transcript_format}")

    def load_data(
        self,
        *args: Any,
        extra_info: Optional[Dict[str, Any]] = None,
        **load_kwargs: Any,
    ) -> List[Document]:
        """
        Transcribes the audio file and loads the transcript into documents.
        """
        return list(self.lazy_load_data(*args, extra_info=extra_info, **load_kwargs))
