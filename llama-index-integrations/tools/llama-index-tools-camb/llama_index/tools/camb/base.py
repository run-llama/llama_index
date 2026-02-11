"""CAMB AI tool spec for LlamaIndex.

Provides 9 audio/speech tools powered by CAMB AI:
- Text-to-Speech (TTS)
- Translation
- Transcription
- Translated TTS
- Voice Cloning
- Voice Listing
- Voice Creation from Description
- Text-to-Sound generation
- Audio Separation
"""

from __future__ import annotations

import base64
import json
import os
import struct
import tempfile
import time
from typing import Any, Dict, List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class CambToolSpec(BaseToolSpec):
    """CAMB AI tool spec for LlamaIndex.

    CAMB AI provides multilingual audio and localization services including
    text-to-speech, translation, transcription, voice cloning, text-to-sound
    generation, and audio separation across 140+ languages.

    Args:
        api_key: CAMB AI API key. Falls back to CAMB_API_KEY env var.
        base_url: Optional custom base URL for CAMB AI API.
        timeout: Request timeout in seconds.
        max_poll_attempts: Maximum number of polling attempts for async tasks.
        poll_interval: Seconds between polling attempts.
    """

    spec_functions = [
        "text_to_speech",
        "translate",
        "transcribe",
        "translated_tts",
        "clone_voice",
        "list_voices",
        "create_voice_from_description",
        "text_to_sound",
        "separate_audio",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_poll_attempts: int = 60,
        poll_interval: float = 2.0,
    ) -> None:
        from camb.client import CambAI

        self.api_key = api_key or os.environ.get("CAMB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "CAMB AI API key is required. "
                "Set via 'api_key' parameter or CAMB_API_KEY environment variable."
            )
        self.base_url = base_url
        self.timeout = timeout
        self.max_poll_attempts = max_poll_attempts
        self.poll_interval = poll_interval

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = CambAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _poll(self, get_status_fn: Any, task_id: str, *, run_id: Any = None) -> Any:
        for _ in range(self.max_poll_attempts):
            status = get_status_fn(task_id, run_id=run_id)
            if hasattr(status, "status"):
                val = status.status
                if val in ("completed", "SUCCESS"):
                    return status
                if val in ("failed", "FAILED", "error"):
                    raise RuntimeError(
                        f"Task failed: {getattr(status, 'error', 'Unknown error')}"
                    )
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Task {task_id} did not complete within "
            f"{self.max_poll_attempts * self.poll_interval}s"
        )

    @staticmethod
    def _detect_audio_format(data: bytes, content_type: str = "") -> str:
        if data.startswith(b"RIFF"):
            return "wav"
        if data.startswith((b"\xff\xfb", b"\xff\xfa", b"ID3")):
            return "mp3"
        if data.startswith(b"fLaC"):
            return "flac"
        if data.startswith(b"OggS"):
            return "ogg"
        ct = content_type.lower()
        for key, fmt in [
            ("wav", "wav"), ("wave", "wav"), ("mpeg", "mp3"),
            ("mp3", "mp3"), ("flac", "flac"), ("ogg", "ogg"),
        ]:
            if key in ct:
                return fmt
        return "pcm"

    @staticmethod
    def _add_wav_header(pcm_data: bytes) -> bytes:
        sr, ch, bps = 24000, 1, 16
        byte_rate = sr * ch * bps // 8
        block_align = ch * bps // 8
        data_size = len(pcm_data)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE", b"fmt ", 16, 1,
            ch, sr, byte_rate, block_align, bps, b"data", data_size,
        )
        return header + pcm_data

    @staticmethod
    def _save_audio(data: bytes, suffix: str = ".wav") -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            return f.name

    @staticmethod
    def _gender_str(g: int) -> str:
        return {0: "not_specified", 1: "male", 2: "female", 9: "not_applicable"}.get(
            g, "unknown"
        )

    # ------------------------------------------------------------------
    # 1. Text-to-Speech
    # ------------------------------------------------------------------

    def text_to_speech(
        self,
        text: str,
        language: str = "en-us",
        voice_id: int = 147320,
        speech_model: str = "mars-flash",
        speed: float = 1.0,
        user_instructions: Optional[str] = None,
    ) -> str:
        """Convert text to speech using CAMB AI.

        Supports 140+ languages and multiple voice models. Returns a file path
        to the generated audio. Available models: 'mars-flash' (fast),
        'mars-pro' (high quality), 'mars-instruct' (follows instructions).

        Args:
            text: Text to convert to speech (3-3000 characters).
            language: BCP-47 language code (e.g., 'en-us', 'es-es', 'fr-fr').
            voice_id: Voice ID. Use list_voices to find available voices.
            speech_model: 'mars-flash', 'mars-pro', or 'mars-instruct'.
            speed: Speech speed multiplier (0.5-2.0).
            user_instructions: Instructions for mars-instruct model.

        Returns:
            File path to the generated WAV audio file.
        """
        from camb import StreamTtsOutputConfiguration, StreamTtsVoiceSettings

        kwargs: Dict[str, Any] = {
            "text": text,
            "language": language,
            "voice_id": voice_id,
            "speech_model": speech_model,
            "output_configuration": StreamTtsOutputConfiguration(format="wav"),
            "voice_settings": StreamTtsVoiceSettings(speed=speed),
        }
        if user_instructions and speech_model == "mars-instruct":
            kwargs["user_instructions"] = user_instructions

        chunks: list[bytes] = []
        for chunk in self._client.text_to_speech.tts(**kwargs):
            chunks.append(chunk)
        return self._save_audio(b"".join(chunks), ".wav")

    # ------------------------------------------------------------------
    # 2. Translation
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        source_language: int,
        target_language: int,
        formality: Optional[int] = None,
    ) -> str:
        """Translate text between 140+ languages using CAMB AI.

        Args:
            text: Text to translate.
            source_language: Source language code (integer). 1=English, 2=Spanish,
                3=French, 4=German, 5=Italian, 6=Portuguese, 7=Dutch, 8=Russian,
                9=Japanese, 10=Korean, 11=Chinese.
            target_language: Target language code (integer).
            formality: Optional formality level: 1=formal, 2=informal.

        Returns:
            The translated text string.
        """
        from camb.core.api_error import ApiError

        kwargs: Dict[str, Any] = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
        }
        if formality:
            kwargs["formality"] = formality

        try:
            result = self._client.translation.translation_stream(**kwargs)
            return self._extract_translation(result)
        except ApiError as e:
            if e.status_code == 200 and e.body:
                return str(e.body)
            raise

    @staticmethod
    def _extract_translation(result: Any) -> str:
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            parts = []
            for chunk in result:
                if hasattr(chunk, "text"):
                    parts.append(chunk.text)
                elif isinstance(chunk, str):
                    parts.append(chunk)
            return "".join(parts)
        if hasattr(result, "text"):
            return result.text
        return str(result)

    # ------------------------------------------------------------------
    # 3. Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        language: int,
        audio_url: Optional[str] = None,
        audio_file_path: Optional[str] = None,
    ) -> str:
        """Transcribe audio to text with speaker identification using CAMB AI.

        Supports audio URLs or local file paths. Returns JSON with full
        transcription text, timed segments, and speaker labels.

        Args:
            language: Language code (integer). 1=English, 2=Spanish, 3=French, etc.
            audio_url: URL of the audio file to transcribe.
            audio_file_path: Local file path to the audio file.

        Returns:
            JSON string with text, segments (start, end, text, speaker),
            and speakers list.
        """
        kwargs: Dict[str, Any] = {"language": language}

        if audio_url:
            import httpx

            resp = httpx.get(audio_url)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            with open(tmp_path, "rb") as f:
                kwargs["media_file"] = f
                result = self._client.transcription.create_transcription(**kwargs)
        elif audio_file_path:
            with open(audio_file_path, "rb") as f:
                kwargs["media_file"] = f
                result = self._client.transcription.create_transcription(**kwargs)
        else:
            return json.dumps({"error": "Provide either audio_url or audio_file_path"})

        task_id = result.task_id
        status = self._poll(
            self._client.transcription.get_transcription_task_status, task_id
        )
        transcription = self._client.transcription.get_transcription_result(
            status.run_id
        )

        out: Dict[str, Any] = {
            "text": getattr(transcription, "text", ""),
            "segments": [],
            "speakers": [],
        }
        if hasattr(transcription, "segments"):
            for seg in transcription.segments:
                out["segments"].append(
                    {
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "text": getattr(seg, "text", ""),
                        "speaker": getattr(seg, "speaker", None),
                    }
                )
        if hasattr(transcription, "speakers"):
            out["speakers"] = list(transcription.speakers)
        elif out["segments"]:
            out["speakers"] = list(
                {s["speaker"] for s in out["segments"] if s.get("speaker")}
            )
        return json.dumps(out, indent=2)

    # ------------------------------------------------------------------
    # 4. Translated TTS
    # ------------------------------------------------------------------

    def translated_tts(
        self,
        text: str,
        source_language: int,
        target_language: int,
        voice_id: int = 147320,
        formality: Optional[int] = None,
    ) -> str:
        """Translate text and convert to speech in one step using CAMB AI.

        Translates text to the target language and generates speech audio.

        Args:
            text: Text to translate and speak.
            source_language: Source language code (integer).
            target_language: Target language code (integer).
            voice_id: Voice ID for TTS output.
            formality: Optional formality: 1=formal, 2=informal.

        Returns:
            File path to the audio file of translated speech.
        """
        import httpx

        kwargs: Dict[str, Any] = {
            "text": text,
            "voice_id": voice_id,
            "source_language": source_language,
            "target_language": target_language,
        }
        if formality:
            kwargs["formality"] = formality

        result = self._client.translated_tts.create_translated_tts(**kwargs)
        status = self._poll(
            self._client.translated_tts.get_translated_tts_task_status,
            result.task_id,
        )

        # Download audio via httpx (SDK workaround)
        audio_data = b""
        audio_fmt = "pcm"
        run_id = getattr(status, "run_id", None)
        if run_id:
            base = getattr(self._client, "_client_wrapper", None)
            if base and hasattr(base, "base_url"):
                url = f"{base.base_url}/tts-result/{run_id}"
            else:
                url = f"https://client.camb.ai/apis/tts-result/{run_id}"
            with httpx.Client() as http:
                resp = http.get(url, headers={"x-api-key": self.api_key})
                if resp.status_code == 200:
                    audio_data = resp.content
                    audio_fmt = self._detect_audio_format(
                        audio_data, resp.headers.get("content-type", "")
                    )

        if not audio_data:
            message = getattr(status, "message", None)
            if message:
                msg_url = None
                if isinstance(message, dict):
                    msg_url = (
                        message.get("output_url")
                        or message.get("audio_url")
                        or message.get("url")
                    )
                elif isinstance(message, str) and message.startswith("http"):
                    msg_url = message
                if msg_url:
                    with httpx.Client() as http:
                        resp = http.get(msg_url)
                        audio_data = resp.content
                        audio_fmt = self._detect_audio_format(
                            audio_data, resp.headers.get("content-type", "")
                        )

        if audio_fmt == "pcm" and audio_data:
            audio_data = self._add_wav_header(audio_data)
            audio_fmt = "wav"

        ext = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg"}.get(
            audio_fmt, ".wav"
        )
        return self._save_audio(audio_data, ext)

    # ------------------------------------------------------------------
    # 5. Voice Cloning
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        voice_name: str,
        audio_file_path: str,
        gender: int,
        description: Optional[str] = None,
        age: Optional[int] = None,
        language: Optional[int] = None,
    ) -> str:
        """Clone a voice from an audio sample using CAMB AI.

        Creates a custom voice from a 2+ second audio sample that can be used
        with text_to_speech and translated_tts.

        Args:
            voice_name: Name for the new cloned voice.
            audio_file_path: Path to audio file (minimum 2 seconds).
            gender: Gender: 1=Male, 2=Female, 0=Not Specified, 9=Not Applicable.
            description: Optional description of the voice.
            age: Optional age of the voice.
            language: Optional language code for the voice.

        Returns:
            JSON string with voice_id, voice_name, and status.
        """
        with open(audio_file_path, "rb") as f:
            kw: Dict[str, Any] = {
                "voice_name": voice_name,
                "gender": gender,
                "file": f,
            }
            if description:
                kw["description"] = description
            if age:
                kw["age"] = age
            if language:
                kw["language"] = language
            result = self._client.voice_cloning.create_custom_voice(**kw)

        out = {
            "voice_id": getattr(result, "voice_id", getattr(result, "id", None)),
            "voice_name": voice_name,
            "status": "created",
        }
        if hasattr(result, "message"):
            out["message"] = result.message
        return json.dumps(out, indent=2)

    # ------------------------------------------------------------------
    # 6. Voice Listing
    # ------------------------------------------------------------------

    def list_voices(self) -> str:
        """List all available voices from CAMB AI.

        Returns voice IDs, names, genders, ages, and languages. Use the
        voice ID with text_to_speech or translated_tts.

        Returns:
            JSON array of voice objects with id, name, gender, age, language.
        """
        voices = self._client.voice_cloning.list_voices()
        out = []
        for v in voices:
            if isinstance(v, dict):
                out.append(
                    {
                        "id": v.get("id"),
                        "name": v.get("voice_name", v.get("name", "Unknown")),
                        "gender": self._gender_str(v.get("gender", 0)),
                        "age": v.get("age"),
                        "language": v.get("language"),
                    }
                )
            else:
                out.append(
                    {
                        "id": getattr(v, "id", None),
                        "name": getattr(
                            v, "voice_name", getattr(v, "name", "Unknown")
                        ),
                        "gender": self._gender_str(getattr(v, "gender", 0)),
                        "age": getattr(v, "age", None),
                        "language": getattr(v, "language", None),
                    }
                )
        return json.dumps(out, indent=2)

    # ------------------------------------------------------------------
    # 7. Voice from Description
    # ------------------------------------------------------------------

    def create_voice_from_description(
        self,
        text: str,
        voice_description: str,
    ) -> str:
        """Generate a synthetic voice from a detailed text description using CAMB AI.

        Provide sample text and a description of the desired voice (minimum 100
        characters / 18+ words). Include accent, tone, age, gender, speaking
        style, etc. Returns preview audio URLs.

        Args:
            text: Sample text the generated voice will speak.
            voice_description: Detailed description of the desired voice (min 100 chars).

        Returns:
            JSON string with preview audio URLs for the generated voice.
        """
        result = self._client.text_to_voice.create_text_to_voice(
            text=text, voice_description=voice_description,
        )
        task_id = result.task_id
        status = self._poll(
            self._client.text_to_voice.get_text_to_voice_status, task_id
        )
        voice_result = self._client.text_to_voice.get_text_to_voice_result(
            status.run_id
        )

        out = {
            "previews": getattr(voice_result, "previews", []),
            "status": "completed",
        }
        return json.dumps(out, indent=2)

    # ------------------------------------------------------------------
    # 8. Text-to-Sound
    # ------------------------------------------------------------------

    def text_to_sound(
        self,
        prompt: str,
        duration: Optional[float] = None,
        audio_type: Optional[str] = None,
    ) -> str:
        """Generate sounds, music, or soundscapes from text descriptions using CAMB AI.

        Describe the audio you want and the tool generates it.

        Args:
            prompt: Description of the sound or music to generate.
            duration: Optional duration in seconds.
            audio_type: Optional type: 'music' or 'sound'.

        Returns:
            File path to the generated audio file.
        """
        kwargs: Dict[str, Any] = {"prompt": prompt}
        if duration:
            kwargs["duration"] = duration
        if audio_type:
            kwargs["audio_type"] = audio_type

        result = self._client.text_to_audio.create_text_to_audio(**kwargs)
        status = self._poll(
            self._client.text_to_audio.get_text_to_audio_status, result.task_id
        )

        chunks: list[bytes] = []
        for chunk in self._client.text_to_audio.get_text_to_audio_result(
            status.run_id
        ):
            chunks.append(chunk)
        return self._save_audio(b"".join(chunks), ".wav")

    # ------------------------------------------------------------------
    # 8. Audio Separation
    # ------------------------------------------------------------------

    def separate_audio(
        self,
        audio_url: Optional[str] = None,
        audio_file_path: Optional[str] = None,
    ) -> str:
        """Separate vocals/speech from background audio using CAMB AI.

        Isolate vocals from background music or noise.

        Args:
            audio_url: URL of the audio file to separate.
            audio_file_path: Local file path to the audio file.

        Returns:
            JSON string with 'vocals' and 'background' file paths or URLs.
        """
        kwargs: Dict[str, Any] = {}
        if audio_file_path:
            with open(audio_file_path, "rb") as f:
                kwargs["media_file"] = f
                result = self._client.audio_separation.create_audio_separation(**kwargs)
        else:
            result = self._client.audio_separation.create_audio_separation(**kwargs)

        status = self._poll(
            self._client.audio_separation.get_audio_separation_status,
            result.task_id,
        )
        sep = self._client.audio_separation.get_audio_separation_run_info(
            status.run_id
        )

        out: Dict[str, Any] = {
            "vocals": None,
            "background": None,
            "status": "completed",
        }
        for attr, key in [
            ("vocals_url", "vocals"),
            ("vocals", "vocals"),
            ("voice_url", "vocals"),
            ("background_url", "background"),
            ("background", "background"),
            ("instrumental_url", "background"),
        ]:
            val = getattr(sep, attr, None)
            if val and out[key] is None:
                if isinstance(val, bytes):
                    out[key] = self._save_audio(val, f"_{key}.wav")
                else:
                    out[key] = val
        return json.dumps(out, indent=2)
