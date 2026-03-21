"""Example usage of CAMB AI tool spec with LlamaIndex."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

import httpx

from llama_index.tools.camb import CambToolSpec

API_KEY = os.environ.get("CAMB_API_KEY")
if not API_KEY:
    raise RuntimeError("Set CAMB_API_KEY environment variable to run examples")

AUDIO_SAMPLE = os.environ.get("CAMB_AUDIO_SAMPLE")
if not AUDIO_SAMPLE:
    raise RuntimeError("Set CAMB_AUDIO_SAMPLE environment variable to a local audio file path")


def play(path: str):
    """Play an audio file with afplay (macOS)."""
    if sys.platform == "darwin":
        print(f"  Playing: {path}")
        subprocess.run(["afplay", path], check=False)
    else:
        print(f"  Audio file at: {path} (afplay not available on this platform)")


def play_url(url: str, label: str = ""):
    """Download and play an audio URL."""
    resp = httpx.get(url)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(resp.content)
        path = f.name
    if label:
        print(f"  {label}")
    play(path)


spec = CambToolSpec(api_key=API_KEY)


def test_tts():
    """1. Text-to-Speech: convert text to audio."""
    path = spec.text_to_speech("Hello from CAMB AI and LlamaIndex! This is a text to speech test.")
    print(f"  Audio saved to: {path}")
    assert path.endswith(".wav")
    play(path)


def test_translation():
    """2. Translation: translate text between languages."""
    result = spec.translate("Hello, how are you?", source_language=1, target_language=2)
    print(f"  Result: {result}")
    assert len(result) > 0


def test_voice_list():
    """3. Voice List: list available voices."""
    result = spec.list_voices()
    print(f"  Voices (first 200 chars): {result[:200]}")
    assert "id" in result


def test_transcription():
    """4. Transcription: transcribe audio from local file."""
    result = spec.transcribe(language=1, audio_file_path=AUDIO_SAMPLE)
    print(f"  Transcription (first 300 chars): {result[:300]}")
    assert "text" in result


def test_translated_tts():
    """5. Translated TTS: translate and speak in one step."""
    path = spec.translated_tts(text="Hello, how are you?", source_language=1, target_language=2)
    print(f"  Audio saved to: {path}")
    assert path.endswith((".wav", ".mp3", ".flac", ".ogg"))
    play(path)


def test_text_to_sound():
    """6. Text-to-Sound: generate audio from a description."""
    path = spec.text_to_sound("gentle rain on a rooftop", duration=5.0, audio_type="sound")
    print(f"  Audio saved to: {path}")
    assert path.endswith(".wav")
    play(path)


def test_voice_clone():
    """7. Voice Clone: clone a voice from an audio sample."""
    result = spec.clone_voice(voice_name="test_clone_llamaindex", audio_file_path=AUDIO_SAMPLE, gender=2)
    print(f"  Result: {result}")
    assert "voice_id" in result
    # Play the cloned voice via TTS
    data = json.loads(result)
    voice_id = data["voice_id"]
    print(f"  Speaking with cloned voice (id: {voice_id})...")
    path = spec.text_to_speech(
        "Hello! This is the cloned voice from the audio sample, generated with CAMB AI and LlamaIndex.",
        voice_id=voice_id,
    )
    play(path)


def test_audio_separation():
    """8. Audio Separation: separate vocals from background."""
    result = spec.separate_audio(audio_file_path=AUDIO_SAMPLE)
    print(f"  Result: {result}")
    assert "status" in result


def test_voice_from_description():
    """9. Voice from Description: generate a voice from text description."""
    result = spec.create_voice_from_description(
        text="Hello, this is a comprehensive test of the voice generation feature from CAMB AI. We are testing whether we can create a new synthetic voice from just a text description alone.",
        voice_description="A warm, friendly female voice with a slight British accent, aged around 30, professional tone suitable for narration and audiobooks, clear enunciation with a calm demeanor",
    )
    print(f"  Result (first 200 chars): {result[:200]}")
    assert "previews" in result
    data = json.loads(result)
    for i, url in enumerate(data.get("previews", [])):
        play_url(url, label=f"Preview {i + 1}")


if __name__ == "__main__":
    tests = [
        test_tts,
        test_translation,
        test_voice_list,
        test_transcription,
        test_translated_tts,
        test_text_to_sound,
        test_voice_clone,
        test_audio_separation,
        test_voice_from_description,
    ]
    for t in tests:
        print(f"\n--- {t.__doc__} ---")
        try:
            t()
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
