"""Typecast text to speech tool spec."""

from typing import List, Optional
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class TypecastToolSpec(BaseToolSpec):
    """Typecast tool spec for text-to-speech synthesis with emotion control."""

    spec_functions = ["get_voices", "get_voice", "text_to_speech"]

    def __init__(self, api_key: str, host: Optional[str] = None) -> None:
        """
        Initialize with parameters.

        Args:
            api_key (str): Your Typecast API key
            host (Optional[str]): The base url of Typecast API (default: https://api.typecast.ai)

        """
        self.api_key = api_key
        self.host = host

    def get_voices(
        self,
        model: Optional[str] = None,
        gender: Optional[str] = None,
        age: Optional[str] = None,
        use_case: Optional[str] = None,
    ) -> List[dict]:
        """
        Get list of available voices from Typecast (V2 API).

        Args:
            model (Optional[str]): Filter by model name (e.g., "ssfm-v21", "ssfm-v30")
            gender (Optional[str]): Filter by gender ("male" or "female")
            age (Optional[str]): Filter by age group ("child", "teenager", "young_adult", "middle_age", "elder")
            use_case (Optional[str]): Filter by use case category (e.g., "Audiobook", "Game", "Podcast")

        Returns:
            List[dict]: List of available voices with their details including:
                - voice_id: Unique voice identifier
                - voice_name: Human-readable name
                - models: List of supported models with emotions
                - gender: Voice gender (optional)
                - age: Voice age group (optional)
                - use_cases: List of suitable use cases (optional)

        Raises:
            Exception: If API request fails

        """
        try:
            from typecast.client import Typecast
            from typecast.models import VoicesV2Filter

            # Create the client
            client = Typecast(host=self.host, api_key=self.api_key)

            # Build filter if any parameters provided
            filter_obj = None
            if any([model, gender, age, use_case]):
                filter_obj = VoicesV2Filter(
                    model=model,
                    gender=gender,
                    age=age,
                    use_cases=use_case,
                )

            # Get the voices using V2 API
            response = client.voices_v2(filter=filter_obj)

            # Return the dumped voice models as dict
            return [voice.model_dump() for voice in response]
        except Exception as e:
            raise Exception(f"Failed to get voices from Typecast API: {e!s}")

    def get_voice(self, voice_id: str) -> dict:
        """
        Get details of a specific voice from Typecast (V2 API).

        Args:
            voice_id (str): The voice ID to get details for (e.g., "tc_62a8975e695ad26f7fb514d1")

        Returns:
            dict: Voice details including:
                - voice_id: Unique voice identifier
                - voice_name: Human-readable name
                - models: List of supported models with emotions
                - gender: Voice gender (optional)
                - age: Voice age group (optional)
                - use_cases: List of suitable use cases (optional)

        Raises:
            Exception: If API request fails or voice not found

        """
        try:
            from typecast.client import Typecast
            from typecast.exceptions import NotFoundError, TypecastError

            # Create the client
            client = Typecast(host=self.host, api_key=self.api_key)

            # Get the voice using V2 API
            try:
                response = client.voice_v2(voice_id)
            except NotFoundError:
                raise Exception(f"Voice not found: {voice_id}")
            except TypecastError as e:
                raise Exception(f"Typecast API error: {e!s}")

            # Return the dumped voice model as dict
            return response.model_dump()
        except Exception as e:
            if "Voice not found" in str(e) or "Typecast API" in str(e):
                raise
            raise Exception(f"Failed to get voice from Typecast API: {e!s}")

    def text_to_speech(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        model: str = "ssfm-v21",
        language: Optional[str] = None,
        emotion_preset: Optional[str] = "normal",
        emotion_intensity: Optional[float] = 1.0,
        volume: Optional[int] = 100,
        audio_pitch: Optional[int] = 0,
        audio_tempo: Optional[float] = 1.0,
        audio_format: Optional[str] = "wav",
        seed: Optional[int] = None,
    ) -> str:
        """
        Convert text to speech using Typecast API.

        Args:
            text (str): The text to convert to speech
            voice_id (str): The voice ID to use (e.g., "tc_62a8975e695ad26f7fb514d1")
            output_path (str): Path to save the audio file
            model (str): Voice model name (default: "ssfm-v21")
            language (Optional[str]): Language code (ISO 639-3, e.g., "eng", "kor")
            emotion_preset (Optional[str]): Emotion preset (normal, happy, sad, angry)
            emotion_intensity (Optional[float]): Emotion intensity (0.0 to 2.0)
            volume (Optional[int]): Volume (0 to 200, default: 100)
            audio_pitch (Optional[int]): Audio pitch (-12 to 12, default: 0)
            audio_tempo (Optional[float]): Audio tempo (0.5 to 2.0, default: 1.0)
            audio_format (Optional[str]): Audio format (wav or mp3, default: wav)
            seed (Optional[int]): Random seed for reproducible results

        Returns:
            str: Path to the generated audio file

        Raises:
            ValueError: If parameters are invalid
            Exception: If API request fails or file save fails

        """
        try:
            from typecast.client import Typecast
            from typecast.models import TTSRequest, Prompt, Output
            from typecast.exceptions import (
                BadRequestError,
                InternalServerError,
                NotFoundError,
                PaymentRequiredError,
                TypecastError,
                UnauthorizedError,
                UnprocessableEntityError,
            )

            # Validate parameters
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            if not voice_id:
                raise ValueError("Voice ID is required")
            if not output_path:
                raise ValueError("Output path is required")

            # Create client
            client = Typecast(host=self.host, api_key=self.api_key)

            # Build the request
            request = TTSRequest(
                voice_id=voice_id,
                text=text,
                model=model,
                language=language,
                prompt=Prompt(
                    emotion_preset=emotion_preset,
                    emotion_intensity=emotion_intensity,
                ),
                output=Output(
                    volume=volume,
                    audio_pitch=audio_pitch,
                    audio_tempo=audio_tempo,
                    audio_format=audio_format,
                ),
                seed=seed,
            )

            # Generate audio
            try:
                response = client.text_to_speech(request)
            except BadRequestError as e:
                raise Exception(
                    f"Invalid request parameters: {e!s}"
                )
            except UnauthorizedError:
                raise Exception(
                    "Typecast API authentication failed. Please check your API key."
                )
            except PaymentRequiredError:
                raise Exception(
                    "Typecast API quota exceeded. Please check your account balance."
                )
            except NotFoundError as e:
                raise Exception(
                    f"Resource not found: {e!s}"
                )
            except UnprocessableEntityError as e:
                raise Exception(
                    f"Validation error: {e!s}"
                )
            except InternalServerError as e:
                raise Exception(
                    f"Typecast API server error: {e!s}"
                )
            except TypecastError as e:
                raise Exception(f"Typecast API error: {e!s}")

            # Save the audio
            try:
                with open(output_path, "wb") as fp:
                    fp.write(response.audio_data)
            except IOError as e:
                raise Exception(f"Failed to save audio file to {output_path}: {e!s}")

            # Return the save location
            return output_path

        except ValueError:
            raise
        except Exception as e:
            if "Typecast API" in str(e) or "quota" in str(e).lower():
                raise
            raise Exception(f"Failed to generate speech: {e!s}")
