"""Google Vertex AI CAMB.AI MARS7 text-to-speech tool spec."""

import json
import os
from typing import Optional, Literal

import base64
from google.cloud import aiplatform

from llama_index.core.tools.tool_spec.base import BaseToolSpec

# Mars7 supported languages constant
Mars7Language = Literal[
    "de-de",
    "en-gb",
    "en-us",
    "es-us",
    "es-es",
    "fr-ca",
    "fr-fr",
    "ja-jp",
    "ko-kr",
    "zh-cn",
]


class GoogleVertexCambToolSpec(BaseToolSpec):
    """Google Vertex AI CAMB.AI MARS7 tool spec for text-to-speech synthesis."""

    spec_functions = ["text_to_speech"]

    def __init__(
        self,
        project_id: str,
        location: str,
        endpoint_id: str,
        credentials_path: str,
    ) -> None:
        """
        Initialize Google Vertex AI CAMB.AI MARS7 tool spec.

        Args:
            project_id (str): Google Cloud project ID
            location (str): Google Cloud location (e.g., 'us-central1')
            endpoint_id (str): Vertex AI endpoint ID for the deployed MARS7 model
            credentials_path (str): Path to Google Cloud service account key file

        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable must be set "
                "with path to service account key file."
            )

        try:
            aiplatform.init(
                project=project_id,
                location=location,
            )
            self.endpoint = aiplatform.Endpoint(endpoint_id)
        except Exception as e:
            raise ValueError(f"Failed to initialize Vertex AI client: {e}")

    def text_to_speech(
        self,
        text: str,
        reference_audio_path: str,
        reference_text: Optional[str] = None,
        language: Mars7Language = "en-us",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Convert text to speech using Google Vertex AI CAMB.AI MARS7 model.

        Args:
            text (str): The text to convert to speech
            reference_audio_path (str): Path to reference audio file for voice cloning
            reference_text (Optional[str]): Transcription of the reference audio
            language (Mars7Language): Target language code (e.g., 'en-us', 'es-es')
            output_path (Optional[str]): Path to save the audio file. If None, generates 'cambai_speech.flac'

        Returns:
            str: Path to the generated audio file

        """
        if reference_audio_path is not None:
            try:
                with open(reference_audio_path, "rb") as f:
                    audio_ref = base64.b64encode(f.read()).decode("utf-8")
            except FileNotFoundError:
                raise ValueError(
                    f"Reference audio file not found: {reference_audio_path}"
                )
            except Exception as e:
                raise ValueError(f"Error reading reference audio file: {e}")
        else:
            audio_ref = None

        instances = {
            "text": text,
            "language": language,
        }

        if audio_ref is not None:
            instances["audio_ref"] = audio_ref

        if reference_text is not None:
            instances["ref_text"] = reference_text

        data = {"instances": [instances]}

        if output_path is None:
            output_path = "cambai_speech.flac"

        response = self.endpoint.raw_predict(
            body=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        response_data = json.loads(response.content)
        predictions = response_data.get("predictions", [])

        if not predictions or len(predictions) == 0:
            raise RuntimeError("No audio predictions returned from the model")

        with open(output_path, "wb") as f:
            audio_bytes = base64.b64decode(predictions[0])
            f.write(audio_bytes)

        return output_path
