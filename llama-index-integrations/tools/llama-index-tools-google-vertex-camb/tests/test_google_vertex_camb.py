"""Tests for Google Vertex AI CAMB.AI MARS7 tool spec."""

import pytest
from unittest.mock import Mock, patch, mock_open
from llama_index.tools.google_vertex_camb.base import GoogleVertexCambToolSpec


class TestGoogleVertexCambToolSpec:
    """Test GoogleVertexCambToolSpec functionality."""

    def test_inheritance(self):
        """Test that GoogleVertexCambToolSpec inherits from BaseToolSpec."""
        from llama_index.core.tools.tool_spec.base import BaseToolSpec

        assert issubclass(GoogleVertexCambToolSpec, BaseToolSpec)

    def test_spec_functions(self):
        """Test that spec_functions is properly defined."""
        assert GoogleVertexCambToolSpec.spec_functions == ["text_to_speech"]

    @patch("llama_index.tools.google_vertex_camb.base.aiplatform")
    @patch("os.environ")
    def test_init_success(self, mock_environ, mock_aiplatform):
        """Test successful initialization."""
        mock_environ.get.return_value = "/path/to/credentials.json"
        mock_endpoint = Mock()
        mock_aiplatform.Endpoint.return_value = mock_endpoint

        tool = GoogleVertexCambToolSpec(
            project_id="test-project",
            location="us-central1",
            endpoint_id="test-endpoint",
            credentials_path="/path/to/credentials.json",
        )

        mock_aiplatform.init.assert_called_once_with(
            project="test-project", location="us-central1"
        )
        mock_aiplatform.Endpoint.assert_called_once_with("test-endpoint")
        assert tool.endpoint == mock_endpoint

    @patch("llama_index.tools.google_vertex_camb.base.aiplatform")
    @patch("os.environ")
    def test_init_no_credentials(self, mock_environ, mock_aiplatform):
        """Test initialization with missing credentials."""
        mock_environ.get.return_value = None

        with pytest.raises(ValueError, match="GOOGLE_APPLICATION_CREDENTIALS"):
            GoogleVertexCambToolSpec(
                project_id="test-project",
                location="us-central1",
                endpoint_id="test-endpoint",
                credentials_path="/path/to/credentials.json",
            )

    @patch("llama_index.tools.google_vertex_camb.base.aiplatform")
    @patch("os.environ")
    def test_init_aiplatform_error(self, mock_environ, mock_aiplatform):
        """Test initialization with aiplatform error."""
        mock_environ.get.return_value = "/path/to/credentials.json"
        mock_aiplatform.init.side_effect = Exception("Init failed")

        with pytest.raises(ValueError, match="Failed to initialize Vertex AI client"):
            GoogleVertexCambToolSpec(
                project_id="test-project",
                location="us-central1",
                endpoint_id="test-endpoint",
                credentials_path="/path/to/credentials.json",
            )

    @patch("llama_index.tools.google_vertex_camb.base.aiplatform")
    @patch("os.environ")
    @patch("builtins.open", new_callable=mock_open, read_data=b"audio_data")
    @patch("base64.b64encode")
    @patch("json.dumps")
    @patch("json.loads")
    @patch("base64.b64decode")
    def test_text_to_speech_with_reference_audio(
        self,
        mock_b64decode,
        mock_json_loads,
        mock_json_dumps,
        mock_b64encode,
        mock_file_open,
        mock_environ,
        mock_aiplatform,
    ):
        """Test text_to_speech with reference audio."""
        # Setup mocks
        mock_environ.get.return_value = "/path/to/credentials.json"
        mock_endpoint = Mock()
        mock_aiplatform.Endpoint.return_value = mock_endpoint

        mock_b64encode.return_value.decode.return_value = "encoded_audio"
        mock_json_dumps.return_value = '{"instances": []}'
        mock_response = Mock()
        mock_response.content = '{"predictions": ["audio_prediction"]}'
        mock_endpoint.raw_predict.return_value = mock_response
        mock_json_loads.return_value = {"predictions": ["audio_prediction"]}
        mock_b64decode.return_value = b"decoded_audio"

        # Create tool instance
        tool = GoogleVertexCambToolSpec(
            project_id="test-project",
            location="us-central1",
            endpoint_id="test-endpoint",
            credentials_path="/path/to/credentials.json",
        )

        # Test text_to_speech
        result = tool.text_to_speech(
            text="Hello world",
            reference_audio_path="/path/to/reference.wav",
            reference_text="Reference text",
            language="en-us",
            output_path="output.flac",
        )

        # Verify calls
        mock_endpoint.raw_predict.assert_called_once()
        assert result == "output.flac"

    @patch("llama_index.tools.google_vertex_camb.base.aiplatform")
    @patch("os.environ")
    def test_text_to_speech_missing_reference_file(self, mock_environ, mock_aiplatform):
        """Test text_to_speech with missing reference audio file."""
        mock_environ.get.return_value = "/path/to/credentials.json"
        mock_endpoint = Mock()
        mock_aiplatform.Endpoint.return_value = mock_endpoint

        tool = GoogleVertexCambToolSpec(
            project_id="test-project",
            location="us-central1",
            endpoint_id="test-endpoint",
            credentials_path="/path/to/credentials.json",
        )

        with pytest.raises(ValueError, match="Reference audio file not found"):
            tool.text_to_speech(
                text="Hello world", reference_audio_path="/nonexistent/file.wav"
            )

    @patch("llama_index.tools.google_vertex_camb.base.aiplatform")
    @patch("os.environ")
    @patch("json.dumps")
    def test_text_to_speech_no_predictions(
        self, mock_json_dumps, mock_environ, mock_aiplatform
    ):
        """Test text_to_speech with no predictions from model."""
        mock_environ.get.return_value = "/path/to/credentials.json"
        mock_endpoint = Mock()
        mock_aiplatform.Endpoint.return_value = mock_endpoint

        mock_json_dumps.return_value = '{"instances": []}'
        mock_response = Mock()
        mock_response.content = '{"predictions": []}'
        mock_endpoint.raw_predict.return_value = mock_response

        tool = GoogleVertexCambToolSpec(
            project_id="test-project",
            location="us-central1",
            endpoint_id="test-endpoint",
            credentials_path="/path/to/credentials.json",
        )

        with pytest.raises(RuntimeError, match="No audio predictions returned"):
            tool.text_to_speech(text="Hello world")

    @patch("llama_index.tools.google_vertex_camb.base.aiplatform")
    @patch("os.environ")
    @patch("json.dumps")
    def test_text_to_speech_without_reference_audio(
        self, mock_json_dumps, mock_environ, mock_aiplatform
    ):
        """Test text_to_speech without reference audio."""
        mock_environ.get.return_value = "/path/to/credentials.json"
        mock_endpoint = Mock()
        mock_aiplatform.Endpoint.return_value = mock_endpoint

        mock_json_dumps.return_value = '{"instances": []}'
        mock_response = Mock()
        mock_response.content = '{"predictions": ["audio_prediction"]}'
        mock_endpoint.raw_predict.return_value = mock_response

        with (
            patch("json.loads") as mock_json_loads,
            patch("base64.b64decode") as mock_b64decode,
            patch("builtins.open", mock_open()) as mock_file_open,
        ):
            mock_json_loads.return_value = {"predictions": ["audio_prediction"]}
            mock_b64decode.return_value = b"decoded_audio"

            tool = GoogleVertexCambToolSpec(
                project_id="test-project",
                location="us-central1",
                endpoint_id="test-endpoint",
                credentials_path="/path/to/credentials.json",
            )

            result = tool.text_to_speech(text="Hello world")

            assert result == "cambai_speech.flac"
            mock_endpoint.raw_predict.assert_called_once()
