from unittest.mock import Mock, mock_open, patch

from llama_index.tools.typecast import TypecastToolSpec


def test_ssfm_v31_preset_and_voice_filter():
    with (
        patch("llama_index.tools.typecast.base.Typecast") as client,
        patch("builtins.open", mock_open()),
    ):
        client.return_value.text_to_speech.return_value = Mock(audio_data=b"audio")
        client.return_value.voices_v3.return_value = []
        tool = TypecastToolSpec(api_key="test")
        tool.text_to_speech("Hello", "tc_test", "unused.wav", model="ssfm-v31")
        request = client.return_value.text_to_speech.call_args.args[0]
        assert request.model.value == "ssfm-v31"
        assert request.prompt.emotion_type == "preset"
        tool.get_voices(model="ssfm-v31")
        assert (
            client.return_value.voices_v3.call_args.kwargs["filter"].model.value
            == "ssfm-v31"
        )
