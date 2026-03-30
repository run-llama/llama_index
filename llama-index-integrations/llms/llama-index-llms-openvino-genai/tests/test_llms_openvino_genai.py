import importlib.util
from unittest.mock import MagicMock, patch

import pytest

has_openvino_genai = importlib.util.find_spec("openvino_genai") is not None


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_llm_class():
    from llama_index.core.llms.custom import CustomLLM
    from llama_index.llms.openvino_genai import OpenVINOGenAILLM

    names_of_base_classes = [b.__name__ for b in OpenVINOGenAILLM.__mro__]
    assert CustomLLM.__name__ in names_of_base_classes


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_streamer_uses_write_method():
    """Verify that the streamer classes define a 'write' method (not deprecated 'put')."""
    import inspect

    from llama_index.llms.openvino_genai.base import OpenVINOGenAILLM

    source = inspect.getsource(OpenVINOGenAILLM)
    assert "def write(" in source
    assert "StreamingStatus" in source
    # Ensure old deprecated 'put' method is not present
    assert "def put(" not in source


def _create_llm_with_mock():
    """Helper to create an OpenVINOGenAILLM instance with mocked pipeline."""
    import openvino_genai

    from llama_index.llms.openvino_genai.base import OpenVINOGenAILLM

    mock_tokenizer = MagicMock(spec=openvino_genai.Tokenizer)
    mock_tokenizer.decode.return_value = "hello"

    mock_pipe = MagicMock()
    mock_pipe.get_generation_config.return_value = MagicMock()
    mock_pipe.get_tokenizer.return_value = mock_tokenizer

    with patch("openvino_genai.LLMPipeline", return_value=mock_pipe):
        llm = OpenVINOGenAILLM(model_path="/fake/path", device="CPU")
    return llm, mock_tokenizer


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_get_stop_flag():
    """Test that get_stop_flag returns StreamingStatus.RUNNING (line 179)."""
    import openvino_genai

    llm, _ = _create_llm_with_mock()
    assert llm._streamer.get_stop_flag() == openvino_genai.StreamingStatus.RUNNING


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_chunk_streamer_write_caches_tokens():
    """Test ChunkStreamer.write caches tokens until tokens_len (lines 286-287)."""
    import openvino_genai

    llm, _ = _create_llm_with_mock()
    streamer = llm._streamer

    # First 3 tokens should be cached (tokens_len=4)
    for token in [10, 11, 12]:
        result = streamer.write(token)
        assert result == openvino_genai.StreamingStatus.RUNNING

    assert streamer.tokens_cache == [10, 11, 12]
    assert streamer.decoded_lengths == [-2, -2, -2]


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_chunk_streamer_write_delegates_on_boundary():
    """Test ChunkStreamer.write delegates to IterableStreamer.write on
    tokens_len boundary (line 289), covering IterableStreamer.write
    single-token path (lines 212-213, 216, 224-232, 234-235, 237)."""
    import openvino_genai

    llm, mock_tokenizer = _create_llm_with_mock()
    streamer = llm._streamer

    # Grow decode output to trigger the delay_n_tokens print logic
    mock_tokenizer.decode.side_effect = lambda tokens: "ab" * len(tokens)

    # Cache 3 tokens, 4th triggers super().write
    streamer.write(1)
    streamer.write(2)
    streamer.write(3)
    result = streamer.write(4)

    assert result == openvino_genai.StreamingStatus.RUNNING
    # tokens_cache still holds all tokens (no newline reset)
    assert streamer.tokens_cache == [1, 2, 3, 4]
    # decoded_lengths was populated for computed positions
    assert len(streamer.decoded_lengths) == 4


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_chunk_streamer_write_list():
    """Test ChunkStreamer.write with list input (lines 283-284)."""
    import openvino_genai

    llm, _ = _create_llm_with_mock()
    streamer = llm._streamer

    result = streamer.write([10, 20, 30])
    assert result == openvino_genai.StreamingStatus.RUNNING


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_iterable_streamer_write_list_tokens():
    """Test IterableStreamer.write with a list of tokens (lines 206-208)."""
    import openvino_genai

    llm, mock_tokenizer = _create_llm_with_mock()
    streamer = llm._streamer

    # Pass a list directly to ChunkStreamer which delegates to super().write
    mock_tokenizer.decode.return_value = "hello"
    result = streamer.write([1, 2, 3, 4])

    assert result == openvino_genai.StreamingStatus.RUNNING
    assert 1 in streamer.tokens_cache
    assert 4 in streamer.tokens_cache


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_iterable_streamer_write_newline_resets():
    """Test IterableStreamer.write resets on newline (line 220)."""
    import openvino_genai

    llm, mock_tokenizer = _create_llm_with_mock()
    streamer = llm._streamer

    streamer.write(1)
    streamer.write(2)
    streamer.write(3)
    # 4th token triggers super().write; decode returns text ending in newline
    mock_tokenizer.decode.return_value = "hello\n"
    streamer.write(4)

    # After newline, tokens_cache and decoded_lengths are reset
    assert streamer.tokens_cache == []
    assert streamer.decoded_lengths == []
    assert streamer.print_len == 0


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_iterable_streamer_write_replacement_char():
    """Test IterableStreamer.write handles replacement character (lines 222-223)."""
    import openvino_genai

    llm, mock_tokenizer = _create_llm_with_mock()
    streamer = llm._streamer

    streamer.write(1)
    streamer.write(2)
    streamer.write(3)
    # 4th token triggers super().write; decode returns text ending with U+FFFD
    mock_tokenizer.decode.return_value = "abc" + chr(65533)
    streamer.write(4)

    # decoded_lengths[-1] should be -1 for the replacement character
    assert streamer.decoded_lengths[-1] == -1


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_compute_decoded_length():
    """Test _compute_decoded_length (lines 244-251)."""
    import openvino_genai

    llm, mock_tokenizer = _create_llm_with_mock()
    streamer = llm._streamer

    # Manually set up state to test _compute_decoded_length directly
    streamer.tokens_cache = [1, 2, 3, 4, 5]
    streamer.decoded_lengths = [-2, -2, -2, -2, 5]

    # Simulate decode returning text without replacement char
    mock_tokenizer.decode.return_value = "ab"
    streamer._compute_decoded_length(1)
    assert streamer.decoded_lengths[1] == 2

    # Test with replacement character at end
    mock_tokenizer.decode.return_value = "x" + chr(65533)
    streamer._compute_decoded_length(0)
    assert streamer.decoded_lengths[0] == -1

    # Test skip when already computed (not -2)
    streamer.decoded_lengths[1] = 5
    streamer._compute_decoded_length(1)
    assert streamer.decoded_lengths[1] == 5  # unchanged
