"""Tests for language detection and processing utilities."""

from llama_index.core.node_parser.text.language_utils import (
    calculate_language_adaptive_threshold,
    detect_language,
    estimate_language_density,
    get_adaptive_buffer_size,
    get_multilingual_tokenizer,
)


def test_detect_language_chinese():
    """Test Chinese language detection."""
    text = "这是一个测试句子。这是另一个测试句子。"
    language = detect_language(text)
    assert language == "zh"


def test_detect_language_japanese():
    """Test Japanese language detection."""
    text = "これはテストです。これは別のテストです。"
    language = detect_language(text)
    assert language == "ja"


def test_detect_language_korean():
    """Test Korean language detection."""
    text = "이것은 테스트입니다. 이것은 다른 테스트입니다."
    language = detect_language(text)
    assert language == "ko"


def test_detect_language_arabic():
    """Test Arabic language detection."""
    text = "هذا اختبار. هذا اختبار آخر."
    language = detect_language(text)
    assert language == "ar"


def test_detect_language_hebrew():
    """Test Hebrew language detection."""
    text = "זוהי בדיקה. זוהי בדיקה אחרת."
    language = detect_language(text)
    assert language == "he"


def test_detect_language_thai():
    """Test Thai language detection."""
    text = "นี่คือการทดสอบ. นี่คือการทดสอบอีกครั้ง."
    language = detect_language(text)
    assert language == "th"


def test_detect_language_hindi():
    """Test Hindi language detection."""
    text = "यह एक परीक्षण है. यह एक अन्य परीक्षण है."
    language = detect_language(text)
    assert language == "hi"


def test_detect_language_english():
    """Test English language detection."""
    text = "This is a test. This is another test."
    language = detect_language(text)
    assert language == "other"


def test_detect_language_empty():
    """Test empty text detection."""
    language = detect_language("")
    assert language == "other"


def test_detect_language_short():
    """Test very short text detection."""
    language = detect_language("Hi")
    assert language == "other"


def test_estimate_language_density_chinese():
    """Test Chinese language density estimation."""
    text = "这是一个测试句子"
    density = estimate_language_density(text)
    assert density == 2.5


def test_estimate_language_density_japanese():
    """Test Japanese language density estimation."""
    text = "これはテストです"
    density = estimate_language_density(text)
    assert density == 2.5


def test_estimate_language_density_korean():
    """Test Korean language density estimation."""
    text = "이것은 테스트입니다"
    density = estimate_language_density(text)
    assert density == 2.5


def test_estimate_language_density_english():
    """Test English language density estimation."""
    text = "This is a test"
    density = estimate_language_density(text)
    assert density == 4.0


def test_get_adaptive_buffer_size_chinese():
    """Test Chinese language adaptive buffer size."""
    buffer = get_adaptive_buffer_size(
        "zh", base_buffer_size=2, min_buffer_size=1, max_buffer_size=5
    )
    # Chinese should get half the buffer size
    assert buffer == 1


def test_get_adaptive_buffer_size_japanese():
    """Test Japanese language adaptive buffer size."""
    buffer = get_adaptive_buffer_size(
        "ja", base_buffer_size=3, min_buffer_size=1, max_buffer_size=5
    )
    # Japanese should get half the buffer size
    assert buffer == 1


def test_get_adaptive_buffer_size_korean():
    """Test Korean language adaptive buffer size."""
    buffer = get_adaptive_buffer_size(
        "ko", base_buffer_size=4, min_buffer_size=1, max_buffer_size=5
    )
    # Korean should get half the buffer size
    assert buffer == 2


def test_get_adaptive_buffer_size_arabic():
    """Test Arabic language adaptive buffer size."""
    buffer = get_adaptive_buffer_size(
        "ar", base_buffer_size=2, min_buffer_size=1, max_buffer_size=5
    )
    # Arabic should get 1/3 of buffer size
    assert buffer == 1


def test_get_adaptive_buffer_size_hebrew():
    """Test Hebrew language adaptive buffer size."""
    buffer = get_adaptive_buffer_size(
        "he", base_buffer_size=3, min_buffer_size=1, max_buffer_size=5
    )
    # Hebrew should get 1/3 of buffer size
    assert buffer == 1


def test_get_adaptive_buffer_size_english():
    """Test English language adaptive buffer size."""
    buffer = get_adaptive_buffer_size(
        "other", base_buffer_size=2, min_buffer_size=1, max_buffer_size=5
    )
    # English should use standard buffer size
    assert buffer == 2


def test_get_adaptive_buffer_size_clamped():
    """Test adaptive buffer size clamping to min/max."""
    buffer = get_adaptive_buffer_size(
        "zh", base_buffer_size=1, min_buffer_size=1, max_buffer_size=5
    )
    # Should be clamped to min
    assert buffer == 1


def test_calculate_language_adaptive_threshold_chinese():
    """Test Chinese language adaptive threshold."""
    threshold = calculate_language_adaptive_threshold("zh", base_threshold=95)
    # Chinese should have higher threshold (0.98)
    assert 0.97 <= threshold <= 0.99


def test_calculate_language_adaptive_threshold_japanese():
    """Test Japanese language adaptive threshold."""
    threshold = calculate_language_adaptive_threshold("ja", base_threshold=95)
    # Japanese should have higher threshold (0.98)
    assert 0.97 <= threshold <= 0.99


def test_calculate_language_adaptive_threshold_korean():
    """Test Korean language adaptive threshold."""
    threshold = calculate_language_adaptive_threshold("ko", base_threshold=95)
    # Korean should have higher threshold (0.98)
    assert 0.97 <= threshold <= 0.99


def test_calculate_language_adaptive_threshold_arabic():
    """Test Arabic language adaptive threshold."""
    threshold = calculate_language_adaptive_threshold("ar", base_threshold=95)
    # Arabic should have slightly higher threshold (0.97)
    assert 0.96 <= threshold <= 0.98


def test_calculate_language_adaptive_threshold_hebrew():
    """Test Hebrew language adaptive threshold."""
    threshold = calculate_language_adaptive_threshold("he", base_threshold=95)
    # Hebrew should have slightly higher threshold (0.97)
    assert 0.96 <= threshold <= 0.98


def test_calculate_language_adaptive_threshold_english():
    """Test English language adaptive threshold."""
    threshold = calculate_language_adaptive_threshold("other", base_threshold=95)
    # English should use base threshold
    assert threshold == 95.0


def test_calculate_language_adaptive_threshold_multiplier():
    """Test adaptive threshold with multiplier."""
    threshold = calculate_language_adaptive_threshold("zh", base_threshold=95)
    # Apply multiplier
    threshold_with_multiplier = min(99.0, threshold + 0.1)
    assert threshold_with_multiplier > threshold


def test_get_multilingual_tokenizer_chinese():
    """Test Chinese multilingual tokenizer."""
    tokenizer = get_multilingual_tokenizer("zh")
    # Should return a character-based tokenizer
    result = tokenizer("测试")
    # With chunk size of 4, a 2-character string returns as single element
    assert len(result) == 1


def test_get_multilingual_tokenizer_japanese():
    """Test Japanese multilingual tokenizer."""
    tokenizer = get_multilingual_tokenizer("ja")
    # Should return a character-based tokenizer
    result = tokenizer("テスト")
    # With chunk size of 4, a 4-character string returns as single element
    assert len(result) == 1


def test_get_multilingual_tokenizer_korean():
    """Test Korean multilingual tokenizer."""
    tokenizer = get_multilingual_tokenizer("ko")
    # Should return a character-based tokenizer
    result = tokenizer("테스트")
    # With chunk size of 4, a 4-character string returns as single element
    assert len(result) == 1


def test_get_multilingual_tokenizer_english():
    """Test English multilingual tokenizer with default fallback."""
    # get_multilingual_tokenizer("other") returns a callable (the base_tokenizer)
    # which is a lambda function returned by split_by_sentence_tokenizer()
    tokenizer_fn = get_multilingual_tokenizer("other")

    # Calling the tokenizer with text should work
    result = tokenizer_fn("This is a test.")
    assert len(result) > 0


def test_get_multilingual_tokenizer_with_base():
    """Test multilingual tokenizer with custom base."""

    def custom_tokenizer(text: str) -> list:
        return [text]

    tokenizer = get_multilingual_tokenizer("zh", base_tokenizer=custom_tokenizer)
    result = tokenizer("测试")
    assert result == ["测试"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
