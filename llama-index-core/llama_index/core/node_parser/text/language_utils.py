"""Language detection and processing utilities for text chunking."""

from typing import Callable, List, Optional, Tuple
import re

# Language density configuration (approximate characters per token)
# Used to determine appropriate buffer sizes
LANGUAGE_DENSITY = {
    "zh": 2.5,  # Chinese, Japanese, Korean (very dense)
    "ja": 2.5,
    "ko": 2.5,
    "ar": 3.0,  # Arabic (medium density, right-to-left)
    "he": 3.0,
    "th": 2.0,  # Thai (dense)
    "hi": 1.5,  # Hindi (medium density)
    "other": 4.0,  # Default (English-like, least dense)
}

# Default buffer size multiplier based on language density
DEFAULT_BUFFER_MULTIPLIER = 2.0


def detect_language(text: str) -> str:
    """
    Detect the dominant language of the text using a combination of approaches.

    Args:
        text: Input text to analyze

    Returns:
        Language code (e.g., 'zh', 'ja', 'en', 'ar', 'other')

    """
    if not text or len(text) < 3:
        return "other"

    # Try fast detection first with regex patterns
    # We need to check Japanese and Korean first as they share some ranges
    # Japanese characters: \u3040-\u309f, \u30a0-\u30ff, \u3005
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff\u3005]", text):
        return "ja"

    # Korean Hangul: \uac00-\ud7af
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"

    # Chinese characters: \u4e00-\u9fff
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"

    # Arabic: \u0600-\u06ff
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"

    # Hebrew: \u0590-\u05ff
    if re.search(r"[\u0590-\u05ff]", text):
        return "he"

    # Thai: \u0e00-\u0e7f
    if re.search(r"[\u0e00-\u0e7f]", text):
        return "th"

    # Hindi/Devanagari: \u0900-\u097f
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"

    # Check for Latin script but not English (e.g., French, German, Spanish)
    # If text has latin characters but doesn't match the above, consider it "other"
    # but might need special handling

    return "other"


def estimate_language_density(text: str) -> float:
    """
    Estimate the language density of text (characters per token).

    More dense languages (like Chinese) need smaller chunks.
    Less dense languages (like English) can use larger chunks.

    Args:
        text: Input text to analyze

    Returns:
        Approximate characters per token

    """
    language = detect_language(text)
    return LANGUAGE_DENSITY.get(language, LANGUAGE_DENSITY["other"])


def get_adaptive_buffer_size(
    language: str,
    base_buffer_size: int = 1,
    min_buffer_size: int = 1,
    max_buffer_size: int = 5,
) -> int:
    """
    Get an adaptive buffer size based on language characteristics.

    Denser languages benefit from smaller buffer sizes to avoid
    creating chunks that are too large in terms of characters.

    Args:
        language: Detected language code
        base_buffer_size: Base buffer size from user configuration
        min_buffer_size: Minimum allowed buffer size
        max_buffer_size: Maximum allowed buffer size

    Returns:
        Adaptive buffer size (integer)

    """
    # Chinese, Japanese, Korean need smaller buffers
    if language in ("zh", "ja", "ko"):
        return max(min_buffer_size, base_buffer_size // 2)

    # Arabic, Hebrew also benefit from slightly smaller buffers
    if language in ("ar", "he"):
        return max(min_buffer_size, base_buffer_size // 3)

    # Other languages can use standard buffer size
    return max(min_buffer_size, min(base_buffer_size, max_buffer_size))


def get_multilingual_tokenizer(
    language: str,
    base_tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Callable[[str], List[str]]:
    """
    Get an appropriate tokenizer for a given language.

    Args:
        language: Detected language code
        base_tokenizer: Fallback tokenizer if specific one not available

    Returns:
        Tokenizer function

    """
    # If no base tokenizer provided, use the default sentence tokenizer
    if base_tokenizer is None:
        from llama_index.core.node_parser.text.utils import (
            split_by_sentence_tokenizer,
        )

        base_tokenizer = split_by_sentence_tokenizer()

    # Chinese/Japanese/Korean: Use character-based tokenization
    # These languages don't use spaces between words
    if language in ("zh", "ja", "ko"):

        def zh_ja_ko_tokenizer(text: str) -> List[str]:
            # For Chinese/Japanese/Korean, split by characters or characters groups
            # This is a simple implementation - could be enhanced with word segmentation
            return [text[i : i + 4] for i in range(0, len(text), 4)]

        return zh_ja_ko_tokenizer

    # Arabic: Needs special handling for right-to-left
    if language in ("ar", "he"):
        # base_tokenizer is already a callable
        def rtl_tokenizer(text: str) -> List[str]:
            return base_tokenizer(text)

        return rtl_tokenizer

    # Other languages: Use base tokenizer
    return base_tokenizer


def analyze_multilingual_balance(
    text: str,
) -> Tuple[str, List[str], List[float]]:
    """
    Analyze text to detect language regions and their embeddings.

    Args:
        text: Input text

    Returns:
        Tuple of (detected_language, language_regions, embedding_densities)

    """
    language = detect_language(text)

    # For simplicity, assume single language for now
    # Could be extended to handle mixed-language text
    return language, [text], [1.0]


def calculate_language_adaptive_threshold(
    language: str,
    base_threshold: float = 0.95,
) -> float:
    """
    Calculate an adaptive dissimilarity threshold based on language.

    Dense languages may need different threshold values to achieve
    appropriate chunking behavior.

    Args:
        language: Detected language code
        base_threshold: Base threshold value

    Returns:
        Adaptive threshold value

    """
    # Chinese/Japanese/Korean: Higher thresholds work better
    if language in ("zh", "ja", "ko"):
        return min(0.98, base_threshold + 0.05)

    # Arabic/Hebrew: Slightly different threshold
    if language in ("ar", "he"):
        return min(0.97, base_threshold + 0.03)

    # Other languages: Use base threshold
    return base_threshold
