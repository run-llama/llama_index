"""
Utilities for (truly) multimodal embeddings with interleaved content.

Uses the same content block types as the rest of the stack (Node, LLM content blocks):
TextBlock, ImageBlock, AudioBlock, VideoBlock from llama_index.core.base.llms.types.
Embedding backends (Cohere, Voyage, etc.) convert these to their API format.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from llama_index.core.base.llms.types import (
    AudioBlock,
    ImageBlock,
    TextBlock,
    VideoBlock,
)


# Typed multimodal content: sequence of embeddable content blocks (same concept as Node resources / LLM blocks).
EmbeddableContentBlock = Union[TextBlock, ImageBlock, AudioBlock, VideoBlock]
MixedEmbeddingContent = List[EmbeddableContentBlock]


def content_blocks_to_mixed_embedding_content(
    blocks: List[Any],
) -> Optional[MixedEmbeddingContent]:
    """
    Extract mixed (interleaved) embeddable content from content blocks.

    Returns None if the blocks do not represent mixed content suitable for
    joint embedding. Otherwise returns the ordered list of embeddable blocks
    (Text, Image, Audio, Video). Backends convert these to their API format.

    Args:
        blocks: List of content blocks from e.g. node.get_content_blocks().

    Returns:
        List of embeddable content blocks (TextBlock, ImageBlock, AudioBlock,
        VideoBlock) when mixed, or None if single-modality or empty.

    Note:
        Support for audio/video in the embedding API is backend-dependent
        (e.g. Voyage supports video; Cohere is text+image only).

    """
    embedding_blocks: MixedEmbeddingContent = [
        b
        for b in blocks
        if isinstance(b, (TextBlock, ImageBlock, AudioBlock, VideoBlock))
    ]
    if not embedding_blocks:
        return None
    has_text = any(isinstance(b, TextBlock) for b in embedding_blocks)
    has_image = any(isinstance(b, ImageBlock) for b in embedding_blocks)
    has_audio = any(isinstance(b, AudioBlock) for b in embedding_blocks)
    has_video = any(isinstance(b, VideoBlock) for b in embedding_blocks)
    has_media = has_image or has_audio or has_video
    if not has_media:
        return None
    if len(embedding_blocks) == 1 and not has_text:
        return None  # single media block: use regular modality-specific path
    return embedding_blocks
