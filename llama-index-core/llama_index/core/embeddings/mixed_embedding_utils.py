"""
Utilities for (truly) multimodal embeddings with interleaved content.

Supports embedding models (e.g. Cohere, Voyage) that accept a single sequence of
content items, e.g.:
  [{"type": "text", "text": "Look at this:"},
   {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}},
   {"type": "audio_url", "audio_url": {"url": "data:audio/...;base64,..."}},
   {"type": "video_url", "video_url": {"url": "data:video/...;base64,..."}}]

Support for audio and video in the embedding API is backend-dependent
(e.g. Voyage supports video; Cohere currently supports text and image only).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Format accepted by multimodal embed APIs (Cohere, Voyage, etc.)
MixedEmbeddingContent = List[Dict[str, Any]]


def content_blocks_to_mixed_embedding_content(
    blocks: List[Any],
) -> Optional[MixedEmbeddingContent]:
    """
    Convert content blocks (Text, Image, Audio, Video) to the mixed embedding API format.

    Returns None if the blocks do not represent mixed (interleaved) content
    suitable for joint embedding. Includes text, image, audio, and video blocks;
    other block types are skipped.

    Args:
        blocks: List of content blocks from e.g. node.get_content_blocks().

    Returns:
        List of content items, e.g.:
        - {"type": "text", "text": "..."}
        - {"type": "image_url", "image_url": {"url": "data:..."}}
        - {"type": "audio_url", "audio_url": {"url": "data:..."}}
        - {"type": "video_url", "video_url": {"url": "data:..."}}
        or None if not mixed (e.g. single text-only or single image-only).

    Note:
        Whether audio/video are actually embedded depends on the embedding
        backend (e.g. Voyage supports video; Cohere is text+image only).

    """
    from llama_index.core.base.llms.types import (
        AudioBlock,
        ImageBlock,
        TextBlock,
        VideoBlock,
    )

    embedding_blocks = [
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
    # Only treat as mixed when we have media and (text or multiple blocks)
    if not has_media:
        return None
    if len(embedding_blocks) == 1 and not has_text:
        return None  # single media block: use regular modality-specific path

    content: MixedEmbeddingContent = []
    for block in embedding_blocks:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": block.inline_url()},
                }
            )
        elif isinstance(block, AudioBlock):
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": block.inline_url()},
                }
            )
        elif isinstance(block, VideoBlock):
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": block.inline_url()},
                }
            )
    return content if content else None
