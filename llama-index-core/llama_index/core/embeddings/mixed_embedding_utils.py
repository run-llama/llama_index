"""
Utilities for (truly) multimodal embeddings with interleaved text and images.

Supports embedding models (e.g. Cohere, Voyage) that accept a single sequence of
content items like:
  [{"type": "text", "text": "Look at this:"},
   {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Format accepted by Cohere and Voyage multimodal embed APIs
MixedEmbeddingContent = List[Dict[str, Any]]


def content_blocks_to_mixed_embedding_content(
    blocks: List[Any],
) -> Optional[MixedEmbeddingContent]:
    """
    Convert content blocks (TextBlock, ImageBlock) to the mixed embedding API format.

    Returns None if the blocks do not represent mixed (interleaved) content
    suitable for joint embedding. Only text and image blocks are included;
    other block types are skipped.

    Args:
        blocks: List of content blocks from e.g. node.get_content_blocks().

    Returns:
        List of {"type": "text", "text": "..."} and
        {"type": "image_url", "image_url": {"url": "data:..."}} items,
        or None if not mixed (e.g. single text-only or single image-only).

    """
    from llama_index.core.base.llms.types import ImageBlock, TextBlock

    embedding_blocks = [b for b in blocks if isinstance(b, (TextBlock, ImageBlock))]
    if not embedding_blocks:
        return None
    has_text = any(isinstance(b, TextBlock) for b in embedding_blocks)
    has_image = any(isinstance(b, ImageBlock) for b in embedding_blocks)
    # Only treat as mixed when we have both modalities or multiple blocks with image
    if not has_image:
        return None
    if len(embedding_blocks) == 1 and not has_text:
        return None  # single image: use regular image embedding path

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
    return content if content else None
