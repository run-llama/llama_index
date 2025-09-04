from typing import Any, Optional, Sequence, List, cast

from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.core.base.llms.types import ContentBlock
from llama_index.core.base.llms.generic_utils import image_node_to_image_block
from llama_index.core.schema import ImageDocument, ImageNode


def generate_gemini_multi_modal_chat_message(
    prompt: str,
    role: str,
    image_documents: Optional[Sequence[ImageDocument]] = None,
    **kwargs: Any,
) -> ChatMessage:
    # if image_documents is empty, return text only chat message
    if image_documents is None or len(image_documents) == 0:
        return ChatMessage(role=role, content=prompt)

    blocks: List[ContentBlock] = [TextBlock(text=prompt)]
    if all(isinstance(doc, ImageNode) for doc in image_documents):
        image_docs: List[ImageBlock] = [
            image_node_to_image_block(doc) for doc in image_documents
        ]
    else:
        image_docs = cast(List[ImageBlock], image_documents)
    blocks.extend(image_docs)
    return ChatMessage(role=role, blocks=blocks)
