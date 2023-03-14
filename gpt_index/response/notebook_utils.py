from typing import Any, Dict, Tuple

from IPython.display import Markdown, display

from gpt_index.img_utils import b64_2_img
from gpt_index.response.schema import Response, SourceNode
from gpt_index.utils import truncate_text

DEFAULT_THUMBNAIL_SIZE = (512, 512)


def display_image(image: str, size: Tuple[int, int] = DEFAULT_THUMBNAIL_SIZE) -> None:
    img = b64_2_img(image)
    img.thumbnail(size)
    display(img)


def display_source_node(source_node: SourceNode, source_length: int = 100) -> None:
    source_text_fmt = truncate_text(source_node.source_text.strip(), source_length)
    text_md = (
        f"**Document ID:** {source_node.doc_id}<br>"
        f"**Similarity:** {source_node.similarity}<br>"
        f"**Text:** {source_text_fmt}<br>"
    )
    if source_node.image is not None:
        text_md += "**Image:**"
    display(Markdown(text_md))
    if source_node.image is not None:
        display_image(source_node.image)


def display_extra_info(extra_info: Dict[str, Any]) -> None:
    display(extra_info)


def display_response(response: Response, source_length: int = 100) -> None:
    if response.response is None:
        response_text = "None"
    else:
        response_text = response.response.strip()

    display(Markdown(f"**`Final Response:`** {response_text}"))
    for ind, source_node in enumerate(response.source_nodes):
        display(Markdown("---"))
        display(Markdown(f"**`Source Node {ind + 1}/{len(response.source_nodes)}`**"))
        display_source_node(source_node, source_length=source_length)
    if response.extra_info is not None:
        display_extra_info(response.extra_info)
