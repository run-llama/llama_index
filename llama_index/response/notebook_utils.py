"""Utils for jupyter notebook."""
from typing import Any, Dict, Tuple

from IPython.display import Markdown, display

from llama_index.data_structs.node import ImageNode, NodeWithScore
from llama_index.img_utils import b64_2_img
from llama_index.response.schema import Response
from llama_index.utils import truncate_text

DEFAULT_THUMBNAIL_SIZE = (512, 512)


def display_image(img_str: str, size: Tuple[int, int] = DEFAULT_THUMBNAIL_SIZE) -> None:
    """Display base64 encoded image str as image for jupyter notebook."""
    img = b64_2_img(img_str)
    img.thumbnail(size)
    display(img)


def display_source_node(source_node: NodeWithScore, source_length: int = 100) -> None:
    """Display source node for jupyter notebook."""
    source_text_fmt = truncate_text(source_node.node.get_text().strip(), source_length)
    text_md = (
        f"**Document ID:** {source_node.node.doc_id}<br>"
        f"**Similarity:** {source_node.score}<br>"
        f"**Text:** {source_text_fmt}<br>"
    )
    if isinstance(source_node.node, ImageNode):
        text_md += "**Image:**"

    display(Markdown(text_md))
    if isinstance(source_node.node, ImageNode) and source_node.node.image is not None:
        display_image(source_node.node.image)


def display_extra_info(extra_info: Dict[str, Any]) -> None:
    """Display extra info for jupyter notebook."""
    display(extra_info)


def display_response(
    response: Response,
    source_length: int = 100,
    show_source: bool = False,
    show_extra_info: bool = False,
) -> None:
    """Display response for jupyter notebook."""
    if response.response is None:
        response_text = "None"
    else:
        response_text = response.response.strip()

    display(Markdown(f"**`Final Response:`** {response_text}"))
    if show_source:
        for ind, source_node in enumerate(response.source_nodes):
            display(Markdown("---"))
            display(
                Markdown(f"**`Source Node {ind + 1}/{len(response.source_nodes)}`**")
            )
            display_source_node(source_node, source_length=source_length)
    if show_extra_info:
        if response.extra_info is not None:
            display_extra_info(response.extra_info)
