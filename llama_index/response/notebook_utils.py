"""Utils for jupyter notebook."""
from typing import Any, Dict, Tuple

from IPython.display import Markdown, display

from llama_index.img_utils import b64_2_img
from llama_index.response.schema import Response
from llama_index.schema import ImageNode, MetadataMode, NodeWithScore
from llama_index.utils import truncate_text

DEFAULT_THUMBNAIL_SIZE = (512, 512)


def display_image(img_str: str, size: Tuple[int, int] = DEFAULT_THUMBNAIL_SIZE) -> None:
    """Display base64 encoded image str as image for jupyter notebook."""
    img = b64_2_img(img_str)
    img.thumbnail(size)
    display(img)


def display_source_node(
    source_node: NodeWithScore,
    source_length: int = 100,
    show_source_metadata: bool = False,
    metadata_mode: MetadataMode = MetadataMode.NONE,
) -> None:
    """Display source node for jupyter notebook."""
    source_text_fmt = truncate_text(
        source_node.node.get_content(metadata_mode=metadata_mode).strip(), source_length
    )
    text_md = (
        f"**Node ID:** {source_node.node.node_id}<br>"
        f"**Similarity:** {source_node.score}<br>"
        f"**Text:** {source_text_fmt}<br>"
    )
    if show_source_metadata:
        text_md += f"**Metadata:** {source_node.node.metadata}<br>"
    if isinstance(source_node.node, ImageNode):
        text_md += "**Image:**"

    display(Markdown(text_md))
    if isinstance(source_node.node, ImageNode) and source_node.node.image is not None:
        display_image(source_node.node.image)


def display_metadata(metadata: Dict[str, Any]) -> None:
    """Display metadata for jupyter notebook."""
    display(metadata)


def display_response(
    response: Response,
    source_length: int = 100,
    show_source: bool = False,
    show_metadata: bool = False,
    show_source_metadata: bool = False,
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
            display_source_node(
                source_node,
                source_length=source_length,
                show_source_metadata=show_source_metadata,
            )
    if show_metadata:
        if response.metadata is not None:
            display_metadata(response.metadata)
