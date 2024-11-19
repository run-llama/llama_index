"""Utils for jupyter notebook."""
import os
from io import BytesIO
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import requests
from IPython.display import Markdown, display
from llama_index.core.base.response.schema import Response
from llama_index.core.img_utils import b64_2_img
from llama_index.core.schema import ImageNode, MetadataMode, NodeWithScore
from llama_index.core.utils import truncate_text
from PIL import Image

DEFAULT_THUMBNAIL_SIZE = (512, 512)
DEFAULT_IMAGE_MATRIX = (3, 3)
DEFAULT_SHOW_TOP_K = 3


def display_image(img_str: str, size: Tuple[int, int] = DEFAULT_THUMBNAIL_SIZE) -> None:
    """Display base64 encoded image str as image for jupyter notebook."""
    img = b64_2_img(img_str)
    img.thumbnail(size)
    display(img)


def display_image_uris(
    image_paths: List[str],
    image_matrix: Tuple[int, int] = DEFAULT_IMAGE_MATRIX,
    top_k: int = DEFAULT_SHOW_TOP_K,
) -> None:
    """Display base64 encoded image str as image for jupyter notebook."""
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths[:top_k]:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(image_matrix[0], image_matrix[1], images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= image_matrix[0] * image_matrix[1]:
                break


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


def display_query_and_multimodal_response(
    query_str: str, response: Response, plot_height: int = 2, plot_width: int = 5
) -> None:
    """For displaying a query and its multi-modal response."""
    if response.metadata:
        image_nodes = response.metadata["image_nodes"] or []
    else:
        image_nodes = []
    num_subplots = len(image_nodes)

    f, axarr = plt.subplots(1, num_subplots)
    f.set_figheight(plot_height)
    f.set_figwidth(plot_width)
    ix = 0
    for ix, scored_img_node in enumerate(image_nodes):
        img_node = scored_img_node.node
        image = None
        if img_node.image_url:
            img_response = requests.get(img_node.image_url)
            image = Image.open(BytesIO(img_response.content)).convert("RGB")
        elif img_node.image_path:
            image = Image.open(img_node.image_path).convert("RGB")
        else:
            raise ValueError(
                "A retrieved image must have image_path or image_url specified."
            )
        if num_subplots > 1:
            axarr[ix].imshow(image)
            axarr[ix].set_title(f"Retrieved Position: {ix}", pad=10, fontsize=9)
        else:
            axarr.imshow(image)
            axarr.set_title(f"Retrieved Position: {ix}", pad=10, fontsize=9)

    f.tight_layout()
    print(f"Query: {query_str}\n=======")
    print(f"Retrieved Images:\n")
    plt.show()
    print("=======")
    print(f"Response: {response.response}\n=======\n")
