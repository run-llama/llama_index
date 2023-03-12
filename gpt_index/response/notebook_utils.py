from typing import Any, Dict

from IPython.display import Markdown, display

from gpt_index.img_utils import b64_2_img
from gpt_index.response.schema import Response, SourceNode
from gpt_index.utils import truncate_text

DEFAULT_THUMBNAIL_SIZE = (512, 512)


def display_source_node(source_node: SourceNode, source_length: int = 100):
    display(Markdown(f"**Document ID:** {source_node.doc_id}"))
    source_text_fmt = truncate_text(source_node.source_text.strip(), source_length)
    display(Markdown(f"**Node Text:** {source_text_fmt}"))
    if source_node.image is not None:
        img = b64_2_img(source_node.image)
        img.thumbnail(DEFAULT_THUMBNAIL_SIZE)
        display(img)


def display_extra_info(extra_info: Dict[str, Any]):
    display(extra_info)


def display_response(response: Response, source_length: int = 100):
    display(Markdown(f"**`Final Response:`** {response.response.strip()}"))
    for ind, source_node in enumerate(response.source_nodes):
        display(Markdown("---"))
        display(Markdown(f"**`Source Node {ind + 1}/{len(response.source_nodes)}`**"))
        display_source_node(source_node, source_length=source_length)
    if response.extra_info is not None:
        display_extra_info(response.extra_info)
