from typing import List

from llama_index.response.schema import Response
from llama_index.schema import Document


def get_context(response: Response) -> List[Document]:
    """Get context information from given Response object using source nodes.

    Args:
        response (Response): Response object from an index based on the query.

    Returns:
        List of Documents of source nodes information as context information.
    """

    return [
        Document(text=source_node.get_content())
        for source_node in response.source_nodes
    ]
