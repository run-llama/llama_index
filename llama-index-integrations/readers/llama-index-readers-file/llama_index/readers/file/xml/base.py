"""JSON Reader."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


def _get_leaf_nodes_up_to_level(root: ET.Element, level: int) -> List[ET.Element]:
    """Get collection of nodes up to certain level including leaf nodes.

    Args:
        root (ET.Element): XML Root Element
        level (int): Levels to traverse in the tree

    Returns:
        List[ET.Element]: List of target nodes
    """

    def traverse(current_node, current_level):
        if len(current_node) == 0 or level == current_level:
            # Keep leaf nodes and target level nodes
            nodes.append(current_node)
        elif current_level < level:
            # Move to the next level
            for child in current_node:
                traverse(child, current_level + 1)

    nodes = []
    traverse(root, 0)
    return nodes


class XMLReader(BaseReader):
    """XML reader.

    Reads XML documents with options to help suss out relationships between nodes.

    Args:
        tree_level_split (int): From which level in the xml tree we split documents,
        the default level is the root which is level 0

    """

    def __init__(self, tree_level_split: Optional[int] = 0) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.tree_level_split = tree_level_split

    def _parse_xmlelt_to_document(
        self, root: ET.Element, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse the xml object into a list of Documents.

        Args:
            root: The XML Element to be converted.
            extra_info (Optional[Dict]): Additional information. Default is None.

        Returns:
            Document: The documents.
        """
        nodes = _get_leaf_nodes_up_to_level(root, self.tree_level_split)
        documents = []
        for node in nodes:
            content = ET.tostring(node, encoding="utf8").decode("utf-8")
            content = re.sub(r"^<\?xml.*", "", content)
            content = content.strip()
            documents.append(Document(text=content, extra_info=extra_info or {}))

        return documents

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Load data from the input file.

        Args:
            file (Path): Path to the input file.
            extra_info (Optional[Dict]): Additional information. Default is None.

        Returns:
            List[Document]: List of documents.
        """
        if not isinstance(file, Path):
            file = Path(file)

        tree = ET.parse(file)
        return self._parse_xmlelt_to_document(tree.getroot(), extra_info)
