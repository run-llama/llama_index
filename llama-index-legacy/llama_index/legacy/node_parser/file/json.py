"""JSON node parser."""

import json
from typing import Any, Dict, Generator, List, Optional, Sequence

from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.node_parser.interface import NodeParser
from llama_index.legacy.node_parser.node_utils import build_nodes_from_splits
from llama_index.legacy.schema import BaseNode, MetadataMode, TextNode
from llama_index.legacy.utils import get_tqdm_iterable


class JSONNodeParser(NodeParser):
    """JSON node parser.

    Splits a document into Nodes using custom JSON splitting logic.

    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "JSONNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "JSONNodeParser"

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node)
            all_nodes.extend(nodes)

        return all_nodes

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Get nodes from document."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Handle invalid JSON input here
            return []

        json_nodes = []
        if isinstance(data, dict):
            lines = [*self._depth_first_yield(data, 0, [])]
            json_nodes.extend(
                build_nodes_from_splits(["\n".join(lines)], node, id_func=self.id_func)
            )
        elif isinstance(data, list):
            for json_object in data:
                lines = [*self._depth_first_yield(json_object, 0, [])]
                json_nodes.extend(
                    build_nodes_from_splits(
                        ["\n".join(lines)], node, id_func=self.id_func
                    )
                )
        else:
            raise ValueError("JSON is invalid")

        return json_nodes

    def _depth_first_yield(
        self, json_data: Dict, levels_back: int, path: List[str]
    ) -> Generator[str, None, None]:
        """Do depth first yield of all of the leaf nodes of a JSON.

        Combines keys in the JSON tree using spaces.

        If levels_back is set to 0, prints all levels.

        """
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                new_path = path[:]
                new_path.append(key)
                yield from self._depth_first_yield(value, levels_back, new_path)
        elif isinstance(json_data, list):
            for _, value in enumerate(json_data):
                yield from self._depth_first_yield(value, levels_back, path)
        else:
            new_path = path[-levels_back:]
            new_path.append(str(json_data))
            yield " ".join(new_path)
