"""JSON node parser."""
import json
from typing import Dict, Generator, List, Optional, Sequence

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.node_utils import build_nodes_from_splits
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.utils import get_tqdm_iterable


class JSONNodeParser(NodeParser):
    """JSON node parser.

    Splits a document into Nodes using custom JSON splitting logic.

    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    include_metadata: bool = Field(
        default=True, description="Whether or not to consider metadata when splitting."
    )
    include_prev_next_rel: bool = Field(
        default=True, description="Include prev/next node relationships."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

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

    def get_nodes_from_documents(
        self,
        documents: Sequence[TextNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse document into nodes.

        Args:
            documents (Sequence[TextNode]): TextNodes or Documents to parse

        """
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: List[BaseNode] = []
            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )

            for document in documents_with_progress:
                nodes = self.get_nodes_from_node(document)
                all_nodes.extend(nodes)

            event.on_end(payload={EventPayload.NODES: all_nodes})

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
            json_nodes.append(self._build_node_from_split("\n".join(lines), node, {}))
        elif isinstance(data, list):
            for json_object in data:
                lines = [*self._depth_first_yield(json_object, 0, [])]
                json_nodes.append(
                    self._build_node_from_split("\n".join(lines), node, {})
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

    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        metadata: dict,
    ) -> TextNode:
        """Build node from single text split."""
        node = build_nodes_from_splits(
            [text_split], node, self.include_metadata, self.include_prev_next_rel
        )[0]

        if self.include_metadata:
            node.metadata = {**node.metadata, **metadata}

        return node
