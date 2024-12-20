from pathlib import Path
import re
from typing import Any, Dict, Sequence, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.langchain_helpers.agents import LlamaIndexTool

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import BaseNode
from llama_index.packs.code_hierarchy.code_hierarchy import CodeHierarchyNodeParser


DEFAULT_TOOL_INSTRUCTIONS = (
    "Search the tool by any element in this list to get more information about that element.\n"
    "If you see 'Code replaced for brevity' then a uuid, you may also search the tool with that uuid to see the full code.\n"
    "You may need to use the tool multiple times to fully answer the user message.\n"
    "The list is:\n"
    "{repo_map}\n"
)


class CodeHierarchyKeywordQueryEngine(CustomQueryEngine):
    """A keyword table made specifically to work with the code hierarchy node parser."""

    nodes: Sequence[BaseNode]
    node_dict: Optional[Dict[str, Tuple[int, BaseNode]]] = None
    repo_map_depth: int = -1
    include_repo_map: bool = True
    repo_map: Optional[Tuple[Dict[str, Any], str]] = None
    tool_instructions: str = DEFAULT_TOOL_INSTRUCTIONS

    def _setup_node_dict(self) -> None:
        """Initialize the index."""
        self.node_dict = {}
        for node in self.nodes:
            keys = self._extract_keywords_from_node(node)
            for key in keys:
                self.node_dict[key] = (node.metadata["start_byte"], node.text)
        self.repo_map = CodeHierarchyNodeParser.get_code_hierarchy_from_nodes(
            self.nodes, max_depth=self.repo_map_depth
        )

    def _extract_keywords_from_node(self, node: BaseNode) -> Set[str]:
        """Determine the keywords associated with the node in the index."""
        keywords = self._extract_uuid_from_node(node)
        keywords |= self._extract_module_from_node(node)
        keywords |= self._extract_name_from_node(node)
        return keywords

    def _extract_uuid_from_node(self, node: BaseNode) -> Set[str]:
        """Extract the uuid from the node."""
        return {node.id_}

    def _extract_module_from_node(self, node: BaseNode) -> Set[str]:
        """Extract the module name from the node."""
        keywords = set()
        if not node.metadata["inclusive_scopes"]:
            path = Path(node.metadata["filepath"])
            name = path.name
            name = re.sub(r"\..*$", "", name)
            if name in self.node_dict:
                its_start_byte, _ = self.node_dict[name]
                if node.metadata["start_byte"] < its_start_byte:
                    keywords.add(name)
            else:
                keywords.add(name)
        return keywords

    def _extract_name_from_node(self, node: BaseNode) -> Set[str]:
        """Extract the name and signature from the node."""
        keywords = set()
        if node.metadata["inclusive_scopes"]:
            name = node.metadata["inclusive_scopes"][-1]["name"]
            start_byte = node.metadata["start_byte"]
            if name in self.node_dict:
                its_start_byte, _ = self.node_dict[name]
                if start_byte < its_start_byte:
                    keywords.add(name)
            else:
                keywords.add(name)
        return keywords

    def custom_query(self, query: str) -> str:
        """Query the index. Only use exact matches.
        If there is no exact match, but there is one for a parent, returns the parent.
        """
        if self.node_dict is None or self.repo_map is None:
            self._setup_node_dict()

        def get_all_dict_recursive(inp: Dict[str, Any]) -> Set[str]:
            """Get all keys and values from a dictionary of dictionaries recursively."""
            kvs = set()
            for key, value in inp.items():
                kvs.add(key)
                if isinstance(value, dict):
                    kvs |= get_all_dict_recursive(value)
                else:
                    kvs.add(value)
            return kvs

        def get_parent_dict_recursive(inp: Dict[str, Any], query: str) -> str:
            """Get the parent of a key in a dictionary of dictionaries recursively."""
            for key, value in inp.items():
                if isinstance(value, dict):
                    if query in value:
                        return key
                    else:
                        parent = get_parent_dict_recursive(value, query)
                        if parent is not None:
                            return parent
            return None

        if query in self.node_dict:
            return self.node_dict[query][1]

        kvs = get_all_dict_recursive(self.repo_map[0])
        if query not in kvs:
            return None
        parent_query = query
        while parent_query not in self.node_dict:
            parent_query = get_parent_dict_recursive(self.repo_map[0], parent_query)
            if parent_query is None:
                return "None"

        # After finding the parent_query, ensure it's in self.node_dict before accessing
        if parent_query in self.node_dict:
            return self.node_dict[parent_query][1]
        else:
            return "None"

    def get_tool_instructions(self) -> str:
        """Get the tool instructions."""
        if self.node_dict is None or self.repo_map is None:
            self._setup_node_dict()
        return self.tool_instructions.format(
            repo_map=self.repo_map[1] if self.include_repo_map else ""
        )

    def as_langchain_tool(
        self,
        **tool_kwargs: Any,
    ) -> "LlamaIndexTool":
        """
        Return the index as a langchain tool.
        Set a repo map depth of -1 to include all nodes.
        otherwise set the depth to the desired max depth.
        """
        from llama_index.core.langchain_helpers.agents import LlamaIndexTool

        if self.node_dict is None or self.repo_map is None:
            self._setup_node_dict()
        return LlamaIndexTool(
            name="Code Search",
            description=self.get_tool_instructions(),
            query_engine=self,
            **tool_kwargs,
        )
