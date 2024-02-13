from pathlib import Path
from typing import Dict, Sequence, Set, Tuple
from llama_index.core import PromptTemplate
from llama_index.core.langchain_helpers.agents import LlamaIndexTool
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import BaseNode
import re
from llama_index.packs.code_hierarchy.code_hierarchy import CodeHierarchyNodeParser


class CodeHierarchyKeywordQueryEngine(CustomQueryEngine):
    """A keyword table made specifically to work with the code hierarchy node parser."""

    nodes: Sequence[BaseNode]
    index: Dict[str, Tuple[int, BaseNode]] | None = None
    tool_instructions: PromptTemplate = PromptTemplate(
        template="""
        Search the tool by any element in this list,
        or any uuid found in the code,
        to get more information about that element.

        {repo_map}
        """
    )

    def _setup_index(
        self,
    ) -> None:
        """Initialize the index."""
        self.index = {}
        for node in self.nodes:
            keys = self._extract_keywords_from_node(node)
            for key in keys:
                self.index[key] = (node.metadata["start_byte"], node.text)

    def _extract_keywords_from_node(self, node: BaseNode) -> Set[str]:
        """Determine the keywords associated with the node in the index."""
        keywords = self._extract_uuid_from_node(node)
        keywords |= self._extract_module_from_node(node)
        keywords |= self._extract_name_from_node(node)
        return keywords

    def _extract_uuid_from_node(self, node) -> Set[str]:
        """Extract the uuid from the node."""
        return {node.node_id}

    def _extract_module_from_node(self, node) -> Set[str]:
        """Extract the module name from the node."""
        keywords = set()
        if not node.metadata["inclusive_scopes"]:
            path = Path(node.metadata["filepath"])
            name = path.name
            name = re.sub(r"\..*$", "", name)
            if name in self.index:
                its_start_byte, _ = self.index[name]
                if node.metadata["start_byte"] < its_start_byte:
                    keywords.add(name)
            else:
                keywords.add(name)
        return keywords

    def _extract_name_from_node(self, node) -> Set[str]:
        """Extract the name and signature from the node."""
        keywords = set()
        if node.metadata["inclusive_scopes"]:
            name = node.metadata["inclusive_scopes"][-1]["name"]
            start_byte = node.metadata["start_byte"]
            if name in self.index:
                its_start_byte, _ = self.index[name]
                if start_byte < its_start_byte:
                    keywords.add(name)
            else:
                keywords.add(name)
        return keywords

    def custom_query(self, query: str) -> str:
        """Query the index. Only use exact matches."""
        if self.index is None:
            self._setup_index()
        return self.index.get(str(query), (0, ""))[1]

    def as_langchain_tool(
        self,
        include_repo_map: bool = True,
        repo_map_depth: int = -1,
        **tool_kwargs,
    ) -> LlamaIndexTool:
        """
        Return the index as a langchain tool.
        Set a repo map depth of -1 to include all nodes.
        otherwise set the depth to the desired max depth.
        """
        repo_map = {}
        if include_repo_map:
            repo_map = CodeHierarchyNodeParser.get_code_hierarchy_from_nodes(
                self.nodes, max_depth=repo_map_depth
            )
        return LlamaIndexTool(
            name="Code Search",
            description=self.tool_instructions.format(repo_map=repo_map),
            query_engine=self,
            **tool_kwargs,
        )
