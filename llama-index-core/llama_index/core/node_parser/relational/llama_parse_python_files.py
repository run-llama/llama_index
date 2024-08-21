from typing import List

try:
    from pychunk.chunkers.python_chunker import PythonChunker
    from pychunk.parser.llama_index_parser import LlamaIndexParser
except ModuleNotFoundError as e:
    raise e("You need to install pychunk first: pip install rag-pychunk")

from llama_index.core.schema import NodeRelationship, BaseNode, RelatedNodeInfo
from llama_index.core.node_parser.interface import NodeParser


class LlamaParsePythonFile(NodeParser):
    
    def get_nodes_from_python_files(self, files: List[str]) -> List[BaseNode]:
        """Convert pychunk nodes into LlamaIndex BaseNode"""
        _chunker = PythonChunker(files_path=files)
        _nodes = _chunker.find_relationships()
        nodes = []
        for _, nodes_of_file in _nodes.items():
          nodes.extend(list(nodes_of_file.values()))
          
        base_nodes = LlamaIndexParser(nodes=nodes).parse_to_llama_index()

        for node in base_nodes:
            metadata = node.metadata
            other_relationships = metadata.get('other_relationships')
            if other_relationships is None or len(other_relationships) < 1:
                continue
            node.relationships[NodeRelationship.REFERENCE] = [RelatedNodeInfo(node_id=id) for id in other_relationships]

        return base_nodes
      
    def _parse_nodes(self):
      ...
