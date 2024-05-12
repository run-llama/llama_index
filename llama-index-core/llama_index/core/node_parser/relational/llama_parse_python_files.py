try:
  from pychunk.chunkers.python_chunker import PythonChunker
  from pychunk.parser.llama_index_parser import LlamaIndexParser
  from pychunk.nodes.base import BaseNode as PychunkBaseNode
except ModuleNotFoundError as :
  raise e("You need to install pychunk first: pip install rag-pychunk")

from llama_index.core.schema import NodeRelationship, BaseNode
from llama_index.core.node_parser.interface import NodeParser


class LlamaParsePythonFile(NodeParser):

  def get_nodes_from_python_files(self, files: List[str]) -> List[BaseNode]:
    ...
    
  def _create_relationsihps_of_nodes(self, nodes: List[PyhunkBaseNode]) -> List[BaseNode]:
    ...

llama_index_parser = LlamaIndexParser(nodes=list(all_nodes.values()))
llama_index_nodes = llama_index_parser.parse_to_llama_index()
