from typing import Any, Iterable, Sequence

from llama_index.core.schema import Document as LIDocument
from llama_index.core.node_parser import NodeParser

from docling_core.transforms.chunker import BaseChunker, HierarchicalChunker
from docling_core.types import Document as DLDocument
from llama_index.core import Document as LIDocument
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.node_utils import IdFuncCallable, default_id_func
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    RelatedNodeType,
    TextNode,
)
from llama_index.core.utils import get_tqdm_iterable

_NODE_TEXT_KEY = "text"


class DoclingNodeParser(NodeParser):
    """Docling format node parser.

    Splits the JSON format of `DoclingReader` into nodes corresponding
    to respective document elements from Docling's data model
    (paragraphs, headings, tables etc.).

    Args:
        chunker (BaseChunker, optional): The chunker to use. Defaults to `HierarchicalChunker(heading_as_metadata=True)`.
        doc_meta_keys_allowed (set[str], optional): The Document metadata keys allowed to be included for embedding and LLM input. Defaults to `set()`.
        node_meta_keys_allowed (set[str], optional): The Node metadata keys allowed to be included for embedding and LLM input. Defaults to `{"heading"}`.
    """

    chunker: BaseChunker = HierarchicalChunker(heading_as_metadata=True)
    doc_meta_keys_allowed: set[str] = set()
    node_meta_keys_allowed: set[str] = {"heading"}

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:
        id_func: IdFuncCallable = self.id_func or default_id_func
        nodes_with_progress: Iterable[BaseNode] = get_tqdm_iterable(
            items=nodes, show_progress=show_progress, desc="Parsing nodes"
        )
        all_nodes: list[BaseNode] = []
        for input_node in nodes_with_progress:
            li_doc = LIDocument.model_validate(input_node)
            dl_doc: DLDocument = DLDocument.model_validate_json(li_doc.get_content())
            chunk_iter = self.chunker.chunk(dl_doc=dl_doc)
            for i, chunk in enumerate(chunk_iter):
                rels: dict[NodeRelationship, RelatedNodeType] = {
                    NodeRelationship.SOURCE: li_doc.as_related_node_info(),
                }
                metadata = chunk.model_dump(
                    exclude=_NODE_TEXT_KEY,
                    exclude_none=True,
                )
                # by default we exclude all meta keys from embedding/LLM — unless allowed
                excl_meta_keys = [
                    k for k in metadata if k not in self.node_meta_keys_allowed
                ]
                if self.include_metadata:
                    excl_meta_keys = [
                        k
                        for k in li_doc.metadata
                        if k not in self.doc_meta_keys_allowed
                    ] + excl_meta_keys
                node = TextNode(
                    id_=id_func(i=i, doc=li_doc),
                    text=chunk.text,
                    excluded_embed_metadata_keys=excl_meta_keys,
                    excluded_llm_metadata_keys=excl_meta_keys,
                    relationships=rels,
                )
                node.metadata = metadata
                all_nodes.append(node)
        return all_nodes
