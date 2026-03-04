from typing import Any, Iterable, Protocol, Sequence, runtime_checkable
import uuid

from llama_index.core.schema import Document as LIDocument
from llama_index.core.node_parser import NodeParser

from docling_core.transforms.chunker import BaseChunker, HierarchicalChunker
from docling_core.types import DoclingDocument as DLDocument
from llama_index.core import Document as LIDocument
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    RelatedNodeType,
    TextNode,
)
from llama_index.core.utils import get_tqdm_iterable


class DoclingNodeParser(NodeParser):
    """
    Docling format node parser.

    Splits the JSON format of `DoclingReader` into nodes corresponding
    to respective document elements from Docling's data model
    (paragraphs, headings, tables etc.).

    Args:
        chunker (BaseChunker, optional): The chunker to use. Defaults to `HierarchicalChunker()`.
        id_func(NodeIDGenCallable, optional): The node ID generation function to use. Defaults to `_uuid4_node_id_gen`.

    """

    @runtime_checkable
    class NodeIDGenCallable(Protocol):
        def __call__(self, i: int, node: BaseNode) -> str: ...

    @staticmethod
    def _uuid4_node_id_gen(i: int, node: BaseNode) -> str:
        return str(uuid.uuid4())

    chunker: BaseChunker = HierarchicalChunker()
    id_func: NodeIDGenCallable = _uuid4_node_id_gen

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:
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
                metadata = chunk.meta.export_json_dict()
                excl_embed_keys = [
                    k for k in chunk.meta.excluded_embed if k in metadata
                ]
                excl_llm_keys = [k for k in chunk.meta.excluded_llm if k in metadata]

                excl_embed_keys.extend(
                    [
                        k
                        for k in li_doc.excluded_embed_metadata_keys
                        if k not in excl_embed_keys
                    ]
                )
                excl_llm_keys.extend(
                    [
                        k
                        for k in li_doc.excluded_llm_metadata_keys
                        if k not in excl_llm_keys
                    ]
                )

                node = TextNode(
                    id_=self.id_func(i=i, node=li_doc),
                    text=chunk.text,
                    excluded_embed_metadata_keys=excl_embed_keys,
                    excluded_llm_metadata_keys=excl_llm_keys,
                    relationships=rels,
                )
                node.metadata = metadata
                all_nodes.append(node)
        return all_nodes
