"""Node parser interface."""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Sequence

from llama_index.core.bridge.pydantic import Field, validator
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.schema import (
    BaseNode,
    Document,
    MetadataMode,
    NodeRelationship,
    TransformComponent,
)
from llama_index.core.utils import get_tqdm_iterable


class NodeParser(TransformComponent, ABC):
    """Base interface for node parser."""

    include_metadata: bool = Field(
        default=True, description="Whether or not to consider metadata when splitting."
    )
    include_prev_next_rel: bool = Field(
        default=True, description="Include prev/next node relationships."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )
    id_func: Callable = Field(
        default=None,
        description="Function to generate node IDs.",
        exclude=True,
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("id_func", pre=True)
    def _validate_id_func(cls, v: Any) -> Any:
        if v is None:
            return default_id_func
        return v

    @abstractmethod
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        ...

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse documents into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            show_progress (bool): whether to show progress bar

        """
        doc_id_to_document = {doc.id_: doc for doc in documents}

        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            nodes = self._parse_nodes(documents, show_progress=show_progress, **kwargs)

            for i, node in enumerate(nodes):
                if (
                    node.ref_doc_id is not None
                    and node.ref_doc_id in doc_id_to_document
                ):
                    ref_doc = doc_id_to_document[node.ref_doc_id]
                    start_char_idx = ref_doc.text.find(
                        node.get_content(metadata_mode=MetadataMode.NONE)
                    )

                    # update start/end char idx
                    if start_char_idx >= 0:
                        node.start_char_idx = start_char_idx
                        node.end_char_idx = start_char_idx + len(
                            node.get_content(metadata_mode=MetadataMode.NONE)
                        )

                    # update metadata
                    if self.include_metadata:
                        node.metadata.update(
                            doc_id_to_document[node.ref_doc_id].metadata
                        )

                if self.include_prev_next_rel:
                    if i > 0:
                        node.relationships[NodeRelationship.PREVIOUS] = nodes[
                            i - 1
                        ].as_related_node_info()
                    if i < len(nodes) - 1:
                        node.relationships[NodeRelationship.NEXT] = nodes[
                            i + 1
                        ].as_related_node_info()

            event.on_end({EventPayload.NODES: nodes})

        return nodes

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        return self.get_nodes_from_documents(nodes, **kwargs)


class TextSplitter(NodeParser):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        ...

    def split_texts(self, texts: List[str]) -> List[str]:
        nested_texts = [self.split_text(text) for text in texts]
        return [item for sublist in nested_texts for item in sublist]

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for node in nodes_with_progress:
            splits = self.split_text(node.get_content())

            all_nodes.extend(
                build_nodes_from_splits(splits, node, id_func=self.id_func)
            )

        return all_nodes


class MetadataAwareTextSplitter(TextSplitter):
    @abstractmethod
    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        ...

    def split_texts_metadata_aware(
        self, texts: List[str], metadata_strs: List[str]
    ) -> List[str]:
        if len(texts) != len(metadata_strs):
            raise ValueError("Texts and metadata_strs must have the same length")
        nested_texts = [
            self.split_text_metadata_aware(text, metadata)
            for text, metadata in zip(texts, metadata_strs)
        ]
        return [item for sublist in nested_texts for item in sublist]

    def _get_metadata_str(self, node: BaseNode) -> str:
        """Helper function to get the proper metadata str for splitting."""
        embed_metadata_str = node.get_metadata_str(mode=MetadataMode.EMBED)
        llm_metadata_str = node.get_metadata_str(mode=MetadataMode.LLM)

        # use the longest metadata str for splitting
        if len(embed_metadata_str) > len(llm_metadata_str):
            metadata_str = embed_metadata_str
        else:
            metadata_str = llm_metadata_str

        return metadata_str

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            metadata_str = self._get_metadata_str(node)
            splits = self.split_text_metadata_aware(
                node.get_content(metadata_mode=MetadataMode.NONE),
                metadata_str=metadata_str,
            )
            all_nodes.extend(
                build_nodes_from_splits(splits, node, id_func=self.id_func)
            )

        return all_nodes
