"""
Metadata extractors for nodes. Applied as a post processor to node parsing.
Currently, only `TextNode` is supported.

Supported metadata:
Node-level:
    - `SummaryExtractor`: Summary of each node, and pre and post nodes
    - `QuestionsAnsweredExtractor`: Questions that the node can answer
    - `KeywordsExtractor`: Keywords that uniquely identify the node
Document-level:
    - `TitleExtractor`: Document title, possible inferred across multiple nodes

Unimplemented (contributions welcome):
Subsection:
    - Position of node in subsection hierarchy (and associated subtitles)
    - Hierarchically organized summary

The prompts used to generate the metadata are specifically aimed to help
disambiguate the document or subsection from other similar documents or subsections.
(similar with contrastive learning)
"""
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

from llama_index.bridge.pydantic import Field

from llama_index.llm_predictor.base import LLMPredictor
from llama_index.llms.base import LLM
from llama_index.extractors.interface import BaseExtractor
from llama_index.prompts import PromptTemplate
from llama_index.schema import BaseNode, TextNode
from llama_index.utils import get_tqdm_iterable


DEFAULT_TITLE_NODE_TEMPLATE = """\
Context: {context_str}. Give a title that summarizes all of \
the unique entities, titles or themes found in the context. Title: """


DEFAULT_TITLE_COMBINE_TEMPLATE = """\
{context_str}. Based on the above candidate titles and content, \
what is the comprehensive title for this document? Title: """


class TitleExtractor(BaseExtractor):
    """Title extractor. Useful for long documents. Extracts `document_title`
    metadata field.
    Args:
        llm_predictor (Optional[LLMPredictor]): LLM predictor
        nodes (int): number of nodes from front to use for title extraction
        node_template (str): template for node-level title clues extraction
        combine_template (str): template for combining node-level clues into
            a document-level title
    """

    is_text_node_only: bool = False  # can work for mixture of text and non-text nodes
    llm_predictor: LLMPredictor = Field(
        description="The LLMPredictor to use for generation."
    )
    nodes: int = Field(
        default=5, description="The number of nodes to extract titles from."
    )
    node_template: str = Field(
        default=DEFAULT_TITLE_NODE_TEMPLATE,
        description="The prompt template to extract titles with.",
    )
    combine_template: str = Field(
        default=DEFAULT_TITLE_COMBINE_TEMPLATE,
        description="The prompt template to merge titles with.",
    )

    def __init__(
        self,
        llm: Optional[LLM] = None,
        # TODO: llm_predictor arg is deprecated
        llm_predictor: Optional[LLMPredictor] = None,
        nodes: int = 5,
        node_template: str = DEFAULT_TITLE_NODE_TEMPLATE,
        combine_template: str = DEFAULT_TITLE_COMBINE_TEMPLATE,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if nodes < 1:
            raise ValueError("num_nodes must be >= 1")

        if llm is not None:
            llm_predictor = LLMPredictor(llm=llm)
        elif llm_predictor is None and llm is None:
            llm_predictor = LLMPredictor()

        super().__init__(
            llm_predictor=llm_predictor,
            nodes=nodes,
            node_template=node_template,
            combine_template=combine_template,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TitleExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        nodes_to_extract_title: List[BaseNode] = []

        for node in nodes:
            if len(nodes_to_extract_title) >= self.nodes:
                break
            if self.is_text_node_only and not isinstance(node, TextNode):
                continue
            nodes_to_extract_title.append(node)

        if len(nodes_to_extract_title) == 0:
            # Could not extract title
            return []

        nodes_queue: Iterable[BaseNode] = get_tqdm_iterable(
            nodes_to_extract_title, self.show_progress, "Extracting titles"
        )
        title_candidates = [
            self.llm_predictor.predict(
                PromptTemplate(template=self.node_template),
                context_str=cast(TextNode, node).text,
            )
            for node in nodes_queue
        ]
        if len(nodes_to_extract_title) > 1:
            titles = reduce(
                lambda x, y: x + "," + y, title_candidates[1:], title_candidates[0]
            )

            title = self.llm_predictor.predict(
                PromptTemplate(template=self.combine_template),
                context_str=titles,
            )
        else:
            title = title_candidates[
                0
            ]  # if single node, just use the title from that node

        metadata_list = [{"document_title": title.strip(' \t\n\r"')} for _ in nodes]
        return metadata_list
