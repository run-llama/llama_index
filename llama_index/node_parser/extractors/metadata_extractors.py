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

from abc import abstractmethod
import json
from typing import List, Optional, Sequence, cast, Dict
from functools import reduce

from llama_index.llm_predictor.base import BaseLLMPredictor, LLMPredictor
from llama_index.node_parser.interface import BaseExtractor
from llama_index.prompts.base import Prompt
from llama_index.schema import BaseNode, TextNode


class MetadataFeatureExtractor(BaseExtractor):
    is_text_node_only = True

    @abstractmethod
    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extracts metadata for a sequence of nodes, returning a list of
        metadata dictionaries corresponding to each node.

        Args:
            nodes (Sequence[Document]): nodes to extract metadata from

        """


DEFAULT_NODE_TEXT_TEMPLATE = """\
[Excerpt from document]\n{metadata_str}\n\
Excerpt:\n-----\n{content}\n-----\n"""


class MetadataExtractor(BaseExtractor):
    """Metadata extractor."""

    def __init__(
        self,
        extractors: Sequence[MetadataFeatureExtractor],
        node_text_template: str = DEFAULT_NODE_TEXT_TEMPLATE,
        disable_template_rewrite: bool = False,
    ) -> None:
        self._extractors = extractors
        self._node_text_template = node_text_template
        self._disable_template_rewrite = disable_template_rewrite

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extract metadata from a document.

        Args:
            nodes (Sequence[BaseNode]): nodes to extract metadata from

        """
        metadata_list: List[Dict] = [{} for _ in nodes]
        for extractor in self._extractors:
            cur_metadata_list = extractor.extract(nodes)
            for i, metadata in enumerate(metadata_list):
                metadata.update(cur_metadata_list[i])

        return metadata_list

    def process_nodes(
        self,
        nodes: List[BaseNode],
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
    ) -> List[BaseNode]:
        """Post process nodes parsed from documents.

        Allows extractors to be chained.

        Args:
            nodes (List[BaseNode]): nodes to post-process
            excluded_embed_metadata_keys (Optional[List[str]]):
                keys to exclude from embed metadata
            excluded_llm_metadata_keys (Optional[List[str]]):
                keys to exclude from llm metadata
        """
        for extractor in self._extractors:
            cur_metadata_list = extractor.extract(nodes)
            for idx, node in enumerate(nodes):
                node.metadata.update(cur_metadata_list[idx])

        for idx, node in enumerate(nodes):
            if excluded_embed_metadata_keys is not None:
                node.excluded_embed_metadata_keys.extend(excluded_embed_metadata_keys)
            if excluded_llm_metadata_keys is not None:
                node.excluded_llm_metadata_keys.extend(excluded_llm_metadata_keys)
            if not self._disable_template_rewrite:
                if isinstance(node, TextNode):
                    cast(TextNode, node).text_template = self._node_text_template
        return nodes


DEFAULT_TITLE_NODE_TEMPLATE = """\
Context: {context_str}. Give a title that summarizes all of \
the unique entities, titles or themes found in the context. Title: """


DEFAULT_TITLE_COMBINE_TEMPLATE = """\
{context_str}. Based on the above candidate titles and content, \
what is the comprehensive title for this document? Title: """


class TitleExtractor(MetadataFeatureExtractor):
    """Title extractor. Useful for long documents. Extracts `document_title`
    metadata field.
    Args:
        llm_predictor (Optional[BaseLLMPredictor]): LLM predictor
        nodes (int): number of nodes from front to use for title extraction
        node_template (str): template for node-level title clues extraction
        combine_template (str): template for combining node-level clues into
            a document-level title
    """

    is_text_node_only = False  # can work for mixture of text and non-text nodes

    def __init__(
        self,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        nodes: int = 5,
        node_template: str = DEFAULT_TITLE_NODE_TEMPLATE,
        combine_template: str = DEFAULT_TITLE_COMBINE_TEMPLATE,
    ) -> None:
        """Init params."""
        if nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        self._nodes = nodes
        self._node_template = node_template
        self._combine_template = combine_template
        self._llm_predictor = llm_predictor or LLMPredictor()

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        nodes_to_extract_title: List[BaseNode] = []
        for node in nodes:
            if len(nodes_to_extract_title) >= self._nodes:
                break
            if self.is_text_node_only and not isinstance(node, TextNode):
                continue
            nodes_to_extract_title.append(node)

        if len(nodes_to_extract_title) == 0:
            # Could not extract title
            return []

        title_candidates = [
            self._llm_predictor.predict(
                Prompt(template=self._node_template),
                context_str=cast(TextNode, node).text,
            )
            for node in nodes_to_extract_title
        ]
        if len(nodes_to_extract_title) > 1:
            titles = reduce(
                lambda x, y: x + "," + y, title_candidates[1:], title_candidates[0]
            )

            title = self._llm_predictor.predict(
                Prompt(template=self._combine_template),
                context_str=titles,
            )
        else:
            title = title_candidates[
                0
            ]  # if single node, just use the title from that node

        metadata_list = [{"document_title": title.strip(' \t\n\r"')} for node in nodes]
        return metadata_list


class KeywordExtractor(MetadataFeatureExtractor):
    """Keyword extractor. Node-level extractor. Extracts
    `excerpt_keywords` metadata field.
    Args:
        llm_predictor (Optional[BaseLLMPredictor]): LLM predictor
        keywords (int): number of keywords to extract
    """

    def __init__(
        self,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        keywords: int = 5,
    ) -> None:
        """Init params."""
        self._llm_predictor = llm_predictor or LLMPredictor()
        if keywords < 1:
            raise ValueError("num_keywords must be >= 1")
        self._keywords = keywords

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        metadata_list: List[Dict] = []
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue

            # TODO: figure out a good way to allow users to customize keyword template
            keywords = self._llm_predictor.predict(
                Prompt(
                    template=f"""\
{{context_str}}. Give {self._keywords} unique keywords for this \
document. Format as comma separated. Keywords: """
                ),
                context_str=cast(TextNode, node).text,
            )
            # node.metadata["excerpt_keywords"] = keywords
            metadata_list.append({"excerpt_keywords": keywords})
        return metadata_list


class QuestionsAnsweredExtractor(MetadataFeatureExtractor):
    """
    Questions answered extractor. Node-level extractor.
    Extracts `questions_this_excerpt_can_answer` metadata field.
    Args:
        llm_predictor (Optional[BaseLLMPredictor]): LLM predictor
        questions (int): number of questions to extract
        prompt_template (str): template for question extraction
    """

    def __init__(
        self,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        questions: int = 5,
        prompt_template: Optional[str] = None,
    ) -> None:
        """Init params."""
        if questions < 1:
            raise ValueError("questions must be >= 1")
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._questions = questions
        self._prompt_template = prompt_template

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        metadata_list: List[Dict] = []
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue
            # Extract the title from the first node
            # TODO: figure out a good way to allow users to customize template
            questions = self._llm_predictor.predict(
                Prompt(
                    template=self._prompt_template
                    or f"""\
{{context_str}}. Given the contextual information, \
generate {self._questions} questions this document can provide \
specific answers to which are unlikely to be found elsewhere: \
"""
                ),
                context_str=f"""\
metadata: {json.dumps(node.metadata)} \
content: {cast(TextNode, node).text}""",
            )
            # node.metadata["questions_this_excerpt_can_answer"] = questions
            # # Only use this for the embedding
            # node.excluded_llm_metadata_keys = ["questions_this_excerpt_can_answer"]
            metadata_list.append({"questions_this_excerpt_can_answer": questions})
        return metadata_list


DEFAULT_SUMMARY_EXTRACT_TEMPLATE = """\
Here is the content of the section: {context_str}. \
Summarize the key topics and entities of the section. Summary: """


class SummaryExtractor(MetadataFeatureExtractor):
    """
    Summary extractor. Node-level extractor with adjacent sharing.
    Extracts `section_summary`, `prev_section_summary`, `next_section_summary`
    metadata fields
    Args:
        llm_predictor (Optional[BaseLLMPredictor]): LLM predictor
        summaries (List[str]): list of summaries to extract: 'self', 'prev', 'next'
        prompt_template (str): template for summary extraction"""

    def __init__(
        self,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        summaries: List[str] = ["self"],
        prompt_template: str = DEFAULT_SUMMARY_EXTRACT_TEMPLATE,
    ):
        self._llm_predictor = llm_predictor or LLMPredictor()
        # validation
        if not all([s in ["self", "prev", "next"] for s in summaries]):
            raise ValueError("summaries must be one of ['self', 'prev', 'next']")
        self._self_summary = "self" in summaries
        self._prev_summary = "prev" in summaries
        self._next_summary = "next" in summaries
        self._prompt_template = prompt_template

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        if not all([isinstance(node, TextNode) for node in nodes]):
            raise ValueError("Only `TextNode` is allowed for `Summary` extractor")
        node_summaries = [
            self._llm_predictor.predict(
                Prompt(template=self._prompt_template),
                context_str=cast(TextNode, node).text,
            )
            for node in nodes
        ]

        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        for i, metadata in enumerate(metadata_list):
            if i > 0 and self._prev_summary:
                metadata["prev_section_summary"] = node_summaries[i - 1]
            if i < len(nodes) - 1 and self._next_summary:
                metadata["next_section_summary"] = node_summaries[i + 1]
            if self._self_summary:
                metadata["section_summary"] = node_summaries[i]

        return metadata_list
