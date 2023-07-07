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

from abc import ABC, abstractmethod
import json
from typing import Any, Callable, List, Optional, Sequence, cast
from functools import reduce

from llama_index.llm_predictor.base import BaseLLMPredictor, LLMPredictor
from llama_index.node_parser.interface import NodeParserPostProcessor
from llama_index.prompts.base import Prompt
from llama_index.schema import BaseNode, TextNode


class MetadataExtractorBase(ABC):
    is_text_node_only = True

    @abstractmethod
    def __call__(self, nodes: Sequence[BaseNode]) -> None:
        """Extracts metadata for a sequence of nodes and mutates the nodes in place.

        Args:
            nodes (Sequence[Document]): nodes to extract metadata from

        """


class MetadataExtractor:
    """Factory for metadata extractors.

    Example:

    ```
    post_processor = MetadataExtractor(llm_predictor=llm_predictor)
        .extract_title(nodes=3)
        .extract_questions_answered(questions=3)
        .extract_keywords(
            node_template="Context: {context_str}. 10 Keywords: "
        )
        .extract_summary(summaries=['self', 'prev', 'next'])
        .extract(lambda nodes:
            for node in nodes:
                node.metadata["custom"]) = llm_predictor.predict(
                    prompt,
                    node.get_content()
                )
        )
        .finish()

    node_parser = SimpleNodeParser(..., post_processors=[post_processor])
    ```
    """

    def __init__(
        self,
        llm_predictor: Optional[BaseLLMPredictor],
        node_text_template: Optional[str] = None,
        disable_template_rewrite: bool = False,
    ) -> None:
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._extractors: List[Callable[[Sequence[BaseNode]], None]] = []
        self._node_text_template = node_text_template
        self._disable_template_rewrite = disable_template_rewrite

    def extract_title(self, **kwargs: Any) -> "MetadataExtractor":
        """Extract document title for the sequence of nodes.
        Args:
            nodes (int): number of nodes (from front of sequence)
                to use for title extraction
            node_template (str): template for node-level title
                clues extraction
            combine_template (str): template for combining node-level
                clues into a document-level title
        """
        self._extractors.append(
            TitleExtractor(llm_predictor=self._llm_predictor, **kwargs)
        )
        return self

    def extract_questions_answered(self, **kwargs: Any) -> "MetadataExtractor":
        """Extract questions answered by the node.
        Args:
            questions (int): number of questions to extract
            prompt_template (str): template for question extraction
        """
        self._extractors.append(
            QuestionsAnsweredExtractor(llm_predictor=self._llm_predictor, **kwargs)
        )
        return self

    def extract_keywords(self, **kwargs: Any) -> "MetadataExtractor":
        """Extract keywords for the node.
        Args:
            keywords (int): number of keywords to extract
        """
        self._extractors.append(
            KeywordExtractor(llm_predictor=self._llm_predictor, **kwargs)
        )
        return self

    def extract_summary(self, **kwargs: Any) -> "MetadataExtractor":
        """Extract summary for the node.
        Args:
            summaries (List[str]): list of summaries to extract: "self", "prev", "next"
            prompt_template (str): template for summary extraction"""
        self._extractors.append(
            SummaryExtractor(llm_predictor=self._llm_predictor, **kwargs)
        )
        return self

    def extract(
        self, extractor: Callable[[Sequence[BaseNode]], None]
    ) -> "MetadataExtractor":
        """Extract metadata using a custom extractor.
        Args:
            extractor (Callable[[Sequence[BaseNode]], None]):
                Callable that takes a sequence of nodes and mutates the
                metadata of the nodes in place. It can be a custom class with an
                  appropriate __call__ or a simply a function/lambda function.
        """
        self._extractors.append(extractor)
        return self

    def build(self) -> "MetadataAugmentationPostProcessor":
        return MetadataAugmentationPostProcessor(
            self._extractors, self._node_text_template, self._disable_template_rewrite
        )


class MetadataAugmentationPostProcessor(NodeParserPostProcessor):
    def __init__(
        self,
        extractors: Sequence[Callable[[Sequence[BaseNode]], None]],
        node_text_template: Optional[str] = None,
        disable_template_rewrite: bool = False,
    ) -> None:
        self._extractors = extractors
        self._node_text_template = (
            node_text_template
            or """\
            [Excerpt from document]\n{metadata_str}\n\
            Excerpt:\n-----\n{content}\n-----\n"""
        )
        self._disable_template_rewrite = disable_template_rewrite

    def post_process_nodes(self, nodes: Sequence[BaseNode]) -> None:
        """Extract metadata from a document.

        Args:
            nodes (Sequence[BaseNode]): nodes to extract metadata from

        """
        if not self._disable_template_rewrite:
            for node in nodes:
                if isinstance(node, TextNode):
                    cast(TextNode, node).text_template = self._node_text_template
        for extractor in self._extractors:
            extractor(nodes)


class TitleExtractor(MetadataExtractorBase):
    """Title extractor. Useful for long documents. Extracts `document_title`
    metadata field.
    Args:
        nodes (int): number of nodes from front to use for title extraction
        node_template (str): template for node-level title clues extraction
        combine_template (str): template for combining node-level clues into
            a document-level title
    """

    def __init__(
        self,
        llm_predictor: BaseLLMPredictor,
        nodes: int = 5,
        node_template: Optional[str] = None,
        combine_template: Optional[str] = None,
    ) -> None:
        """Init params."""
        if nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        self._nodes = nodes
        self._node_template = node_template
        self._combine_template = combine_template
        self._llm_predictor = llm_predictor

    def __call__(self, nodes: Sequence[BaseNode]) -> None:
        nodes_to_extract_title: List[BaseNode] = []
        for node in nodes:
            if len(nodes_to_extract_title) >= self._nodes:
                break
            if self.is_text_node_only and not isinstance(node, TextNode):
                continue
            nodes_to_extract_title.append(node)

        if len(nodes_to_extract_title) == 0:
            # Could not extract title
            return

        title_candidates = [
            self._llm_predictor.predict(
                Prompt(
                    template=self._node_template
                    or """\
Context: {context_str}. Give a title that summarizes all of \
the unique entities, titles or themes found in the context. Title: \
                    """
                ),
                context_str=cast(TextNode, node).text,
            )
            for node in nodes_to_extract_title
        ]
        if len(nodes_to_extract_title) > 1:
            titles = reduce(
                lambda x, y: x + "," + y, title_candidates[1:], title_candidates[0]
            )

            title = self._llm_predictor.predict(
                Prompt(
                    template=self._combine_template
                    or """\
{context_str}. Based on the above candidate titles and content, \
what is the comprehensive title for this document? Title: \
                    """
                ),
                context_str=titles,
            )
        else:
            title = title_candidates[
                0
            ]  # if single node, just use the title from that node

        for node in nodes:
            node.metadata["document_title"] = title.strip(' \t\n\r"')


class KeywordExtractor(MetadataExtractorBase):
    """Keyword extractor. Node-level extractor. Extracts
    `excerpt_keywords` metadata field.
    Args:
        llm_predictor (BaseLLMPredictor): LLM predictor
        keywords (int): number of keywords to extract
    """

    def __init__(
        self,
        llm_predictor: BaseLLMPredictor,
        keywords: int = 5,
    ) -> None:
        """Init params."""
        self._llm_predictor = llm_predictor
        if keywords < 1:
            raise ValueError("num_keywords must be >= 1")
        self._keywords = keywords

    def __call__(self, nodes: Sequence[BaseNode]) -> None:
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                continue
            keywords = self._llm_predictor.predict(
                Prompt(
                    template=f"""\
{{context_str}}. Give {self._keywords} unique keywords for this \
document. Format as comma separated. Keywords: """
                ),
                context_str=cast(TextNode, node).text,
            )
            node.metadata["excerpt_keywords"] = keywords


class QuestionsAnsweredExtractor(MetadataExtractorBase):
    """
    Questions answered extractor. Node-level extractor.
    Extracts `questions_this_excerpt_can_answer` metadata field.
    Args:
        llm_predictor (BaseLLMPredictor): LLM predictor
        questions (int): number of questions to extract
        prompt_template (str): template for question extraction
    """

    def __init__(
        self,
        llm_predictor: BaseLLMPredictor,
        questions: int = 5,
        prompt_template: Optional[str] = None,
    ) -> None:
        """Init params."""
        if questions < 1:
            raise ValueError("questions must be >= 1")
        self._llm_predictor = llm_predictor
        self._questions = questions
        self._prompt_template = prompt_template

    def __call__(self, nodes: Sequence[BaseNode]) -> None:
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                continue
            # Extract the title from the first node
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
            node.metadata["questions_this_excerpt_can_answer"] = questions
            # Only use this for the embedding
            node.excluded_llm_metadata_keys = ["questions_this_excerpt_can_answer"]


class SummaryExtractor(MetadataExtractorBase):
    """
    Summary extractor. Node-level extractor with adjacent sharing.
    Extracts `section_summary`, `prev_section_summary`, `next_section_summary`
    metadata fields
    Args:
        llm_predictor (BaseLLMPredictor): LLM predictor
        summaries (List[str]): list of summaries to extract: 'self', 'prev', 'next'
        prompt_template (str): template for summary extraction"""

    def __init__(
        self,
        llm_predictor: BaseLLMPredictor,
        summaries: List[str] = ["self"],
        prompt_template: Optional[str] = None,
    ):
        self._llm_predictor = llm_predictor
        # validation
        if not all([s in ["self", "prev", "next"] for s in summaries]):
            raise ValueError("summaries must be one of ['self', 'prev', 'next']")
        self._self_summary = "self" in summaries
        self._prev_summary = "prev" in summaries
        self._next_summary = "next" in summaries
        self._prompt_template = prompt_template

    def __call__(self, nodes: Sequence[BaseNode]) -> None:
        node_summaries = [
            self._llm_predictor.predict(
                Prompt(
                    template=self._prompt_template
                    or """\
Here is the content of the section: {context_str}. \
Summarize the key topics and entities of the section. Summary: \
                    """
                ),
                context_str=cast(TextNode, node).text,
            )
            for node in nodes
        ]

        if self._embedding_only:
            excluded_llm_metadata_keys = []
            if self._self_summary:
                excluded_llm_metadata_keys.append("section_summary")
            if self._prev_summary:
                excluded_llm_metadata_keys.append("prev_section_summary")
            if self._next_summary:
                excluded_llm_metadata_keys.append("next_section_summary")

        # Extract node-level summary metadata
        for i, node in enumerate(nodes):
            if i > 0 and self._prev_summary:
                node.metadata["prev_section_summary"] = node_summaries[i - 1]
            if i < len(nodes) - 1 and self._next_summary:
                node.metadata["next_section_summary"] = node_summaries[i + 1]
            if self._self_summary:
                node.metadata["section_summary"] = node_summaries[i]
            if self._embedding_only:
                node.excluded_llm_metadata_keys = excluded_llm_metadata_keys
