"""
Metadata extractors for nodes.
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

from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, cast

from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.bridge.pydantic import (
    Field,
    PrivateAttr,
    SerializeAsAny,
)
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.settings import Settings
from llama_index.core.types import BasePydanticProgram, Model

DEFAULT_TITLE_NODE_TEMPLATE = """\
Context: {context_str}. Give a title that summarizes all of \
the unique entities, titles or themes found in the context. Title: """


DEFAULT_TITLE_COMBINE_TEMPLATE = """\
{context_str}. Based on the above candidate titles and content, \
what is the comprehensive title for this document? Title: """


def add_class_name(value: Any, handler: Callable, info: Any) -> Dict[str, Any]:
    partial_result = handler(value, info)
    if hasattr(value, "class_name"):
        partial_result.update({"class_name": value.class_name()})
    return partial_result


class TitleExtractor(BaseExtractor):
    """
    Title extractor. Useful for long documents. Extracts `document_title`
    metadata field.

    Args:
        llm (Optional[LLM]): LLM
        nodes (int): number of nodes from front to use for title extraction
        node_template (str): template for node-level title clues extraction
        combine_template (str): template for combining node-level clues into
            a document-level title

    """

    is_text_node_only: bool = False  # can work for mixture of text and non-text nodes
    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for generation.")
    nodes: int = Field(
        default=5,
        description="The number of nodes to extract titles from.",
        gt=0,
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
        llm_predictor: Optional[LLM] = None,
        nodes: int = 5,
        node_template: str = DEFAULT_TITLE_NODE_TEMPLATE,
        combine_template: str = DEFAULT_TITLE_COMBINE_TEMPLATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if nodes < 1:
            raise ValueError("num_nodes must be >= 1")

        super().__init__(
            llm=llm or llm_predictor or Settings.llm,
            nodes=nodes,
            node_template=node_template,
            combine_template=combine_template,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TitleExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        nodes_by_doc_id = self.separate_nodes_by_ref_id(nodes)
        titles_by_doc_id = await self.extract_titles(nodes_by_doc_id)
        return [{"document_title": titles_by_doc_id[node.ref_doc_id]} for node in nodes]

    def filter_nodes(self, nodes: Sequence[BaseNode]) -> List[BaseNode]:
        filtered_nodes: List[BaseNode] = []
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                continue
            filtered_nodes.append(node)
        return filtered_nodes

    def separate_nodes_by_ref_id(self, nodes: Sequence[BaseNode]) -> Dict:
        separated_items: Dict[Optional[str], List[BaseNode]] = {}

        for node in nodes:
            key = node.ref_doc_id
            if key not in separated_items:
                separated_items[key] = []

            if len(separated_items[key]) < self.nodes:
                separated_items[key].append(node)

        return separated_items

    async def extract_titles(self, nodes_by_doc_id: Dict) -> Dict:
        titles_by_doc_id = {}
        for key, nodes in nodes_by_doc_id.items():
            title_candidates = await self.get_title_candidates(nodes)
            combined_titles = ", ".join(title_candidates)
            titles_by_doc_id[key] = await self.llm.apredict(
                PromptTemplate(template=self.combine_template),
                context_str=combined_titles,
            )
        return titles_by_doc_id

    async def get_title_candidates(self, nodes: List[BaseNode]) -> List[str]:
        title_jobs = [
            self.llm.apredict(
                PromptTemplate(template=self.node_template),
                context_str=cast(TextNode, node).text,
            )
            for node in nodes
        ]
        return await run_jobs(
            title_jobs, show_progress=self.show_progress, workers=self.num_workers
        )


DEFAULT_KEYWORD_EXTRACT_TEMPLATE = """\
{context_str}. Give {keywords} unique keywords for this \
document. Format as comma separated. Keywords: """


class KeywordExtractor(BaseExtractor):
    """
    Keyword extractor. Node-level extractor. Extracts
    `excerpt_keywords` metadata field.

    Args:
        llm (Optional[LLM]): LLM
        keywords (int): number of keywords to extract
        prompt_template (str): template for keyword extraction

    """

    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for generation.")
    keywords: int = Field(
        default=5, description="The number of keywords to extract.", gt=0
    )

    prompt_template: str = Field(
        default=DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
        description="Prompt template to use when generating keywords.",
    )

    def __init__(
        self,
        llm: Optional[LLM] = None,
        # TODO: llm_predictor arg is deprecated
        llm_predictor: Optional[LLM] = None,
        keywords: int = 5,
        prompt_template: str = DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if keywords < 1:
            raise ValueError("num_keywords must be >= 1")

        super().__init__(
            llm=llm or llm_predictor or Settings.llm,
            keywords=keywords,
            prompt_template=prompt_template,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "KeywordExtractor"

    async def _aextract_keywords_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract keywords from a node and return it's metadata dict."""
        if self.is_text_node_only and not isinstance(node, TextNode):
            return {}

        context_str = node.get_content(metadata_mode=self.metadata_mode)
        keywords = await self.llm.apredict(
            PromptTemplate(template=self.prompt_template),
            keywords=self.keywords,
            context_str=context_str,
        )

        return {"excerpt_keywords": keywords.strip()}

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        keyword_jobs = []
        for node in nodes:
            keyword_jobs.append(self._aextract_keywords_from_node(node))

        metadata_list: List[Dict] = await run_jobs(
            keyword_jobs, show_progress=self.show_progress, workers=self.num_workers
        )

        return metadata_list


DEFAULT_QUESTION_GEN_TMPL = """\
Here is the context:
{context_str}

Given the contextual information, \
generate {num_questions} questions this context can provide \
specific answers to which are unlikely to be found elsewhere.

Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer.

"""


class QuestionsAnsweredExtractor(BaseExtractor):
    """
    Questions answered extractor. Node-level extractor.
    Extracts `questions_this_excerpt_can_answer` metadata field.

    Args:
        llm (Optional[LLM]): LLM
        questions (int): number of questions to extract
        prompt_template (str): template for question extraction,
        embedding_only (bool): whether to use embedding only

    """

    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for generation.")
    questions: int = Field(
        default=5,
        description="The number of questions to generate.",
        gt=0,
    )
    prompt_template: str = Field(
        default=DEFAULT_QUESTION_GEN_TMPL,
        description="Prompt template to use when generating questions.",
    )
    embedding_only: bool = Field(
        default=True, description="Whether to use metadata for emebddings only."
    )

    def __init__(
        self,
        llm: Optional[LLM] = None,
        # TODO: llm_predictor arg is deprecated
        llm_predictor: Optional[LLM] = None,
        questions: int = 5,
        prompt_template: str = DEFAULT_QUESTION_GEN_TMPL,
        embedding_only: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if questions < 1:
            raise ValueError("questions must be >= 1")

        super().__init__(
            llm=llm or llm_predictor or Settings.llm,
            questions=questions,
            prompt_template=prompt_template,
            embedding_only=embedding_only,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "QuestionsAnsweredExtractor"

    async def _aextract_questions_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract questions from a node and return it's metadata dict."""
        if self.is_text_node_only and not isinstance(node, TextNode):
            return {}

        context_str = node.get_content(metadata_mode=self.metadata_mode)
        prompt = PromptTemplate(template=self.prompt_template)
        questions = await self.llm.apredict(
            prompt, num_questions=self.questions, context_str=context_str
        )

        return {"questions_this_excerpt_can_answer": questions.strip()}

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        questions_jobs = []
        for node in nodes:
            questions_jobs.append(self._aextract_questions_from_node(node))

        metadata_list: List[Dict] = await run_jobs(
            questions_jobs, show_progress=self.show_progress, workers=self.num_workers
        )

        return metadata_list


DEFAULT_SUMMARY_EXTRACT_TEMPLATE = """\
Here is the content of the section:
{context_str}

Summarize the key topics and entities of the section. \

Summary: """


class SummaryExtractor(BaseExtractor):
    """
    Summary extractor. Node-level extractor with adjacent sharing.
    Extracts `section_summary`, `prev_section_summary`, `next_section_summary`
    metadata fields.

    Args:
        llm (Optional[LLM]): LLM
        summaries (List[str]): list of summaries to extract: 'self', 'prev', 'next'
        prompt_template (str): template for summary extraction

    """

    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for generation.")
    summaries: List[str] = Field(
        description="List of summaries to extract: 'self', 'prev', 'next'"
    )
    prompt_template: str = Field(
        default=DEFAULT_SUMMARY_EXTRACT_TEMPLATE,
        description="Template to use when generating summaries.",
    )

    _self_summary: bool = PrivateAttr()
    _prev_summary: bool = PrivateAttr()
    _next_summary: bool = PrivateAttr()

    def __init__(
        self,
        llm: Optional[LLM] = None,
        # TODO: llm_predictor arg is deprecated
        llm_predictor: Optional[LLM] = None,
        summaries: List[str] = ["self"],
        prompt_template: str = DEFAULT_SUMMARY_EXTRACT_TEMPLATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ):
        # validation
        if not all(s in ["self", "prev", "next"] for s in summaries):
            raise ValueError("summaries must be one of ['self', 'prev', 'next']")

        super().__init__(
            llm=llm or llm_predictor or Settings.llm,
            summaries=summaries,
            prompt_template=prompt_template,
            num_workers=num_workers,
            **kwargs,
        )

        self._self_summary = "self" in summaries
        self._prev_summary = "prev" in summaries
        self._next_summary = "next" in summaries

    @classmethod
    def class_name(cls) -> str:
        return "SummaryExtractor"

    async def _agenerate_node_summary(self, node: BaseNode) -> str:
        """Generate a summary for a node."""
        if self.is_text_node_only and not isinstance(node, TextNode):
            return ""

        context_str = node.get_content(metadata_mode=self.metadata_mode)
        summary = await self.llm.apredict(
            PromptTemplate(template=self.prompt_template), context_str=context_str
        )

        return summary.strip()

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        if not all(isinstance(node, TextNode) for node in nodes):
            raise ValueError("Only `TextNode` is allowed for `Summary` extractor")

        node_summaries_jobs = []
        for node in nodes:
            node_summaries_jobs.append(self._agenerate_node_summary(node))

        node_summaries = await run_jobs(
            node_summaries_jobs,
            show_progress=self.show_progress,
            workers=self.num_workers,
        )

        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        for i, metadata in enumerate(metadata_list):
            if i > 0 and self._prev_summary and node_summaries[i - 1]:
                metadata["prev_section_summary"] = node_summaries[i - 1]
            if i < len(nodes) - 1 and self._next_summary and node_summaries[i + 1]:
                metadata["next_section_summary"] = node_summaries[i + 1]
            if self._self_summary and node_summaries[i]:
                metadata["section_summary"] = node_summaries[i]

        return metadata_list


DEFAULT_ENTITY_MAP = {
    "PER": "persons",
    "ORG": "organizations",
    "LOC": "locations",
    "ANIM": "animals",
    "BIO": "biological",
    "CEL": "celestial",
    "DIS": "diseases",
    "EVE": "events",
    "FOOD": "foods",
    "INST": "instruments",
    "MEDIA": "media",
    "PLANT": "plants",
    "MYTH": "mythological",
    "TIME": "times",
    "VEHI": "vehicles",
}

DEFAULT_ENTITY_MODEL = "tomaarsen/span-marker-mbert-base-multinerd"


DEFAULT_EXTRACT_TEMPLATE_STR = """\
Here is the content of the section:
----------------
{context_str}
----------------
Given the contextual information, extract out a {class_name} object.\
"""


class PydanticProgramExtractor(BaseExtractor, Generic[Model]):
    """
    Pydantic program extractor.

    Uses an LLM to extract out a Pydantic object. Return attributes of that object
    in a dictionary.

    """

    program: SerializeAsAny[BasePydanticProgram[Model]] = Field(
        ..., description="Pydantic program to extract."
    )
    input_key: str = Field(
        default="input",
        description=(
            "Key to use as input to the program (the program "
            "template string must expose this key)."
        ),
    )
    extract_template_str: str = Field(
        default=DEFAULT_EXTRACT_TEMPLATE_STR,
        description="Template to use for extraction.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "PydanticModelExtractor"

    async def _acall_program(self, node: BaseNode) -> Dict[str, Any]:
        """Call the program on a node."""
        if self.is_text_node_only and not isinstance(node, TextNode):
            return {}

        extract_str = self.extract_template_str.format(
            context_str=node.get_content(metadata_mode=self.metadata_mode),
            class_name=self.program.output_cls.__name__,
        )

        ret_object = await self.program.acall(**{self.input_key: extract_str})
        assert not isinstance(ret_object, list)

        return ret_object.model_dump()

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extract pydantic program."""
        program_jobs = []
        for node in nodes:
            program_jobs.append(self._acall_program(node))

        metadata_list: List[Dict] = await run_jobs(
            program_jobs, show_progress=self.show_progress, workers=self.num_workers
        )

        return metadata_list
