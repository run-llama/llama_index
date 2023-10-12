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
from copy import deepcopy
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.llm_predictor.base import BaseLLMPredictor, LLMPredictor
from llama_index.llms.base import LLM
from llama_index.node_parser.interface import BaseExtractor
from llama_index.prompts import PromptTemplate
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.types import BasePydanticProgram
from llama_index.utils import get_tqdm_iterable


class MetadataFeatureExtractor(BaseExtractor):
    is_text_node_only: bool = True
    show_progress: bool = True
    metadata_mode: MetadataMode = MetadataMode.ALL

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

    extractors: Sequence[MetadataFeatureExtractor] = Field(
        default_factory=list,
        description="Metadta feature extractors to apply to each node.",
    )
    node_text_template: str = Field(
        default=DEFAULT_NODE_TEXT_TEMPLATE,
        description="Template to represent how node text is mixed with metadata text.",
    )
    disable_template_rewrite: bool = Field(
        default=False, description="Disable the node template rewrite."
    )

    in_place: bool = Field(
        default=True, description="Whether to process nodes in place."
    )

    @classmethod
    def class_name(cls) -> str:
        return "MetadataExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extract metadata from a document.

        Args:
            nodes (Sequence[BaseNode]): nodes to extract metadata from

        """
        metadata_list: List[Dict] = [{} for _ in nodes]
        for extractor in self.extractors:
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
        if self.in_place:
            new_nodes = nodes
        else:
            new_nodes = [deepcopy(node) for node in nodes]
        for extractor in self.extractors:
            cur_metadata_list = extractor.extract(new_nodes)
            for idx, node in enumerate(new_nodes):
                node.metadata.update(cur_metadata_list[idx])

        for idx, node in enumerate(new_nodes):
            if excluded_embed_metadata_keys is not None:
                node.excluded_embed_metadata_keys.extend(excluded_embed_metadata_keys)
            if excluded_llm_metadata_keys is not None:
                node.excluded_llm_metadata_keys.extend(excluded_llm_metadata_keys)
            if not self.disable_template_rewrite:
                if isinstance(node, TextNode):
                    cast(TextNode, node).text_template = self.node_text_template
        return new_nodes


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

    is_text_node_only: bool = False  # can work for mixture of text and non-text nodes
    llm_predictor: BaseLLMPredictor = Field(
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
        llm_predictor: Optional[BaseLLMPredictor] = None,
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

        title_candidates = [
            self.llm_predictor.predict(
                PromptTemplate(template=self.node_template),
                context_str=cast(TextNode, node).text,
            )
            for node in nodes_to_extract_title
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

        return [{"document_title": title.strip(' \t\n\r"')} for _ in nodes]


class KeywordExtractor(MetadataFeatureExtractor):
    """Keyword extractor. Node-level extractor. Extracts
    `excerpt_keywords` metadata field.

    Args:
        llm_predictor (Optional[BaseLLMPredictor]): LLM predictor
        keywords (int): number of keywords to extract
    """

    llm_predictor: BaseLLMPredictor = Field(
        description="The LLMPredictor to use for generation."
    )
    keywords: int = Field(default=5, description="The number of keywords to extract.")

    def __init__(
        self,
        llm: Optional[LLM] = None,
        # TODO: llm_predictor arg is deprecated
        llm_predictor: Optional[BaseLLMPredictor] = None,
        keywords: int = 5,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if keywords < 1:
            raise ValueError("num_keywords must be >= 1")

        if llm is not None:
            llm_predictor = LLMPredictor(llm=llm)
        elif llm_predictor is None and llm is None:
            llm_predictor = LLMPredictor()

        super().__init__(llm_predictor=llm_predictor, keywords=keywords, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "KeywordExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        metadata_list: List[Dict] = []
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue

            # TODO: figure out a good way to allow users to customize keyword template
            keywords = self.llm_predictor.predict(
                PromptTemplate(
                    template=f"""\
{{context_str}}. Give {self.keywords} unique keywords for this \
document. Format as comma separated. Keywords: """
                ),
                context_str=cast(TextNode, node).text,
            )
            # node.metadata["excerpt_keywords"] = keywords
            metadata_list.append({"excerpt_keywords": keywords.strip()})
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


class QuestionsAnsweredExtractor(MetadataFeatureExtractor):
    """
    Questions answered extractor. Node-level extractor.
    Extracts `questions_this_excerpt_can_answer` metadata field.

    Args:
        llm_predictor (Optional[BaseLLMPredictor]): LLM predictor
        questions (int): number of questions to extract
        prompt_template (str): template for question extraction,
        embedding_only (bool): whether to use embedding only
    """

    llm_predictor: BaseLLMPredictor = Field(
        description="The LLMPredictor to use for generation."
    )
    questions: int = Field(
        default=5, description="The number of questions to generate."
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
        llm_predictor: Optional[BaseLLMPredictor] = None,
        questions: int = 5,
        prompt_template: str = DEFAULT_QUESTION_GEN_TMPL,
        embedding_only: bool = True,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if questions < 1:
            raise ValueError("questions must be >= 1")

        if llm is not None:
            llm_predictor = LLMPredictor(llm=llm)
        elif llm_predictor is None and llm is None:
            llm_predictor = LLMPredictor()

        super().__init__(
            llm_predictor=llm_predictor,
            questions=questions,
            prompt_template=prompt_template,
            embedding_only=embedding_only,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "QuestionsAnsweredExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        metadata_list: List[Dict] = []
        nodes_queue = get_tqdm_iterable(
            nodes, self.show_progress, "Extracting questions"
        )
        for node in nodes_queue:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue

            context_str = node.get_content(metadata_mode=self.metadata_mode)
            prompt = PromptTemplate(template=self.prompt_template)
            questions = self.llm_predictor.predict(
                prompt, num_questions=self.questions, context_str=context_str
            )

            if self.embedding_only:
                node.excluded_llm_metadata_keys = ["questions_this_excerpt_can_answer"]
            metadata_list.append(
                {"questions_this_excerpt_can_answer": questions.strip()}
            )
        return metadata_list


DEFAULT_SUMMARY_EXTRACT_TEMPLATE = """\
Here is the content of the section:
{context_str}

Summarize the key topics and entities of the section. \

Summary: """


class SummaryExtractor(MetadataFeatureExtractor):
    """
    Summary extractor. Node-level extractor with adjacent sharing.
    Extracts `section_summary`, `prev_section_summary`, `next_section_summary`
    metadata fields.

    Args:
        llm_predictor (Optional[BaseLLMPredictor]): LLM predictor
        summaries (List[str]): list of summaries to extract: 'self', 'prev', 'next'
        prompt_template (str): template for summary extraction
    """

    llm_predictor: BaseLLMPredictor = Field(
        description="The LLMPredictor to use for generation."
    )
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
        llm_predictor: Optional[BaseLLMPredictor] = None,
        summaries: List[str] = ["self"],
        prompt_template: str = DEFAULT_SUMMARY_EXTRACT_TEMPLATE,
        **kwargs: Any,
    ):
        if llm is not None:
            llm_predictor = LLMPredictor(llm=llm)
        elif llm_predictor is None and llm is None:
            llm_predictor = LLMPredictor()

        # validation
        if not all(s in ["self", "prev", "next"] for s in summaries):
            raise ValueError("summaries must be one of ['self', 'prev', 'next']")
        self._self_summary = "self" in summaries
        self._prev_summary = "prev" in summaries
        self._next_summary = "next" in summaries

        super().__init__(
            llm_predictor=llm_predictor,
            summaries=summaries,
            prompt_template=prompt_template,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SummaryExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        if not all(isinstance(node, TextNode) for node in nodes):
            raise ValueError("Only `TextNode` is allowed for `Summary` extractor")
        nodes_queue = get_tqdm_iterable(
            nodes, self.show_progress, "Extracting summaries"
        )
        node_summaries = []
        for node in nodes_queue:
            node_context = cast(TextNode, node).get_content(
                metadata_mode=self.metadata_mode
            )
            summary = self.llm_predictor.predict(
                PromptTemplate(template=self.prompt_template),
                context_str=node_context,
            ).strip()
            node_summaries.append(summary)

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


class EntityExtractor(MetadataFeatureExtractor):
    """
    Entity extractor. Extracts `entities` into a metadata field using a default model
    `tomaarsen/span-marker-mbert-base-multinerd` and the SpanMarker library.

    Install SpanMarker with `pip install span-marker`.
    """

    model_name: str = Field(
        default=DEFAULT_ENTITY_MODEL,
        description="The model name of the SpanMarker model to use.",
    )
    prediction_threshold: float = Field(
        default=0.5, description="The confidence threshold for accepting predictions."
    )
    span_joiner: str = Field(description="The separator between entity names.")
    label_entities: bool = Field(
        default=False, description="Include entity class labels or not."
    )
    device: Optional[str] = Field(
        default=None, description="Device to run model on, i.e. 'cuda', 'cpu'"
    )
    entity_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of entity class names to usable names.",
    )

    _tokenizer: Callable = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_ENTITY_MODEL,
        prediction_threshold: float = 0.5,
        span_joiner: str = " ",
        label_entities: bool = False,
        device: Optional[str] = None,
        entity_map: Optional[Dict[str, str]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        **kwargs: Any,
    ):
        """
        Entity extractor for extracting entities from text and inserting
        into node metadata.

        Args:
            model_name (str):
                Name of the SpanMarker model to use.
            prediction_threshold (float):
                Minimum prediction threshold for entities. Defaults to 0.5.
            span_joiner (str):
                String to join spans with. Defaults to " ".
            label_entities (bool):
                Whether to label entities with their type. Setting to true can be
                slightly error prone, but can be useful for downstream tasks.
                Defaults to False.
            device (Optional[str]):
                Device to use for SpanMarker model, i.e. "cpu" or "cuda".
                Loads onto "cpu" by default.
            entity_map (Optional[Dict[str, str]]):
                Mapping from entity class name to label.
            tokenizer (Optional[Callable[[str], List[str]]]):
                Tokenizer to use for splitting text into words.
                Defaults to NLTK word_tokenize.
        """
        try:
            from span_marker import SpanMarkerModel
        except ImportError:
            raise ImportError(
                "SpanMarker is not installed. Install with `pip install span-marker`."
            )

        try:
            from nltk.tokenize import word_tokenize
        except ImportError:
            raise ImportError("NLTK is not installed. Install with `pip install nltk`.")

        self._model = SpanMarkerModel.from_pretrained(model_name)
        if device is not None:
            self._model = self._model.to(device)

        self._tokenizer = tokenizer or word_tokenize

        base_entity_map = DEFAULT_ENTITY_MAP
        if entity_map is not None:
            base_entity_map.update(entity_map)

        super().__init__(
            model_name=model_name,
            prediction_threshold=prediction_threshold,
            span_joiner=span_joiner,
            label_entities=label_entities,
            device=device,
            entity_map=base_entity_map,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "EntityExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        # Extract node-level entity metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        for i, metadata in enumerate(metadata_list):
            node_text = nodes[i].get_content(metadata_mode=self.metadata_mode)
            words = self._tokenizer(node_text)
            spans = self._model.predict(words)
            for span in spans:
                if span["score"] > self.prediction_threshold:
                    ent_label = self.entity_map.get(span["label"], span["label"])
                    metadata_label = ent_label if self.label_entities else "entities"

                    if metadata_label not in metadata:
                        metadata[metadata_label] = set()

                    metadata[metadata_label].add(self.span_joiner.join(span["span"]))

        # convert metadata from set to list
        for metadata in metadata_list:
            for key, val in metadata.items():
                metadata[key] = list(val)

        return metadata_list


DEFAULT_EXTRACT_TEMPLATE_STR = """\
Here is the content of the section:
----------------
{context_str}
----------------
Given the contextual information, extract out a {class_name} object.\
"""


class PydanticProgramExtractor(MetadataFeatureExtractor):
    """Pydantic program extractor.

    Uses an LLM to extract out a Pydantic object. Return attributes of that object
    in a dictionary.

    """

    program: BasePydanticProgram = Field(
        ..., description="Pydantic program to extract."
    )
    input_key: str = Field(
        default="input",
        description=(
            "Key to use as input to the program (the program "
            "template string must expose this key).",
        ),
    )
    extract_template_str: str = Field(
        default=DEFAULT_EXTRACT_TEMPLATE_STR,
        description="Template to use for extraction.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "PydanticModelExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extract pydantic program."""
        metadata_list: List[Dict] = []
        nodes_queue = get_tqdm_iterable(
            nodes, self.show_progress, "Extracting Pydantic object"
        )
        for node in nodes_queue:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue
            extract_str = self.extract_template_str.format(
                context_str=node.get_content(metadata_mode=self.metadata_mode),
                class_name=self.program.output_cls.__name__,
            )

            object = self.program(**{self.input_key: extract_str})
            fields_and_values = object.dict()
            metadata_list.append(fields_and_values)

        return metadata_list
