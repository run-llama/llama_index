"""Node postprocessor."""

import logging
from typing import Dict, List, Optional, cast

from llama_index.legacy.bridge.pydantic import Field, validator
from llama_index.legacy.postprocessor.types import BaseNodePostprocessor
from llama_index.legacy.prompts.base import PromptTemplate
from llama_index.legacy.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.legacy.schema import NodeRelationship, NodeWithScore, QueryBundle
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.storage.docstore import BaseDocumentStore

logger = logging.getLogger(__name__)


class KeywordNodePostprocessor(BaseNodePostprocessor):
    """Keyword-based Node processor."""

    required_keywords: List[str] = Field(default_factory=list)
    exclude_keywords: List[str] = Field(default_factory=list)
    lang: str = Field(default="en")

    @classmethod
    def class_name(cls) -> str:
        return "KeywordNodePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "Spacy is not installed, please install it with `pip install spacy`."
            )
        from spacy.matcher import PhraseMatcher

        nlp = spacy.blank(self.lang)
        required_matcher = PhraseMatcher(nlp.vocab)
        exclude_matcher = PhraseMatcher(nlp.vocab)
        required_matcher.add("RequiredKeywords", list(nlp.pipe(self.required_keywords)))
        exclude_matcher.add("ExcludeKeywords", list(nlp.pipe(self.exclude_keywords)))

        new_nodes = []
        for node_with_score in nodes:
            node = node_with_score.node
            doc = nlp(node.get_content())
            if self.required_keywords and not required_matcher(doc):
                continue
            if self.exclude_keywords and exclude_matcher(doc):
                continue
            new_nodes.append(node_with_score)

        return new_nodes


class SimilarityPostprocessor(BaseNodePostprocessor):
    """Similarity-based Node processor."""

    similarity_cutoff: float = Field(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "SimilarityPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        sim_cutoff_exists = self.similarity_cutoff is not None

        new_nodes = []
        for node in nodes:
            should_use_node = True
            if sim_cutoff_exists:
                similarity = node.score
                if similarity is None:
                    should_use_node = False
                elif cast(float, similarity) < cast(float, self.similarity_cutoff):
                    should_use_node = False

            if should_use_node:
                new_nodes.append(node)

        return new_nodes


def get_forward_nodes(
    node_with_score: NodeWithScore, num_nodes: int, docstore: BaseDocumentStore
) -> Dict[str, NodeWithScore]:
    """Get forward nodes."""
    node = node_with_score.node
    nodes: Dict[str, NodeWithScore] = {node.node_id: node_with_score}
    cur_count = 0
    # get forward nodes in an iterative manner
    while cur_count < num_nodes:
        if NodeRelationship.NEXT not in node.relationships:
            break

        next_node_info = node.next_node
        if next_node_info is None:
            break

        next_node_id = next_node_info.node_id
        next_node = docstore.get_node(next_node_id)
        nodes[next_node.node_id] = NodeWithScore(node=next_node)
        node = next_node
        cur_count += 1
    return nodes


def get_backward_nodes(
    node_with_score: NodeWithScore, num_nodes: int, docstore: BaseDocumentStore
) -> Dict[str, NodeWithScore]:
    """Get backward nodes."""
    node = node_with_score.node
    # get backward nodes in an iterative manner
    nodes: Dict[str, NodeWithScore] = {node.node_id: node_with_score}
    cur_count = 0
    while cur_count < num_nodes:
        prev_node_info = node.prev_node
        if prev_node_info is None:
            break
        prev_node_id = prev_node_info.node_id
        prev_node = docstore.get_node(prev_node_id)
        if prev_node is None:
            break
        nodes[prev_node.node_id] = NodeWithScore(node=prev_node)
        node = prev_node
        cur_count += 1
    return nodes


class PrevNextNodePostprocessor(BaseNodePostprocessor):
    """Previous/Next Node post-processor.

    Allows users to fetch additional nodes from the document store,
    based on the relationships of the nodes.

    NOTE: this is a beta feature.

    Args:
        docstore (BaseDocumentStore): The document store.
        num_nodes (int): The number of nodes to return (default: 1)
        mode (str): The mode of the post-processor.
            Can be "previous", "next", or "both.

    """

    docstore: BaseDocumentStore
    num_nodes: int = Field(default=1)
    mode: str = Field(default="next")

    @validator("mode")
    def _validate_mode(cls, v: str) -> str:
        """Validate mode."""
        if v not in ["next", "previous", "both"]:
            raise ValueError(f"Invalid mode: {v}")
        return v

    @classmethod
    def class_name(cls) -> str:
        return "PrevNextNodePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        all_nodes: Dict[str, NodeWithScore] = {}
        for node in nodes:
            all_nodes[node.node.node_id] = node
            if self.mode == "next":
                all_nodes.update(get_forward_nodes(node, self.num_nodes, self.docstore))
            elif self.mode == "previous":
                all_nodes.update(
                    get_backward_nodes(node, self.num_nodes, self.docstore)
                )
            elif self.mode == "both":
                all_nodes.update(get_forward_nodes(node, self.num_nodes, self.docstore))
                all_nodes.update(
                    get_backward_nodes(node, self.num_nodes, self.docstore)
                )
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

        all_nodes_values: List[NodeWithScore] = list(all_nodes.values())
        sorted_nodes: List[NodeWithScore] = []
        for node in all_nodes_values:
            # variable to check if cand node is inserted
            node_inserted = False
            for i, cand in enumerate(sorted_nodes):
                node_id = node.node.node_id
                # prepend to current candidate
                prev_node_info = cand.node.prev_node
                next_node_info = cand.node.next_node
                if prev_node_info is not None and node_id == prev_node_info.node_id:
                    node_inserted = True
                    sorted_nodes.insert(i, node)
                    break
                # append to current candidate
                elif next_node_info is not None and node_id == next_node_info.node_id:
                    node_inserted = True
                    sorted_nodes.insert(i + 1, node)
                    break

            if not node_inserted:
                sorted_nodes.append(node)

        return sorted_nodes


DEFAULT_INFER_PREV_NEXT_TMPL = (
    "The current context information is provided. \n"
    "A question is also provided. \n"
    "You are a retrieval agent deciding whether to search the "
    "document store for additional prior context or future context. \n"
    "Given the context and question, return PREVIOUS or NEXT or NONE. \n"
    "Examples: \n\n"
    "Context: Describes the author's experience at Y Combinator."
    "Question: What did the author do after his time at Y Combinator? \n"
    "Answer: NEXT \n\n"
    "Context: Describes the author's experience at Y Combinator."
    "Question: What did the author do before his time at Y Combinator? \n"
    "Answer: PREVIOUS \n\n"
    "Context: Describe the author's experience at Y Combinator."
    "Question: What did the author do at Y Combinator? \n"
    "Answer: NONE \n\n"
    "Context: {context_str}\n"
    "Question: {query_str}\n"
    "Answer: "
)


DEFAULT_REFINE_INFER_PREV_NEXT_TMPL = (
    "The current context information is provided. \n"
    "A question is also provided. \n"
    "An existing answer is also provided.\n"
    "You are a retrieval agent deciding whether to search the "
    "document store for additional prior context or future context. \n"
    "Given the context, question, and previous answer, "
    "return PREVIOUS or NEXT or NONE.\n"
    "Examples: \n\n"
    "Context: {context_msg}\n"
    "Question: {query_str}\n"
    "Existing Answer: {existing_answer}\n"
    "Answer: "
)


class AutoPrevNextNodePostprocessor(BaseNodePostprocessor):
    """Previous/Next Node post-processor.

    Allows users to fetch additional nodes from the document store,
    based on the prev/next relationships of the nodes.

    NOTE: difference with PrevNextPostprocessor is that
    this infers forward/backwards direction.

    NOTE: this is a beta feature.

    Args:
        docstore (BaseDocumentStore): The document store.
        num_nodes (int): The number of nodes to return (default: 1)
        infer_prev_next_tmpl (str): The template to use for inference.
            Required fields are {context_str} and {query_str}.

    """

    docstore: BaseDocumentStore
    service_context: ServiceContext
    num_nodes: int = Field(default=1)
    infer_prev_next_tmpl: str = Field(default=DEFAULT_INFER_PREV_NEXT_TMPL)
    refine_prev_next_tmpl: str = Field(default=DEFAULT_REFINE_INFER_PREV_NEXT_TMPL)
    verbose: bool = Field(default=False)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def class_name(cls) -> str:
        return "AutoPrevNextNodePostprocessor"

    def _parse_prediction(self, raw_pred: str) -> str:
        """Parse prediction."""
        pred = raw_pred.strip().lower()
        if "previous" in pred:
            return "previous"
        elif "next" in pred:
            return "next"
        elif "none" in pred:
            return "none"
        raise ValueError(f"Invalid prediction: {raw_pred}")

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        if query_bundle is None:
            raise ValueError("Missing query bundle.")

        infer_prev_next_prompt = PromptTemplate(
            self.infer_prev_next_tmpl,
        )
        refine_infer_prev_next_prompt = PromptTemplate(self.refine_prev_next_tmpl)

        all_nodes: Dict[str, NodeWithScore] = {}
        for node in nodes:
            all_nodes[node.node.node_id] = node
            # use response builder instead of llm directly
            # to be more robust to handling long context
            response_builder = get_response_synthesizer(
                service_context=self.service_context,
                text_qa_template=infer_prev_next_prompt,
                refine_template=refine_infer_prev_next_prompt,
                response_mode=ResponseMode.TREE_SUMMARIZE,
            )
            raw_pred = response_builder.get_response(
                text_chunks=[node.node.get_content()],
                query_str=query_bundle.query_str,
            )
            raw_pred = cast(str, raw_pred)
            mode = self._parse_prediction(raw_pred)

            logger.debug(f"> Postprocessor Predicted mode: {mode}")
            if self.verbose:
                print(f"> Postprocessor Predicted mode: {mode}")

            if mode == "next":
                all_nodes.update(get_forward_nodes(node, self.num_nodes, self.docstore))
            elif mode == "previous":
                all_nodes.update(
                    get_backward_nodes(node, self.num_nodes, self.docstore)
                )
            elif mode == "none":
                pass
            else:
                raise ValueError(f"Invalid mode: {mode}")

        sorted_nodes = sorted(all_nodes.values(), key=lambda x: x.node.node_id)
        return list(sorted_nodes)


class LongContextReorder(BaseNodePostprocessor):
    """
    Models struggle to access significant details found
    in the center of extended contexts. A study
    (https://arxiv.org/abs/2307.03172) observed that the best
    performance typically arises when crucial data is positioned
    at the start or conclusion of the input context. Additionally,
    as the input context lengthens, performance drops notably, even
    in models designed for long contexts.".
    """

    @classmethod
    def class_name(cls) -> str:
        return "LongContextReorder"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        reordered_nodes: List[NodeWithScore] = []
        ordered_nodes: List[NodeWithScore] = sorted(
            nodes, key=lambda x: x.score if x.score is not None else 0
        )
        for i, node in enumerate(ordered_nodes):
            if i % 2 == 0:
                reordered_nodes.insert(0, node)
            else:
                reordered_nodes.append(node)
        return reordered_nodes
