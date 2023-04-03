"""Node postprocessor."""

import re
from abc import abstractmethod
from typing import Dict, List, Optional, cast

from pydantic import BaseModel, Field, validator

import logging
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.service_context import ServiceContext
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from gpt_index.docstore import DocumentStore
from gpt_index.data_structs.node_v2 import Node, DocumentRelationship
from gpt_index.indices.postprocessor.base import BasePostprocessor
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.response.builder import ResponseBuilder, TextChunk

logger = logging.getLogger(__name__)


class BaseNodePostprocessor(BasePostprocessor, BaseModel):
    """Node postprocessor."""

    @abstractmethod
    def postprocess_nodes(
        self, nodes: List[Node], extra_info: Optional[Dict] = None
    ) -> List[Node]:
        """Postprocess nodes."""


class KeywordNodePostprocessor(BaseNodePostprocessor):
    """Keyword-based Node processor."""

    required_keywords: List[str] = Field(default_factory=list)
    exclude_keywords: List[str] = Field(default_factory=list)

    def postprocess_nodes(
        self, nodes: List[Node], extra_info: Optional[Dict] = None
    ) -> List[Node]:
        """Postprocess nodes."""
        new_nodes = []
        for node in nodes:
            words = re.findall(r"\w+", node.get_text())
            should_use_node = True
            if self.required_keywords is not None:
                for w in self.required_keywords:
                    if w not in words:
                        should_use_node = False

            if self.exclude_keywords is not None:
                for w in self.exclude_keywords:
                    if w in words:
                        should_use_node = False

            if should_use_node:
                new_nodes.append(node)

        return new_nodes


class SimilarityPostprocessor(BaseNodePostprocessor):
    """Similarity-based Node processor."""

    similarity_cutoff: float = Field(default=None)

    def postprocess_nodes(
        self, nodes: List[Node], extra_info: Optional[Dict] = None
    ) -> List[Node]:
        """Postprocess nodes."""
        extra_info = extra_info or {}
        similarity_tracker = extra_info.get("similarity_tracker", None)
        if similarity_tracker is None:
            return nodes
        sim_cutoff_exists = (
            similarity_tracker is not None and self.similarity_cutoff is not None
        )

        new_nodes = []
        for node in nodes:
            should_use_node = True
            if sim_cutoff_exists:
                similarity = cast(SimilarityTracker, similarity_tracker).find(node)
                if similarity is None:
                    should_use_node = False
                if cast(float, similarity) < cast(float, self.similarity_cutoff):
                    should_use_node = False

            if should_use_node:
                new_nodes.append(node)

        return new_nodes


def get_forward_nodes(
    node: Node, num_nodes: int, docstore: DocumentStore
) -> Dict[str, Node]:
    """Get forward nodes."""
    nodes: Dict[str, Node] = {node.get_doc_id(): node}
    cur_count = 0
    # get forward nodes in an iterative manner
    while cur_count < num_nodes:
        if DocumentRelationship.NEXT not in node.relationships:
            break
        next_node_id = node.relationships[DocumentRelationship.NEXT]
        next_node = docstore.get_node(next_node_id)
        if next_node is None:
            break
        nodes[next_node.get_doc_id()] = next_node
        node = next_node
        cur_count += 1
    return nodes


def get_backward_nodes(
    node: Node, num_nodes: int, docstore: DocumentStore
) -> Dict[str, Node]:
    """Get backward nodes."""
    # get backward nodes in an iterative manner
    nodes: Dict[str, Node] = {node.get_doc_id(): node}
    cur_count = 0
    while cur_count < num_nodes:
        if DocumentRelationship.PREVIOUS not in node.relationships:
            break
        prev_node_id = node.relationships[DocumentRelationship.PREVIOUS]
        prev_node = docstore.get_node(prev_node_id)
        if prev_node is None:
            break
        nodes[prev_node.get_doc_id()] = prev_node
        node = prev_node
        cur_count += 1
    return nodes


class PrevNextNodePostprocessor(BaseNodePostprocessor):
    """Previous/Next Node post-processor.

    Allows users to fetch additional nodes from the document store,
    based on the relationships of the nodes.

    NOTE: this is a beta feature.

    Args:
        docstore (DocumentStore): The document store.
        num_nodes (int): The number of nodes to return (default: 1)
        mode (str): The mode of the post-processor.
            Can be "previous", "next", or "both.

    """

    docstore: DocumentStore
    num_nodes: int = Field(default=1)
    mode: str = Field(default="next")

    def _get_backward_nodes(self, node: Node) -> Dict[str, Node]:
        """Get backward nodes."""
        # get backward nodes in an iterative manner
        nodes: Dict[str, Node] = {node.get_doc_id(): node}
        cur_count = 0
        while cur_count < self.num_nodes:
            if DocumentRelationship.PREVIOUS not in node.relationships:
                break
            prev_node_id = node.relationships[DocumentRelationship.PREVIOUS]
            prev_node = self.docstore.get_node(prev_node_id)
            if prev_node is None:
                break
            nodes[prev_node.get_doc_id()] = prev_node
            node = prev_node
            cur_count += 1
        return nodes

    @validator("mode")
    def _validate_mode(cls, v: str) -> str:
        """Validate mode."""
        if v not in ["next", "previous", "both"]:
            raise ValueError(f"Invalid mode: {v}")
        return v

    def postprocess_nodes(
        self, nodes: List[Node], extra_info: Optional[Dict] = None
    ) -> List[Node]:
        """Postprocess nodes."""
        all_nodes: Dict[str, Node] = {}
        for node in nodes:
            all_nodes[node.get_doc_id()] = node
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

        sorted_nodes = sorted(all_nodes.values(), key=lambda x: x.get_doc_id())
        return list(sorted_nodes)


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
        docstore (DocumentStore): The document store.
        llm_predictor (LLMPredictor): The LLM predictor.
        num_nodes (int): The number of nodes to return (default: 1)
        infer_prev_next_tmpl (str): The template to use for inference.
            Required fields are {context_str} and {query_str}.

    """

    docstore: DocumentStore
    service_context: ServiceContext
    num_nodes: int = Field(default=1)
    infer_prev_next_tmpl: str = Field(default=DEFAULT_INFER_PREV_NEXT_TMPL)
    refine_prev_next_tmpl: str = Field(default=DEFAULT_REFINE_INFER_PREV_NEXT_TMPL)
    verbose: bool = Field(default=False)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

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

    def postprocess_nodes(
        self, nodes: List[Node], extra_info: Optional[Dict] = None
    ) -> List[Node]:
        """Postprocess nodes."""
        if extra_info is None or "query_bundle" not in extra_info:
            raise ValueError("Missing query bundle in extra info.")

        query_bundle = cast(QueryBundle, extra_info["query_bundle"])

        infer_prev_next_prompt = QuestionAnswerPrompt(
            self.infer_prev_next_tmpl,
        )
        refine_infer_prev_next_prompt = RefinePrompt(self.refine_prev_next_tmpl)

        all_nodes: Dict[str, Node] = {}
        for node in nodes:
            all_nodes[node.get_doc_id()] = node
            # use response builder instead of llm_predictor directly
            # to be more robust to handling long context
            response_builder = ResponseBuilder(
                self.service_context,
                infer_prev_next_prompt,
                refine_infer_prev_next_prompt,
            )
            response_builder.add_text_chunks([TextChunk(node.get_text())])
            raw_pred = response_builder.get_response(
                query_str=query_bundle.query_str,
                response_mode="tree_summarize",
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

        sorted_nodes = sorted(all_nodes.values(), key=lambda x: x.get_doc_id())
        return list(sorted_nodes)
