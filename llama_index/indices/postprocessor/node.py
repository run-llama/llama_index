"""Node postprocessor."""

import logging
import re
from abc import abstractmethod
from typing import Dict, List, Optional, cast

from pydantic import BaseModel, Field, validator

from llama_index.data_structs.node import DocumentRelationship, NodeWithScore
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.response import get_response_builder
from llama_index.indices.response.type import ResponseMode
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.storage.docstore import BaseDocumentStore

logger = logging.getLogger(__name__)


class BasePydanticNodePostprocessor(BaseModel, BaseNodePostprocessor):
    """Node postprocessor."""

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""


class KeywordNodePostprocessor(BasePydanticNodePostprocessor):
    """Keyword-based Node processor."""

    required_keywords: List[str] = Field(default_factory=list)
    exclude_keywords: List[str] = Field(default_factory=list)

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        new_nodes = []
        for node_with_score in nodes:
            node = node_with_score.node
            should_use_node = True
            if self.required_keywords is not None:
                for keyword in self.required_keywords:
                    pattern = r"\b" + re.escape(keyword) + r"\b"
                    keyword_presence = re.search(pattern, node.get_text())
                    if not keyword_presence:
                        should_use_node = False

            if self.exclude_keywords is not None:
                for keyword in self.exclude_keywords:
                    pattern = r"\b" + re.escape(keyword) + r"\b"
                    keyword_presence = re.search(keyword, node.get_text())
                    if keyword_presence:
                        should_use_node = False

            if should_use_node:
                new_nodes.append(node_with_score)

        return new_nodes


class SimilarityPostprocessor(BasePydanticNodePostprocessor):
    """Similarity-based Node processor."""

    similarity_cutoff: float = Field(default=None)

    def postprocess_nodes(
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
    nodes: Dict[str, NodeWithScore] = {node.get_doc_id(): node_with_score}
    cur_count = 0
    # get forward nodes in an iterative manner
    while cur_count < num_nodes:
        if DocumentRelationship.NEXT not in node.relationships:
            break
        next_node_id = node.relationships[DocumentRelationship.NEXT]
        next_node = docstore.get_node(next_node_id)
        if next_node is None:
            break
        nodes[next_node.get_doc_id()] = NodeWithScore(next_node)
        node = next_node
        cur_count += 1
    return nodes


def get_backward_nodes(
    node_with_score: NodeWithScore, num_nodes: int, docstore: BaseDocumentStore
) -> Dict[str, NodeWithScore]:
    """Get backward nodes."""
    node = node_with_score.node
    # get backward nodes in an iterative manner
    nodes: Dict[str, NodeWithScore] = {node.get_doc_id(): node_with_score}
    cur_count = 0
    while cur_count < num_nodes:
        if DocumentRelationship.PREVIOUS not in node.relationships:
            break
        prev_node_id = node.relationships[DocumentRelationship.PREVIOUS]
        prev_node = docstore.get_node(prev_node_id)
        if prev_node is None:
            break
        nodes[prev_node.get_doc_id()] = NodeWithScore(prev_node)
        node = prev_node
        cur_count += 1
    return nodes


class PrevNextNodePostprocessor(BasePydanticNodePostprocessor):
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

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        all_nodes: Dict[str, NodeWithScore] = {}
        for node in nodes:
            all_nodes[node.node.get_doc_id()] = node
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

        sorted_nodes = sorted(all_nodes.values(), key=lambda x: x.node.get_doc_id())
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


class AutoPrevNextNodePostprocessor(BasePydanticNodePostprocessor):
    """Previous/Next Node post-processor.

    Allows users to fetch additional nodes from the document store,
    based on the prev/next relationships of the nodes.

    NOTE: difference with PrevNextPostprocessor is that
    this infers forward/backwards direction.

    NOTE: this is a beta feature.

    Args:
        docstore (BaseDocumentStore): The document store.
        llm_predictor (LLMPredictor): The LLM predictor.
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
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        if query_bundle is None:
            raise ValueError("Missing query bundle.")

        infer_prev_next_prompt = QuestionAnswerPrompt(
            self.infer_prev_next_tmpl,
        )
        refine_infer_prev_next_prompt = RefinePrompt(self.refine_prev_next_tmpl)

        all_nodes: Dict[str, NodeWithScore] = {}
        for node in nodes:
            all_nodes[node.node.get_doc_id()] = node
            # use response builder instead of llm_predictor directly
            # to be more robust to handling long context
            response_builder = get_response_builder(
                service_context=self.service_context,
                text_qa_template=infer_prev_next_prompt,
                refine_template=refine_infer_prev_next_prompt,
                mode=ResponseMode.TREE_SUMMARIZE,
            )
            raw_pred = response_builder.get_response(
                text_chunks=[node.node.get_text()],
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

        sorted_nodes = sorted(all_nodes.values(), key=lambda x: x.node.get_doc_id())
        return list(sorted_nodes)
