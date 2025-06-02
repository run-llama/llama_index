import math
from typing import List, Optional

from llama_index.core.agent.react.types import (
    BaseReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.core.prompts import PromptTemplate

# taken from the paper
DEFAULT_REFLECTION_PROMPT_STR = """\
Given a query and a conversation trajectory, evaluate two things regarding whether the conversation answers the question:
- **correctness**: Whether the thoughts and actions so far are correctly answering the query, even if the answer is not found yet. Rate from 1-10, where 1 is incorrect and 10 is correct.
- **completeness**: Whether the answer is found yet.
Provide your reasoning and analysis in detail.
Focus on the latest thought, action, and observation.
Incomplete trajectories can be correct if the thoughts and actions so far are correct, \
even if the answer is not found yet.
Do not generate additional thoughts or actions.


Query: {query}
Conversation History:
{conversation_history}
"""

DEFAULT_REFLECTION_PROMPT = PromptTemplate(DEFAULT_REFLECTION_PROMPT_STR)

DEFAULT_CANDIDATES_PROMPT_STR = """\
Given a query and a conversation trajectory, provide a list of candidates {num_candidates} for the next reasoning step.
Focus on the latest thought, action, and observation.
Do not generate additional thoughts or actions.

Query: {query}
Conversation History:
{conversation_history}
"""

DEFAULT_CANDIDATES_PROMPT = PromptTemplate(DEFAULT_CANDIDATES_PROMPT_STR)


class Candidates(BaseModel):
    """Candidates for the next reasoning step."""

    candidates: List[str]


class Evaluation(BaseModel):
    """Evaluation of a given node."""

    score: int = Field(
        description="Score of the reflection indicating **correctness**. Integer from 1-10",
        le=10,
        ge=0,
    )
    is_done: bool = Field(
        False, description="Whether the answer is found yet (**completeness**)."
    )
    reasoning: str = Field(
        default="", description="Reasoning and justification for the evaluation."
    )


class SearchNode(BaseModel):
    """
    Search node.

    Named differently from `Node` which is a core module in LlamaIndex.

    """

    current_reasoning: List[BaseReasoningStep] = Field(
        ..., description="Current reasoning."
    )
    parent: Optional["SearchNode"] = Field(default=None, description="Parent node.")
    children: List["SearchNode"] = Field(
        default_factory=list, description="Children nodes."
    )
    evaluation: Evaluation = Field(..., description="Evaluation of the node.")
    visits: int = Field(default=0, description="Number of visits to the node.")

    @property
    def answer(self) -> Optional[str]:
        """Answer."""
        if not self.current_reasoning:
            return None

        if isinstance(self.current_reasoning[-1], ResponseReasoningStep):
            return self.current_reasoning[-1].response
        else:
            return self.current_reasoning[-1].get_content()

    @property
    def is_done(self) -> bool:
        """Is the node done."""
        return self.evaluation.is_done

    @property
    def score(self) -> float:
        """Score of the node."""
        return self.evaluation.score

    @property
    def upper_confidence_bound(self) -> float:
        """Upper confidence bound."""
        return self.score + 1.0 * math.sqrt(math.log(self.parent.visits) / self.visits)

    def backpropagate(self, reward: float) -> None:
        """Backpropagate the reward."""
        cur_node = self
        while cur_node is not None:
            cur_node.visits += 1
            cur_node.evaluation.score = (
                reward + (cur_node.visits - 1) * cur_node.score
            ) / cur_node.visits
            cur_node = cur_node.parent

    def get_best_leaf(self) -> "SearchNode":
        """
        Get best leaf node.

        Get best leaf node across any children nodes.

        """
        # only get children that aren't done yet
        free_children = [c for c in self.children if not c.is_done]
        if not free_children:
            return self

        best_child = max(free_children, key=lambda x: x.upper_confidence_bound)
        return best_child.get_best_leaf()
