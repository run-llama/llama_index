from typing import Dict, Optional

from llama_index.evaluation.base import Evaluation
from llama_index.indices.query.query_transform.base import BaseQueryTransform
from llama_index.indices.query.schema import QueryBundle


class FeedbackQueryTransformation(BaseQueryTransform):
    """Transform the query given the evaluation feedback.

    Args:
        eval(Evaluation): An evaluation object.

    """

    def __init__(
        self,
        evaluation: Evaluation,
    ) -> None:
        super().__init__()
        self.evaluation = evaluation

    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> QueryBundle:
        query_str = query_bundle.query_str
        new_query = (
            query_str
            + "\n----------------\n"
            + self.construct_feedback(self.evaluation.response.response)
        )
        return QueryBundle(new_query)

    def construct_feedback(self, response: Optional[str]) -> str:
        """Construct feedback from response."""
        if response is None:
            return ""
        else:
            return "Here is a previous bad answer.\n" + response
