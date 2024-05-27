from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set

import nest_asyncio

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import (
    CBEvent,
    CBEventType,
)


class UpTrainDataSchema:
    """UpTrain data schema."""

    def __init__(self, project_name: str) -> None:
        """Initialize the UpTrain data schema."""
        # For tracking project name and results
        self.project_name: str = project_name
        self.uptrain_results: DefaultDict[str, Any] = defaultdict(list)

        # For tracking event types - reranking, sub_question
        self.eval_types: Set[str] = set()

        ## SYNTHESIZE
        self.question: str = ""
        self.context: str = ""
        self.response: str = ""

        ## RERANKING
        self.old_context: List[str] = []
        self.new_context: List[str] = []
        self.reranking_type: Literal["resize", "rerank"] = "rerank"

        ## SUB_QUESTION
        # Map of sub question ID to question, context, and response
        self.sub_question_map: DefaultDict[str, dict] = defaultdict(dict)
        # Parent ID of sub questions
        self.sub_question_parent_id: str = ""
        # Parent question
        self.parent_question: str = ""


class UpTrainCallbackHandler(BaseCallbackHandler):
    """
    UpTrain callback handler.

    This class is responsible for handling the UpTrain API and logging events to UpTrain.

    """

    def __init__(
        self,
        api_key: str,
        key_type: Literal["uptrain", "openai"],
        project_name: str = "uptrain_llamaindex",
    ) -> None:
        """Initialize the UpTrain callback handler."""
        try:
            from uptrain import APIClient, EvalLLM, Settings
        except ImportError:
            raise ImportError(
                "UpTrainCallbackHandler requires the 'uptrain' package. "
                "Please install it using 'pip install uptrain'."
            )
        nest_asyncio.apply()
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self.schema = UpTrainDataSchema(project_name=project_name)
        self._event_pairs_by_id: Dict[str, List[CBEvent]] = defaultdict(list)
        self._trace_map: Dict[str, List[str]] = defaultdict(list)

        # Based on whether the user enters an UpTrain API key or an OpenAI API key, the client is initialized
        # If both are entered, the UpTrain API key is used
        if key_type == "uptrain":
            settings = Settings(uptrain_access_token=api_key)
            self.uptrain_client = APIClient(settings=settings)
        elif key_type == "openai":
            settings = Settings(openai_api_key=api_key)
            self.uptrain_client = EvalLLM(settings=settings)
        else:
            raise ValueError("Invalid key type: Must be 'uptrain' or 'openai'")

    def uptrain_evaluate(
        self,
        evaluation_name: str,
        data: List[Dict[str, str]],
        checks: List[str],
    ) -> None:
        """Run an evaluation on the UpTrain server using UpTrain client."""
        if self.uptrain_client.__class__.__name__ == "APIClient":
            uptrain_result = self.uptrain_client.log_and_evaluate(
                project_name=self.schema.project_name,
                evaluation_name=evaluation_name,
                data=data,
                checks=checks,
            )
        else:
            uptrain_result = self.uptrain_client.evaluate(
                project_name=self.schema.project_name,
                evaluation_name=evaluation_name,
                data=data,
                checks=checks,
            )
        self.schema.uptrain_results[self.schema.project_name].append(uptrain_result)

        score_name_map = {
            "score_context_relevance": "Context Relevance Score",
            "score_factual_accuracy": "Factual Accuracy Score",
            "score_response_completeness": "Response Completeness Score",
            "score_sub_query_completeness": "Sub Query Completeness Score",
            "score_context_reranking": "Context Reranking Score",
            "score_context_conciseness": "Context Conciseness Score",
        }

        # Print the results
        for row in uptrain_result:
            columns = list(row.keys())
            for column in columns:
                if column == "question":
                    print(f"\nQuestion: {row[column]}")
                elif column == "response":
                    print(f"Response: {row[column]}\n")
                elif column.startswith("score"):
                    if column in score_name_map:
                        print(f"{score_name_map[column]}: {row[column]}")
                    else:
                        print(f"{column}: {row[column]}")
            print()

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Any = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self._event_pairs_by_id[event.id_].append(event)

        if event_type is CBEventType.QUERY:
            self.schema.question = payload["query_str"]
        if event_type is CBEventType.TEMPLATING and "template_vars" in payload:
            template_vars = payload["template_vars"]
            self.schema.context = template_vars.get("context_str", "")
        elif event_type is CBEventType.RERANKING and "nodes" in payload:
            self.schema.eval_types.add("reranking")
            # Store old context data
            self.schema.old_context = [node.text for node in payload["nodes"]]
        elif event_type is CBEventType.SUB_QUESTION:
            # For the first sub question, store parent question and parent id
            if "sub_question" not in self.schema.eval_types:
                self.schema.parent_question = self.schema.question
                self.schema.eval_types.add("sub_question")
            # Store sub question data - question and parent id
            self.schema.sub_question_parent_id = parent_id
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Any = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        try:
            from uptrain import Evals
        except ImportError:
            raise ImportError(
                "UpTrainCallbackHandler requires the 'uptrain' package. "
                "Please install it using 'pip install uptrain'."
            )
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self._event_pairs_by_id[event.id_].append(event)
        self._trace_map = defaultdict(list)
        if event_id == self.schema.sub_question_parent_id:
            # Perform individual evaluations for sub questions (but send all sub questions at once)
            self.uptrain_evaluate(
                evaluation_name="sub_question_answering",
                data=list(self.schema.sub_question_map.values()),
                checks=[
                    Evals.CONTEXT_RELEVANCE,
                    Evals.FACTUAL_ACCURACY,
                    Evals.RESPONSE_COMPLETENESS,
                ],
            )
            # Perform evaluation for question and all sub questions (as a whole)
            sub_questions = [
                sub_question["question"]
                for sub_question in self.schema.sub_question_map.values()
            ]
            sub_questions_formatted = "\n".join(
                [
                    f"{index}. {string}"
                    for index, string in enumerate(sub_questions, start=1)
                ]
            )
            self.uptrain_evaluate(
                evaluation_name="sub_query_completeness",
                data=[
                    {
                        "question": self.schema.parent_question,
                        "sub_questions": sub_questions_formatted,
                    }
                ],
                checks=[Evals.SUB_QUERY_COMPLETENESS],
            )
            self.schema.eval_types.remove("sub_question")
        # Should not be called for sub questions
        if (
            event_type is CBEventType.SYNTHESIZE
            and "sub_question" not in self.schema.eval_types
        ):
            self.schema.response = payload["response"].response
            # Perform evaluation for synthesization
            if "reranking" in self.schema.eval_types:
                if self.schema.reranking_type == "rerank":
                    evaluation_name = "question_answering_rerank"
                else:
                    evaluation_name = "question_answering_resize"
                self.schema.eval_types.remove("reranking")
            else:
                evaluation_name = "question_answering"
            self.uptrain_evaluate(
                evaluation_name=evaluation_name,
                data=[
                    {
                        "question": self.schema.question,
                        "context": self.schema.context,
                        "response": self.schema.response,
                    }
                ],
                checks=[
                    Evals.CONTEXT_RELEVANCE,
                    Evals.FACTUAL_ACCURACY,
                    Evals.RESPONSE_COMPLETENESS,
                ],
            )

        elif event_type is CBEventType.RERANKING:
            # Store new context data
            self.schema.new_context = [node.text for node in payload["nodes"]]
            if len(self.schema.old_context) == len(self.schema.new_context):
                self.schema.reranking_type = "rerank"
                context = "\n".join(
                    [
                        f"{index}. {string}"
                        for index, string in enumerate(self.schema.old_context, start=1)
                    ]
                )
                reranked_context = "\n".join(
                    [
                        f"{index}. {string}"
                        for index, string in enumerate(self.schema.new_context, start=1)
                    ]
                )
                # Perform evaluation for reranking
                self.uptrain_evaluate(
                    evaluation_name="context_reranking",
                    data=[
                        {
                            "question": self.schema.question,
                            "context": context,
                            "reranked_context": reranked_context,
                        }
                    ],
                    checks=[
                        Evals.CONTEXT_RERANKING,
                    ],
                )
            else:
                self.schema.reranking_type = "resize"
                context = "\n".join(self.schema.old_context)
                concise_context = "\n".join(self.schema.new_context)
                # Perform evaluation for resizing
                self.uptrain_evaluate(
                    evaluation_name="context_conciseness",
                    data=[
                        {
                            "question": self.schema.question,
                            "context": context,
                            "concise_context": concise_context,
                        }
                    ],
                    checks=[
                        Evals.CONTEXT_CONCISENESS,
                    ],
                )
        elif event_type is CBEventType.SUB_QUESTION:
            # Store sub question data
            self.schema.sub_question_map[event_id]["question"] = payload[
                "sub_question"
            ].sub_q.sub_question
            self.schema.sub_question_map[event_id]["context"] = (
                payload["sub_question"].sources[0].node.text
            )
            self.schema.sub_question_map[event_id]["response"] = payload[
                "sub_question"
            ].answer

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self._trace_map = defaultdict(list)
        return super().start_trace(trace_id)

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._trace_map = trace_map or defaultdict(list)
        return super().end_trace(trace_id, trace_map)

    def build_trace_map(
        self,
        cur_event_id: str,
        trace_map: Any,
    ) -> Dict[str, Any]:
        event_pair = self._event_pairs_by_id[cur_event_id]
        if event_pair:
            event_data = {
                "event_type": event_pair[0].event_type,
                "event_id": event_pair[0].id_,
                "children": {},
            }
            trace_map[cur_event_id] = event_data

        child_event_ids = self._trace_map[cur_event_id]
        for child_event_id in child_event_ids:
            self.build_trace_map(child_event_id, event_data["children"])
        return trace_map
