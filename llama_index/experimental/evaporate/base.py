"""Evaporate wrapper."""

import random
import re
import signal
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple

from llama_index.schema import BaseNode, MetadataMode
from llama_index.experimental.evaporate.prompts import (
    FN_GENERATION_PROMPT,
    SCHEMA_ID_PROMPT,
    FnGeneratePrompt,
    SchemaIDPrompt,
)
from llama_index.indices.list.base import ListIndex
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.prompts import QuestionAnswerPrompt


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: int) -> Any:
    """Time limit context manager.

    NOTE: copied from https://github.com/HazyResearch/evaporate.

    """

    def signal_handler(signum: Any, frame: Any) -> Any:
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_function_field_from_attribute(attribute: str) -> str:
    """Get function field from attribute.

    NOTE: copied from https://github.com/HazyResearch/evaporate.

    """
    return re.sub(r"[^A-Za-z0-9]", "_", attribute)


def extract_field_dicts(result: str, text_chunk: str) -> Set:
    """Extract field dictionaries."""
    existing_fields = set()
    result = result.split("---")[0].strip("\n")
    results = result.split("\n")
    results = [r.strip("-").strip() for r in results]
    results = [r[2:].strip() if len(r) > 2 and r[1] == "." else r for r in results]
    for result in results:
        try:
            field = result.split(": ")[0].strip(":")
            value = ": ".join(result.split(": ")[1:])
        except Exception:
            print(f"Skipped: {result}")
            continue
        field_versions = [
            field,
            field.replace(" ", ""),
            field.replace("-", ""),
            field.replace("_", ""),
        ]
        if not any([f.lower() in text_chunk.lower() for f in field_versions]):
            continue
        if not value:
            continue
        field = field.lower().strip("-").strip("_").strip(" ").strip(":")
        if field in existing_fields:
            continue
        existing_fields.add(field)

    return existing_fields


node_text: str
result: List


# since we define globals below
class EvaporateExtractor:
    """Wrapper around Evaporate.

    Evaporate is an open-source project from Stanford's AI Lab:
    https://github.com/HazyResearch/evaporate.
    Offering techniques for structured datapoint extraction.

    In the current version, we use the function generator
    from a set of documents.

    Args:
        service_context (Optional[ServiceContext]): Service Context to use.
    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        schema_id_prompt: Optional[SchemaIDPrompt] = None,
        fn_generate_prompt: Optional[FnGeneratePrompt] = None,
    ) -> None:
        """Initialize params."""
        # TODO: take in an entire index instead of forming a response builder
        self._service_context = service_context or ServiceContext.from_defaults()
        self._schema_id_prompt = schema_id_prompt or SCHEMA_ID_PROMPT
        self._fn_generate_prompt = fn_generate_prompt or FN_GENERATION_PROMPT

    def identify_fields(
        self, nodes: List[BaseNode], topic: str, fields_top_k: int = 5
    ) -> List:
        """Identify fields from nodes."""
        field2count: dict = defaultdict(int)
        for node in nodes:
            llm_predictor = self._service_context.llm_predictor
            result, _ = llm_predictor.predict(
                self._schema_id_prompt,
                topic=topic,
                chunk=node.get_content(metadata_mode=MetadataMode.LLM),
            )

            existing_fields = extract_field_dicts(
                result, node.get_content(metadata_mode=MetadataMode.LLM)
            )

            for field in existing_fields:
                field2count[field] += 1

        sorted_tups: List[Tuple[str, int]] = sorted(
            list(field2count.items()), key=lambda x: x[1], reverse=True
        )
        sorted_fields = [f[0] for f in sorted_tups]
        sorted_fields = sorted_fields[:fields_top_k]

        return sorted_fields

    def extract_fn_from_nodes(self, nodes: List[BaseNode], field: str) -> str:
        """Extract function from nodes."""
        function_field = get_function_field_from_attribute(field)
        index = ListIndex(nodes)
        new_prompt = self._fn_generate_prompt.partial_format(
            attribute=field, function_field=function_field
        )
        qa_prompt = QuestionAnswerPrompt.from_prompt(new_prompt)
        # ignore refine prompt for now
        query_str = (
            f'Write a python function to extract the entire "{field}" field from text, '
            "but not any other metadata. Return the result as a list."
        )
        query_engine = index.as_query_engine(
            response_mode="compact", text_qa_template=qa_prompt
        )
        response = query_engine.query(query_str)
        fn_str = f"""def get_{function_field}_field(text: str):
    \"""
    Function to extract {field}. 
    \"""
    {str(response)}
"""

        # format fn_str
        return_idx_list = [i for i, s in enumerate(fn_str.split("\n")) if "return" in s]
        if not return_idx_list:
            return ""

        return_idx = return_idx_list[0]
        fn_str = "\n".join(fn_str.split("\n")[: return_idx + 1])
        fn_str = "\n".join([s for s in fn_str.split("\n") if "print(" not in s])
        fn_str = "\n".join(
            [
                s
                for s in fn_str.split("\n")
                if s.startswith(" ") or s.startswith("\t") or s.startswith("def")
            ]
        )
        return fn_str

    def run_fn_on_nodes(
        self, nodes: List[BaseNode], fn_str: str, field_name: str, num_timeouts: int = 1
    ) -> List:
        """Run function on nodes.

        Calls python exec().

        There are definitely security holes with this approach, use with caution.

        """
        function_field = get_function_field_from_attribute(field_name)
        results = []
        for node in nodes:
            global result
            global node_text
            node_text = node.get_content()
            # this is temporary
            result = []
            try:
                with time_limit(1):
                    exec(fn_str, globals())
                    exec(f"result = get_{function_field}_field(node_text)", globals())
            except TimeoutException as e:
                raise e
            results.append(result)
        return results

    def extract_datapoints_with_fn(
        self,
        nodes: List[BaseNode],
        topic: str,
        sample_k: int = 5,
        fields_top_k: int = 5,
    ) -> List[Dict]:
        """Extract datapoints from a list of nodes, given a topic."""
        idxs = list(range(len(nodes)))
        sample_k = min(sample_k, len(nodes))
        subset_idxs = random.sample(idxs, sample_k)
        subset_nodes = [nodes[si] for si in subset_idxs]

        # get existing fields
        existing_fields = self.identify_fields(
            subset_nodes, topic, fields_top_k=fields_top_k
        )

        # then, for each existing field, generate function
        function_dict = {}
        for field in existing_fields:
            fn = self.extract_fn_from_nodes(subset_nodes, field)
            function_dict[field] = fn

        # then, run function for all nodes
        result_dict = {}
        for field in existing_fields:
            result_list = self.run_fn_on_nodes(nodes, function_dict[field], field)
            result_dict[field] = result_list

        # convert into list of dictionaries
        result_list = []
        for i in range(len(nodes)):
            result_dict_i = {}
            for field in existing_fields:
                result_dict_i[field] = result_dict[field][i]
            result_list.append(result_dict_i)
        return result_list
