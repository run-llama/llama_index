import ast
import logging
import random
import re
import signal
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple

from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.program.evaporate.prompts import (
    DEFAULT_EXPECTED_OUTPUT_PREFIX_TMPL,
    DEFAULT_FIELD_EXTRACT_QUERY_TMPL,
    FN_GENERATION_PROMPT,
    SCHEMA_ID_PROMPT,
    FnGeneratePrompt,
    SchemaIDPrompt,
)


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: int) -> Any:
    """
    Time limit context manager.

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
    """
    Get function field from attribute.

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
        if not any(f.lower() in text_chunk.lower() for f in field_versions):
            continue
        if not value:
            continue
        field = field.lower().strip("-").strip("_").strip(" ").strip(":")
        if field in existing_fields:
            continue
        existing_fields.add(field)

    return existing_fields


logger = logging.getLogger(__name__)

# Builtins allowed in the sandboxed execution environment for LLM-generated
# extraction functions.  Dangerous builtins (eval, exec, compile, open,
# __import__, getattr, setattr, delattr, globals, locals, vars, breakpoint)
# are intentionally excluded.
_SANDBOX_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hasattr": hasattr,
    "hash": hash,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


_SANDBOX_ALLOWED_IMPORTS = {
    "re",
    "math",
    "datetime",
    "json",
    "string",
    "typing",
    "time",
    "collections",
    "itertools",
    "functools",
    "decimal",
    "fractions",
    "statistics",
    "textwrap",
    "unicodedata",
    "operator",
}


def _validate_generated_code(code: str) -> None:
    """Reject generated code that accesses dunder/private attrs or unsafe imports."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _SANDBOX_ALLOWED_IMPORTS:
                    raise RuntimeError(
                        f"Generated code imports '{alias.name}' which is not "
                        f"in the allowed set: {_SANDBOX_ALLOWED_IMPORTS}"
                    )
        if isinstance(node, ast.ImportFrom):
            if (
                node.module
                and node.module.split(".")[0] not in _SANDBOX_ALLOWED_IMPORTS
            ):
                raise RuntimeError(
                    f"Generated code imports from '{node.module}' which is not "
                    f"in the allowed set: {_SANDBOX_ALLOWED_IMPORTS}"
                )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RuntimeError(
                f"Generated code accesses a dunder name '{node.id}' which is "
                f"not allowed in the sandbox"
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise RuntimeError(
                f"Generated code accesses a dunder attribute '{node.attr}' "
                f"which is not allowed in the sandbox"
            )


def _restricted_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    if name in _SANDBOX_ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is not allowed in the sandbox")


def _build_sandbox(node_text: str) -> Dict[str, Any]:
    """Build a restricted globals dict for executing generated functions."""
    builtins = {**_SANDBOX_BUILTINS, "__import__": _restricted_import}
    return {
        "__builtins__": builtins,
        "re": re,
        "node_text": node_text,
    }


class EvaporateExtractor:
    """
    Wrapper around Evaporate.

    Evaporate is an open-source project from Stanford's AI Lab:
    https://github.com/HazyResearch/evaporate.
    Offering techniques for structured datapoint extraction.

    In the current version, we use the function generator
    from a set of documents.
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        schema_id_prompt: Optional[SchemaIDPrompt] = None,
        fn_generate_prompt: Optional[FnGeneratePrompt] = None,
        field_extract_query_tmpl: str = DEFAULT_FIELD_EXTRACT_QUERY_TMPL,
        expected_output_prefix_tmpl: str = DEFAULT_EXPECTED_OUTPUT_PREFIX_TMPL,
        verbose: bool = False,
    ) -> None:
        """Initialize params."""
        # TODO: take in an entire index instead of forming a response builder
        self._llm = llm or Settings.llm
        self._schema_id_prompt = schema_id_prompt or SCHEMA_ID_PROMPT
        self._fn_generate_prompt = fn_generate_prompt or FN_GENERATION_PROMPT
        self._field_extract_query_tmpl = field_extract_query_tmpl
        self._expected_output_prefix_tmpl = expected_output_prefix_tmpl
        self._verbose = verbose

    def identify_fields(
        self, nodes: List[BaseNode], topic: str, fields_top_k: int = 5
    ) -> List:
        """
        Identify fields from nodes.

        Will extract fields independently per node, and then
        return the top k fields.

        Args:
            nodes (List[BaseNode]): List of nodes to extract fields from.
            topic (str): Topic to use for extraction.
            fields_top_k (int): Number of fields to return.

        """
        field2count: dict = defaultdict(int)
        for node in nodes:
            result = self._llm.predict(
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
            field2count.items(), key=lambda x: x[1], reverse=True
        )
        sorted_fields = [f[0] for f in sorted_tups]
        return sorted_fields[:fields_top_k]

    def extract_fn_from_nodes(
        self, nodes: List[BaseNode], field: str, expected_output: Optional[Any] = None
    ) -> str:
        """Extract function from nodes."""
        # avoid circular import
        from llama_index.core.response_synthesizers import (
            ResponseMode,
            get_response_synthesizer,
        )

        function_field = get_function_field_from_attribute(field)
        # TODO: replace with new response synthesis module

        if expected_output is not None:
            expected_output_str = (
                f"{self._expected_output_prefix_tmpl}{expected_output!s}\n"
            )
        else:
            expected_output_str = ""

        qa_prompt = self._fn_generate_prompt.partial_format(
            attribute=field,
            function_field=function_field,
            expected_output_str=expected_output_str,
        )

        response_synthesizer = get_response_synthesizer(
            llm=self._llm,
            text_qa_template=qa_prompt,
            response_mode=ResponseMode.TREE_SUMMARIZE,
        )

        # ignore refine prompt for now
        query_str = self._field_extract_query_tmpl.format(field=function_field)
        query_bundle = QueryBundle(query_str=query_str)
        response = response_synthesizer.synthesize(
            query_bundle,
            [NodeWithScore(node=n, score=1.0) for n in nodes],
        )
        fn_str = f"""def get_{function_field}_field(text: str):
    \"""
    Function to extract {field}.
    \"""
    {response!s}
"""

        # format fn_str
        return_idx_list = [i for i, s in enumerate(fn_str.split("\n")) if "return" in s]
        if not return_idx_list:
            return ""

        return_idx = return_idx_list[0]
        fn_str = "\n".join(fn_str.split("\n")[: return_idx + 1])
        fn_str = "\n".join([s for s in fn_str.split("\n") if "print(" not in s])
        return "\n".join(
            [s for s in fn_str.split("\n") if s.startswith((" ", "\t", "def"))]
        )

    def run_fn_on_nodes(
        self, nodes: List[BaseNode], fn_str: str, field_name: str, num_timeouts: int = 1
    ) -> List:
        """
        Run LLM-generated extraction function on nodes.

        Executes the generated code in a sandboxed environment with restricted
        builtins and no access to dangerous functions like __import__, eval,
        exec, open, etc.  The ``re`` module is pre-loaded since the prompt
        templates produce code that uses it.

        """
        function_field = get_function_field_from_attribute(field_name)

        # Validate the generated code before execution
        _validate_generated_code(fn_str)

        results = []
        for node in nodes:
            sandbox = _build_sandbox(node.get_content())
            try:
                with time_limit(1):
                    exec(fn_str, sandbox)
                    exec(
                        f"__result__ = get_{function_field}_field(node_text)",
                        sandbox,
                    )
            except TimeoutException:
                raise
            results.append(sandbox.get("__result__", []))
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
