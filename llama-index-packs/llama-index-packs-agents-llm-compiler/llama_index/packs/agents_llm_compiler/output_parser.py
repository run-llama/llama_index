"""LLM Compiler Output Parser."""

import re
from typing import Any, Dict, List, Sequence

from llama_index.core.tools import BaseTool
from llama_index.core.types import BaseOutputParser

from .schema import JoinerOutput, LLMCompilerParseResult
from .utils import get_graph_dict

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
# $1 or ${1} -> 1
ID_PATTERN = r"\$\{?(\d+)\}?"

END_OF_PLAN = "<END_OF_PLAN>"
JOINER_REPLAN = "Replan"


def default_dependency_rule(idx: int, args: str) -> bool:
    """Default dependency rule."""
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


class LLMCompilerPlanParser(BaseOutputParser):
    """LLM Compiler plan output parser.

    Directly adapted from source code: https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/output_parser.py.

    """

    def __init__(self, tools: Sequence[BaseTool]):
        """Init params."""
        self.tools = tools

    def parse(self, text: str) -> Dict[int, Any]:
        # 1. search("Ronaldo number of kids") -> 1, "search", '"Ronaldo number of kids"'
        # pattern = r"(\d+)\. (\w+)\(([^)]+)\)"
        pattern = rf"(?:{THOUGHT_PATTERN}\n)?{ACTION_PATTERN}"
        matches = re.findall(pattern, text)

        # convert matches to a list of LLMCompilerParseResult
        results: List[LLMCompilerParseResult] = []
        for match in matches:
            thought, idx, tool_name, args, _ = match
            idx = int(idx)
            results.append(
                LLMCompilerParseResult(
                    thought=thought, idx=idx, tool_name=tool_name, args=args
                )
            )

        # get graph dict
        return get_graph_dict(results, self.tools)


### Helper functions


class LLMCompilerJoinerParser(BaseOutputParser):
    """LLM Compiler output parser for the join step.

    Adapted from _parse_joiner_output in
    https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/llm_compiler.py

    """

    def parse(self, text: str) -> JoinerOutput:
        """Parse."""
        thought, answer, is_replan = "", "", False  # default values
        raw_answers = text.split("\n")
        for answer in raw_answers:
            if answer.startswith("Action:"):
                answer = answer[answer.find("(") + 1 : answer.find(")")]
                is_replan = JOINER_REPLAN in answer
            elif answer.startswith("Thought:"):
                thought = answer.split("Thought:")[1].strip()
        return JoinerOutput(thought=thought, answer=answer, is_replan=is_replan)
