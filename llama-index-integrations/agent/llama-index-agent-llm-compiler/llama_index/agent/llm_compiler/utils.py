"""Utils for LLM Compiler."""
import ast
import re
from typing import Any, Dict, List, Sequence, Tuple, Union

from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import BaseTool, adapt_to_async_tool

from .schema import (
    LLMCompilerParseResult,
    LLMCompilerTask,
)

# $1 or ${1} -> 1
ID_PATTERN = r"\$\{?(\d+)\}?"


def default_dependency_rule(idx: int, args: str) -> bool:
    """Default dependency rule."""
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


def parse_llm_compiler_action_args(args: str) -> Union[List, Tuple]:
    """Parse arguments from a string."""
    # This will convert the string into a python object
    # e.g. '"Ronaldo number of kids"' -> ("Ronaldo number of kids", )
    # '"I can answer the question now.", [3]' -> ("I can answer the question now.", [3])
    if args == "":
        return ()
    try:
        eval_args: Union[List, Tuple, str] = ast.literal_eval(args)
    except Exception:
        eval_args = args
    if not isinstance(eval_args, list) and not isinstance(eval_args, tuple):
        new_args: Union[List, Tuple] = (eval_args,)
    else:
        new_args = eval_args
    return new_args


def _find_tool(tool_name: str, tools: Sequence[BaseTool]) -> BaseTool:
    """Find a tool by name.

    Args:
        tool_name: Name of the tool to find.

    Returns:
        Tool or StructuredTool.
    """
    for tool in tools:
        if tool.metadata.name == tool_name:
            return tool
    raise ValueError(f"Tool {tool_name} not found.")


def _get_dependencies_from_graph(idx: int, tool_name: str, args: str) -> List[int]:
    """Get dependencies from a graph."""
    if tool_name == "join":
        # depends on the previous step
        dependencies = list(range(1, idx))
    else:
        # define dependencies based on the dependency rule in tool_definitions.py
        dependencies = [i for i in range(1, idx) if default_dependency_rule(i, args)]

    return dependencies


def instantiate_new_step(
    tools: Sequence[BaseTool],
    idx: int,
    tool_name: str,
    args: str,
    thought: str,
) -> LLMCompilerTask:
    """Instantiate a new step."""
    dependencies = _get_dependencies_from_graph(idx, tool_name, args)
    args_list = parse_llm_compiler_action_args(args)
    if tool_name == "join":
        # tool: Optional[BaseTool] = None
        # assume that the only tool that returns None is join
        tool: BaseTool = FunctionTool.from_defaults(fn=lambda x: None)
    else:
        tool = _find_tool(tool_name, tools)

    return LLMCompilerTask(
        idx=idx,
        name=tool_name,
        tool=adapt_to_async_tool(tool),
        args=args_list,
        dependencies=dependencies,
        # TODO: look into adding a stringify rule
        # stringify_rule=stringify_rule,
        thought=thought,
        is_join=tool_name == "join",
    )


def get_graph_dict(
    parse_results: List[LLMCompilerParseResult],
    tools: Sequence[BaseTool],
) -> Dict[int, Any]:
    """Get graph dict."""
    graph_dict = {}

    for parse_result in parse_results:
        # idx = 1, function = "search", args = "Ronaldo number of kids"
        # thought will be the preceding thought, if any, otherwise an empty string
        # thought, idx, tool_name, args, _ = match
        idx = int(parse_result.idx)

        task = instantiate_new_step(
            tools=tools,
            idx=idx,
            tool_name=parse_result.tool_name,
            args=parse_result.args,
            thought=parse_result.thought,
        )

        graph_dict[idx] = task
        if task.is_join:
            break

    return graph_dict


def generate_context_for_replanner(
    tasks: Dict[int, LLMCompilerTask], joiner_thought: str
) -> str:
    """Generate context for replanning.

    Formatted like this.
    ```
    1. action 1
    Observation: xxx
    2. action 2
    Observation: yyy
    ...
    Thought: joinner_thought
    ```
    """
    previous_plan_and_observations = "\n".join(
        [
            task.get_thought_action_observation(
                include_action=True, include_action_idx=True
            )
            for task in tasks.values()
            if not task.is_join
        ]
    )
    joiner_thought = f"Thought: {joiner_thought}"
    # use f-string instead
    return f"{previous_plan_and_observations}\n\n{joiner_thought}"


def format_contexts(contexts: Sequence[str]) -> str:
    """Format contexts.

    Taken from https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/llm_compiler.py

    Contexts is a list of context.
    Each context is formatted as the description of generate_context_for_replanner
    """
    formatted_contexts = ""
    for context in contexts:
        formatted_contexts += f"Previous Plan:\n\n{context}\n\n"
    formatted_contexts += "Current Plan:\n\n"
    return formatted_contexts
