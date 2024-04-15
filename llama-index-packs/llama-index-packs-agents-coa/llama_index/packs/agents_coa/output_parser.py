"""Chain-of-Abstraction Output Parser."""

import asyncio
import json
import networkx as nx
import re
from collections import defaultdict
from typing import Dict, Tuple

from llama_index.core.tools import AsyncBaseTool, ToolOutput
from llama_index.core.types import BaseOutputParser


class ChainOfAbstractionParser(BaseOutputParser):
    """Chain of abstraction output parser.

    This parser is used to parse the output using the default prompt
    defined in prompts.py.

    If the prompt formatting changes the function format, this parser
    will not work and should be updated.
    """

    def __init__(self, verbose: bool = False):
        """Init params."""
        self._verbose = verbose

    def parse(
        self, solution: str, tools_by_name: Dict[str, AsyncBaseTool]
    ) -> Tuple[str, int]:
        return asyncio.run(self.aparse(solution, tools_by_name))

    async def aparse(
        self, solution: str, tools_by_name: Dict[str, AsyncBaseTool]
    ) -> Tuple[str, int]:
        # Extract function calls and placeholders
        func_calls = re.findall(r"\[FUNC (\w+)\((.*?)\) = (\w+)\]", solution)

        placeholders = set()
        for match in re.finditer(r"\[FUNC (\w+)\((.*?)\) = (\w+)\]", solution):
            placeholders.add(match.group(3))

        # Create a dependency graph
        graph = nx.DiGraph()
        for func_name, inputs, output in func_calls:
            inputs = json.loads("[" + inputs + "]")
            graph.add_node(output, func_name=func_name, inputs=inputs)
            for inp in inputs:
                graph.add_edge(inp, output)

        # Find the execution levels
        execution_levels = defaultdict(list)
        for node in nx.topological_sort(graph):
            level = (
                max(
                    [execution_levels[pred] for pred in graph.predecessors(node)],
                    default=-1,
                )
                + 1
            )
            execution_levels[node] = level

        # Group nodes by execution level
        level_groups = defaultdict(list)
        for node, level in execution_levels.items():
            level_groups[level].append(node)

        # Execute functions and replace placeholders
        results = {}
        tool_outputs = []
        graph_nodes = {node[0]: node[1] for node in graph.nodes(data=True)}
        for level in sorted(level_groups.keys()):
            level_nodes = level_groups[level]
            parallel_results = {}
            for placeholder in level_nodes:
                if len(graph_nodes[placeholder]) == 0:
                    continue

                # get function name and inputs
                func_name, inputs = (
                    graph_nodes[placeholder]["func_name"],
                    graph_nodes[placeholder]["inputs"],
                )

                # loop up any inputs that depend on other functions
                input_values = [results.get(inp, inp) for inp in inputs]
                if self._verbose:
                    print(
                        f"==== Executing {func_name} with inputs {input_values} ====",
                        flush=True,
                    )

                # execute function and store result
                try:
                    raw_tool_output = await tools_by_name[func_name].acall(
                        *input_values
                    )
                    tool_outputs.append(
                        ToolOutput(
                            content=str(raw_tool_output),
                            tool_name=func_name,
                            raw_output=raw_tool_output,
                            raw_input={"args": input_values},
                            is_error=False,
                        )
                    )
                except Exception as e:
                    tool_outputs.append(
                        ToolOutput(
                            content=str(e),
                            tool_name=func_name,
                            raw_output=None,
                            raw_input={"args": input_values},
                            is_error=True,
                        )
                    )

                    # If an error occurs, stop execution
                    break

                parallel_results[placeholder] = str(raw_tool_output)
            results.update(parallel_results)

        # Replace placeholders in the solution text
        for placeholder, value in results.items():
            solution = solution.replace(f"{placeholder}", '"' + str(value) + '"')

        return solution, tool_outputs
