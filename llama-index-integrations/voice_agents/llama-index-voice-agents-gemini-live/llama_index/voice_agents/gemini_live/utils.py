from llama_index.core.tools import BaseTool
from typing import Dict, List, Callable, Any
from google.genai import types


def tool_to_fn(
    tool: BaseTool,
) -> Callable[[Dict[str, Any], str, str], types.FunctionResponse]:
    def fn(args: Dict[str, Any], id_: str, name: str) -> types.FunctionResponse:
        return types.FunctionResponse(
            id=id_, name=name, response={"result": tool(**args).raw_output}
        )

    return fn


def tools_to_gemini_tools(
    tools: List[BaseTool],
) -> List[Dict[str, List[Dict[str, str]]]]:
    d = {"function_declarations": []}
    for tool in tools:
        d["function_declarations"].append(
            {
                "name": tool.metadata.get_name(),
                "description": tool.metadata.description,
                "parameters": tool.metadata.get_parameters_dict(),
            }
        )
    return [d]


def tools_to_functions_dict(
    tools: List[BaseTool],
) -> Dict[str, Callable[[Dict[str, Any], str, str], types.FunctionResponse]]:
    tools_dict = {}
    for tool in tools:
        tools_dict.update({tool.metadata.get_name(): tool_to_fn(tool)})
    return tools_dict
