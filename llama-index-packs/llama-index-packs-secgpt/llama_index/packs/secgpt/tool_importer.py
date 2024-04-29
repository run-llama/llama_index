"""
To better manage tools, we introduce a class called ToolImporter, which is used for importing and managing tool usage in SecGPT. Moreover, we also define some tool helper functions for spoke definition.
"""

from llama_index.core.tools import FunctionTool


class ToolImporter:
    # Initialize the tool importer
    def __init__(self, tools, tool_specs=[]) -> None:
        self.tool_functions = {}
        # Load individual tools
        self.tools = tools
        self.tool_name_obj_dict = {tool.metadata.name: tool for tool in tools}
        self.tool_functions = {
            tool.metadata.name: [tool.metadata.name] for tool in tools
        }

        # Load tool specs
        for tool_spec in tool_specs:
            tool_list = tool_spec.to_tool_list()
            self.tools.extend(tool_list)
            spec_tool_name_obj_dict = {tool.metadata.name: tool for tool in tool_list}
            self.tool_name_obj_dict.update(spec_tool_name_obj_dict)
            self.tool_functions[tool_spec.__class__.__name__] = list(
                spec_tool_name_obj_dict.keys()
            )

    # Get the list of tool objects
    def get_all_tools(self):
        return self.tools

    # Get the list of available tool names
    def get_tool_names(self):
        return [tool.metadata.name for tool in self.tools]

    # Get the list of available functionalities excluding the specified tool
    def get_collab_functions(self, tool_name=None):
        if tool_name:
            return [
                tool.metadata.name
                for tool in self.tools
                if tool.metadata.name != tool_name
            ]
        else:
            return [tool.metadata.name for tool in self.tools]

    # Get the specification of a specific tool_function
    def get_tool_spec(self, function):
        tool_obj = self.tool_name_obj_dict[function]
        return tool_obj.metadata.get_parameters_dict()

    # Get the tool object by name
    def get_tool_by_name(self, tool_name):
        return self.tool_name_obj_dict[tool_name]

    # Get the tool functions mapping
    def get_tool_functions(self):
        return self.tool_functions

    # Get the tool information
    def get_tool_info(self):
        return "\n".join(
            [
                f"{tool.metadata.name}: {tool.metadata.description}"
                for tool in self.tools
            ]
        )


# Create a placeholder for functions
def create_function_placeholder(function_names):
    func_placeholders = []
    for func in function_names:
        func_placeholder = FunctionTool.from_defaults(
            fn=(lambda *args, **kwargs: None), name=func, description=func
        )
        func_placeholders.append(func_placeholder)
    return func_placeholders


# Create a tool for messaging between spoke_operator and spoke llm
def create_message_spoke_tool():
    def message_spoke(message: str):
        return message

    return FunctionTool.from_defaults(
        fn=message_spoke,
        name="message_spoke",
        description="send message from the spoke_operator to the spoke LLM",
    )
