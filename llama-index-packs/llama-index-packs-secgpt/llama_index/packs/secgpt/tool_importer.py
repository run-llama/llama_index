"""
To better manage tools, we introduce a class called ToolImporter, which is used for importing and managing tool usage in SecGPT. Moreover, we also define some tool helper functions for spoke definition.
"""

from llama_index.core.tools import FunctionTool


class ToolImporter:
    """
    A class to manage the importing and usage of tools in SecGPT.

    Attributes:
        tools (list): A list of tools.
        tool_name_obj_dict (dict): A dictionary mapping tool names to tool objects.
        tool_functions (dict): A dictionary mapping tool names to their functions.

    """

    def __init__(self, tools, tool_specs=[]) -> None:
        """
        Initialize the ToolImporter with tools and tool specifications.

        Args:
            tools (list): A list of tools.
            tool_specs (list, optional): A list of tool specifications. Defaults to [].

        """
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

    def get_all_tools(self):
        """
        Get the list of all tool objects.

        Returns:
            list: A list of all tool objects.

        """
        return self.tools

    def get_tool_names(self):
        """
        Get the list of all available tool names.

        Returns:
            list: A list of available tool names.

        """
        return [tool.metadata.name for tool in self.tools]

    def get_collab_functions(self, tool_name=None):
        """
        Get the list of available functionalities excluding the specified tool.

        Args:
            tool_name (str, optional): The name of the tool to exclude. Defaults to None.

        Returns:
            list: A list of available functionalities.

        """
        if tool_name:
            return [
                tool.metadata.name
                for tool in self.tools
                if tool.metadata.name != tool_name
            ]
        else:
            return [tool.metadata.name for tool in self.tools]

    def get_tool_spec(self, function):
        """
        Get the specification of a specific tool function.

        Args:
            function (str): The name of the tool function.

        Returns:
            dict: The tool function's specifications.

        """
        tool_obj = self.tool_name_obj_dict[function]
        return tool_obj.metadata.get_parameters_dict()

    def get_tool_by_name(self, tool_name):
        """
        Get the tool object by its name.

        Args:
            tool_name (str): The name of the tool.

        Returns:
            FunctionTool: The tool object.

        """
        return self.tool_name_obj_dict[tool_name]

    def get_tool_functions(self):
        """
        Get the mapping of tool functions.

        Returns:
            dict: A dictionary mapping tool names to their functions.

        """
        return self.tool_functions

    def get_tool_info(self):
        """
        Get the information of all tools.

        Returns:
            str: A string containing the tool information.

        """
        return "\n".join(
            [
                f"{tool.metadata.name}: {tool.metadata.description}"
                for tool in self.tools
            ]
        )


def create_function_placeholder(function_names):
    """
    Create placeholders for functions.

    Args:
        function_names (list): A list of function names.

    Returns:
        list: A list of FunctionTool placeholders.

    """
    func_placeholders = []
    for func in function_names:
        func_placeholder = FunctionTool.from_defaults(
            fn=(lambda *args, **kwargs: None), name=func, description=func
        )
        func_placeholders.append(func_placeholder)
    return func_placeholders


def create_message_spoke_tool():
    """
    Create a tool for messaging between spoke_operator and spoke LLM.

    Returns:
        FunctionTool: The message spoke tool.

    """

    def message_spoke(message: str):
        return message

    return FunctionTool.from_defaults(
        fn=message_spoke,
        name="message_spoke",
        description="send message from the spoke_operator to the spoke LLM",
    )
