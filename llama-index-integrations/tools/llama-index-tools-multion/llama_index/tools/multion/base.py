"""Multion tool spec."""

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class MultionToolSpec(BaseToolSpec):
    """Multion tool spec."""

    spec_functions = ["browse"]

    def __init__(self, api_key: str) -> None:
        """Initialize with parameters."""
        from multion.client import MultiOn

        self.multion = MultiOn(api_key=api_key)

    def browse(self, cmd: str):
        """
        Browse the web using Multion
        Multion gives the ability for LLMs to control web browsers using natural language instructions.

        You may have to repeat the instruction through multiple steps or update your instruction to get to
        the final desired state. If the status is 'CONTINUE', reissue the same instruction to continue execution

        Args:
            cmd (str): The detailed and specific natural language instructrion for web browsing

        """
        return self.multion.browse(cmd=cmd, local=True)
