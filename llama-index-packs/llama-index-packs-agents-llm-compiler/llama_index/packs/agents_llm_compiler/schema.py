from typing import Any, Collection, List, Optional, Tuple, Union

from llama_index.core.tools.types import AsyncBaseTool
from pydantic import BaseModel


class LLMCompilerParseResult(BaseModel):
    """LLMCompiler parser result."""

    thought: str
    idx: int
    tool_name: str
    args: str


class JoinerOutput(BaseModel):
    """Joiner output."""

    thought: str
    answer: str
    is_replan: bool = False


def _default_stringify_rule_for_arguments(args: Union[List, Tuple]) -> str:
    if len(args) == 1:
        return str(args[0])
    else:
        return str(tuple(args))


class LLMCompilerTask(BaseModel):
    """
    LLM Compiler Task.

    Object taken from
    https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/task_fetching_unit.py.

    """

    idx: int
    name: str
    # tool: Callable
    tool: AsyncBaseTool
    args: Union[List, Tuple]
    dependencies: Collection[int]
    # TODO: look into this
    # stringify_rule: Optional[Callable] = None
    thought: Optional[str] = None
    observation: Optional[str] = None
    is_join: bool = False

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self) -> Any:
        return await self.tool.acall(*self.args)

    def get_thought_action_observation(
        self,
        include_action: bool = True,
        include_thought: bool = True,
        include_action_idx: bool = False,
    ) -> str:
        thought_action_observation = ""
        if self.thought and include_thought:
            thought_action_observation = f"Thought: {self.thought}\n"
        if include_action:
            idx = f"{self.idx}. " if include_action_idx else ""
            # if self.stringify_rule:
            #     # If the user has specified a custom stringify rule for the
            #     # function argument, use it
            #     thought_action_observation += f"{idx}{self.stringify_rule(self.args)}\n"
            # else:
            # Otherwise, we have a default stringify rule
            thought_action_observation += (
                f"{idx}{self.name}{_default_stringify_rule_for_arguments(self.args)}\n"
            )
        if self.observation is not None:
            thought_action_observation += f"Observation: {self.observation}\n"
        return thought_action_observation
