"""Function calling agent."""


from llama_index.core.agent.runner.base import AgentRunner


class FunctionCallingAgent(AgentRunner):
    """Function calling agent.

    Calls any LLM that supports function calling in a while loop until the task is complete.

    """

    # def __init__(
    #     self,
    #     tools: List[BaseTool],
    #     llm: OpenAI,
    #     memory: BaseMemory,
    #     prefix_messages: List[ChatMessage],
    #     verbose: bool = False,
    #     max_function_calls: int = 5,
    #     default_tool_choice: str = "auto",
    # )
