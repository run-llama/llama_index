import os

import pytest

CI = os.getenv("CI", "").lower() in ("1", "true", "yes")

try:
    from llama_index.llms.openai import OpenAI  # noqa: F401

    has_openai = True
except ImportError:
    has_openai = False


@pytest.mark.skipif(CI or not has_openai, reason="Skipping in CI environment")
@pytest.mark.asyncio
async def test_return_direct_e2e():
    from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
    from llama_index.core.tools import FunctionTool
    from llama_index.core.workflow import Context
    from llama_index.llms.openai import OpenAI

    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "sk-" + (
        "a" * 48
    ):
        pytest.skip("OPENAI_API_KEY is not defined")

    llm = OpenAI()

    def return_direct_tool(previously_failed: bool = False):
        """
        Test Tool to check return direct function

        Args:
            previously_failed (bool, optional): Set to true if previous calls return an error. Defaults to False.

        Raises:
            RuntimeError: Raised if Tool Call Error, Retry this tool with previously_failed=True if this happened

        Returns:
            str: Tool Success Status

        """
        if not previously_failed:
            raise RuntimeError(
                "Tool Failed, run again with `previously_failed=True` to success"
            )
        return "Tool Run is Success"

    agent = FunctionAgent(
        name="e2e_test_agent",
        description="Test agent for testing",
        tools=[FunctionTool.from_defaults(fn=return_direct_tool, return_direct=True)],
        llm=llm,
        system_prompt="You are a test agent for testing function call. Execute the tools requested as-is so that the results can be evaluated",
    )
    ctx = Context(agent)

    handler = agent.run(
        "Run the return_direct_tool(False), observe the output. if error, retry based on the error message",
        ctx=ctx,
    )

    ever_error = False
    ever_succeed = False
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            if ev.tool_output.is_error:
                assert (
                    ev.tool_output.raw_output
                    == "Tool Failed, run again with `previously_failed=True` to success"
                )
                ever_error = True
            else:
                assert ev.tool_output.raw_output == "Tool Run is Success"
                ever_succeed = True

    result = await handler

    assert result
    assert ever_error
    assert ever_succeed


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
