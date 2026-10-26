"""
Example Famulor agent setup.

Running this file does not call the Famulor API and does not place a call.
"""

import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.famulor import FamulorToolSpec


def build_agent(api_key: str) -> FunctionAgent:
    """
    Build an agent with the Famulor tools.

    ``make_call`` places a real outbound phone call. Do not invoke it unless
    the destination should actually be dialed.
    """
    famulor_tool = FamulorToolSpec(api_key=api_key)
    return FunctionAgent(
        tools=famulor_tool.to_tool_list(),
        llm=OpenAI(model="gpt-4o"),
    )


def main() -> None:
    """Show how to configure the API key without contacting Famulor."""
    if not os.environ.get("FAMULOR_API_KEY"):
        print("Set FAMULOR_API_KEY. Create a key at https://app.famulor.io.")
    print("Import build_agent() from this example. It does not dial on import.")


if __name__ == "__main__":
    main()
