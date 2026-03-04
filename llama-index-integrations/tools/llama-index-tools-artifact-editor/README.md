# Artifact Editor Tool Spec

```bash
pip install llama-index-tools-artifact-editor
```

The `ArtifactEditorToolSpec` is a stateful tool spec that allows you to edit an artifact in-memory.

Using JSON patch operations, an LLM/Agent can be prompted to create, modify, and iterate on an artifact like a report, code, or anything that can be represented as a Pydantic model.

The tool package also includes an `ArtifactMemoryBlock` that can be used to store the artifact and inject it into the LLM/Agent's memory.

## Usage

Below is an example of how to use the `ArtifactEditorToolSpec` and `ArtifactMemoryBlock` to create and iterate on a report.

```python
import asyncio
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any

from llama_index.core.agent.workflow import (
    FunctionAgent,
    AgentStream,
    ToolCallResult,
)
from llama_index.core.memory import Memory
from llama_index.tools.artifact_editor import (
    ArtifactEditorToolSpec,
    ArtifactMemoryBlock,
)
from llama_index.llms.openai import OpenAI

# Define the Artifact Pydantic Model


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    content: str = Field(description="The content of the text block")


class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    headers: List[str] = Field(description="The headers of the table")
    rows: List[List[str]] = Field(description="The rows of the table")


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    image_url: str = Field(description="The URL of the image")


class Report(BaseModel):
    """Creates an instance of a report, which is a collection of text, tables, and images."""

    title: str = Field(description="The title of the report")
    content: List[TextBlock | TableBlock | ImageBlock] = Field(
        description="The content of the report"
    )


# Initialize the tool spec and tools
tool_spec = ArtifactEditorToolSpec(Report)
tools = tool_spec.to_tool_list()

# Initialize the memory
memory = Memory.from_defaults(
    session_id="artifact_editor_01",
    memory_blocks=[ArtifactMemoryBlock(artifact_spec=tool_spec)],
    token_limit=60000,
    chat_history_token_ratio=0.7,
)

# Create the agent
agent = FunctionAgent(
    tools=tools,
    llm=OpenAI(model="o3-mini"),
    system_prompt="You are an expert in writing reports. When you write a report, I will be able to see it (and also any changes you make to it!), so no need to repeat it back to me once its written.",
)


# Run the agent in a basic chat loop
# As it runs, the artifact will be updated in-memory and
# can be accessed via the `get_current_artifact` method.
async def main():
    while True:
        user_msg = input("User: ").strip()
        if user_msg.lower() in ["exit", "quit"]:
            break

        handler = agent.run(user_msg, memory=memory)
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream):
                print(ev.delta, end="", flush=True)
            elif isinstance(ev, ToolCallResult):
                print(
                    f"\n\nCalling tool: {ev.tool_name} with kwargs: {ev.tool_kwargs}"
                )

        response = await handler
        print(str(response))
        print("Current artifact: ", tool_spec.get_current_artifact())


if __name__ == "__main__":
    asyncio.run(main())
```

When running this, you might initially ask the agent:

```
User: Create a ficticous report about the history of the internet
```

And you will get a report with a list of blocks. Try asking it to modify the report!

```
User: Move the image to the top of the report
```

And you will get a report with the image moved to the top.

Check out the documentation for more example on [agents](https://docs.llamaindex.ai/en/stable/understanding/agent/), [memory](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/), and [tools](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/).
