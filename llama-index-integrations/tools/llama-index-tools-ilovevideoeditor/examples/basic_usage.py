import asyncio

from llama_index.core.agent import ReActAgent
from llama_index.core.llms import MockLLM
from llama_index.tools.ilovevideoeditor import ILoveVideoEditorTool


async def main():
    # 1. Initialize the Tool Spec
    # In a real scenario, set ILOVEVIDEOEDITOR_API_KEY in the environment or
    # pass api_key="vf_live_..." here.
    print("Initializing iLoveVideoEditor Tool...")
    tool_spec = ILoveVideoEditorTool(api_key="vf_live_your_api_key")

    # 2. Convert the spec into a list of FunctionTools
    agent_tools = tool_spec.to_tool_list()
    print(f"Loaded tools: {[t.metadata.name for t in agent_tools]}")

    # 3. Initialize the Agent
    # We use MockLLM here for demonstration purposes
    llm = MockLLM()
    agent = ReActAgent.from_tools(agent_tools, llm=llm, verbose=True)

    print(
        "\nAgent is ready! It can now call 'render_video' and "
        "'get_render_status' when asked to create videos."
    )

    # Example async interaction (Mocked)
    # response = await agent.achat(
    #     "Render a 3-second video with the text 'Hello from LlamaIndex'"
    # )


if __name__ == "__main__":
    asyncio.run(main())
