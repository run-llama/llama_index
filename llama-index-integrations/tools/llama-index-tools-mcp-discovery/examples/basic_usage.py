import asyncio
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import MockLLM
from llama_index.tools.mcp_discovery import MCPDiscoveryTool

async def main():
    # 1. Initialize the Tool Spec
    # In a real scenario, this URL would point to your running MCP server or discovery service
    print("Initializing MCP Discovery Tool...")
    discovery_tool = MCPDiscoveryTool(api_url="https://demo.mcp-server.com/api")

    # 2. Convert the spec into a list of FunctionTools
    agent_tools = discovery_tool.to_tool_list()
    print(f"Loaded tools: {[t.metadata.name for t in agent_tools]}")

    # 3. Initialize the Agent
    # We use MockLLM here for demonstration purposes
    llm = MockLLM()
    agent = ReActAgent.from_tools(agent_tools, llm=llm, verbose=True)

    print("\nAgent is ready! It can now call 'discover_tools' when asked to find new capabilities.")

    # Example async interaction (Mocked)
    # response = await agent.achat("I need a tool to calculate the square root of a number.")

if __name__ == "__main__":
    asyncio.run(main())
