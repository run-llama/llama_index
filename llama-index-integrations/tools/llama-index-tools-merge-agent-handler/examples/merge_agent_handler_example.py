"""Merge Agent Handler - Connect a LlamaIndex agent to Merge Tool Packs via MCP."""

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.merge_agent_handler import MergeAgentHandlerToolSpec

# Initialize the tool spec
merge_tools = MergeAgentHandlerToolSpec(
    api_key="your-merge-api-key",
    tool_pack_id="your-tool-pack-id",
    registered_user_id="your-registered-user-id",
)

# Convert to LlamaIndex tools
tools = merge_tools.to_tool_list()

# Create agent
llm = OpenAI(model="gpt-4o")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# Query
response = agent.chat(
    "List the available tools in my Merge Tool Pack, then use the "
    "appropriate tool to fetch my recent tickets from Jira."
)
print(response)
