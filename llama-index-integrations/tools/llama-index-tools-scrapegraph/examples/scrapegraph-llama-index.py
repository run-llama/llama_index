# Setup OpenAI Agent
import openai

openai.api_key = "sk-xxx"
from llama_index.agent.openai import OpenAIAgent

from llama_index.tools.scrapegraph import ScrapegraphToolSpec

agent = OpenAIAgent.from_tools(
    ScrapegraphToolSpec().to_tool_list(),
    verbose=True,
)
agent.chat_history.clear()
print(agent.chat("Extract the title and description from https://www.wired.com"))
