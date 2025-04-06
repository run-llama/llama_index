from dotenv import load_dotenv
import os
import openai

# Load environment variables from the .env file in the current directory
load_dotenv()

# Retrieve API keys from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
pplx_api_key = os.getenv("PPLX_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

from llama_index.tools.perplexity.base import PerplexityToolSpec
from llama_index.agent.openai import OpenAIAgent

# Initialize your Perplexity tool with the PPLX_API_KEY
perplexity_tool = PerplexityToolSpec(api_key=pplx_api_key)

# Call chat_completion and specify that the model should be "sonar-pro"
response = perplexity_tool.chat_completion(
    "What is the latest news in the US today?",
    model="sonar-pro"
)

print(response)