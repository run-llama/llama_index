# Setup OpenAI Agent
import openai
import os
from typing import List
from pydantic import BaseModel
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.scrapegraph import ScrapegraphToolSpec

# Set your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
SCRAPEGRAPH_API_KEY = "your-scrapegraph-api-key"

# Initialize the agent with ScrapegraphToolSpec
agent = OpenAIAgent.from_tools(
    ScrapegraphToolSpec().to_tool_list(),
    verbose=True,
)

# Clear chat history
agent.chat_history.clear()

# Example 1: Using smartscraper with a schema
class ArticleSchema(BaseModel):
    title: str
    description: str
    author: str

# Define the schema for scraping
schema = [ArticleSchema]

# Example of smartscraper
response = agent.chat(
    f"""Use scrapegraph_smartscraper to extract the title, description, and author from 
    https://www.wired.com/latest with this prompt: 
    'Extract the main article title, description, and author'"""
)
print("Smartscraper Response:", response)

# Example 2: Using markdownify
response = agent.chat(
    f"""Use scrapegraph_markdownify to convert the content from 
    https://www.wired.com/latest into markdown format"""
)
print("\nMarkdownify Response:", response)

# Example 3: Using local_scrape
sample_text = """
Product: iPhone 13 Pro
Price: $999
Storage: 128GB
Color: Sierra Blue
Release Date: September 24, 2021
"""

response = agent.chat(
    f"""Use scrapegraph_local_scrape to extract structured data from this text:
    {sample_text}"""
)
print("\nLocal Scrape Response:", response)

# Example of chaining operations
response = agent.chat(
    """First use markdownify on https://www.wired.com/latest to get the content,
    then use local_scrape to extract key information from that markdown content."""
)
print("\nChained Operations Response:", response)

# Example of interactive conversation
response = agent.chat(
    """What are the main topics discussed in the latest articles on 
    https://www.wired.com/latest? Please analyze the content using the available tools."""
)
print("\nInteractive Analysis Response:", response)
