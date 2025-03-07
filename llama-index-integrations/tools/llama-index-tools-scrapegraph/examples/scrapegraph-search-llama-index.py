"""Example of using scrapegraph search with LlamaIndex."""

from llama_index.core.agent import ReActAgent
from llama_index.tools.scrapegraph import ScrapegraphToolSpec
from llama_index.core.llms import OpenAI

# Initialize the tool spec with your API key
SCRAPEGRAPH_API_KEY = "your-api-key-here"  # Replace with your actual API key
tool_spec = ScrapegraphToolSpec()

# Create a list of tools from the tool spec
tools = tool_spec.to_tool_list()

# Initialize the LLM
llm = OpenAI(temperature=0)

# Create an agent with the tools
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)


# Example queries to demonstrate usage
def run_search_example():
    """Run an example search query using the scrapegraph tool."""
    # Example 1: Basic search
    query = "What are the latest developments in artificial intelligence?"
    response = agent.chat(
        f"Use the scrapegraph search tool to find information about: {query}"
    )
    print("\nSearch Results:")
    print(f"Query: {query}")
    print(f"Response: {response}")

    # Example 2: More specific search
    query = "What are the key features of Python 3.12?"
    response = agent.chat(
        f"Use the scrapegraph search tool to find detailed information about: {query}"
    )
    print("\nSearch Results:")
    print(f"Query: {query}")
    print(f"Response: {response}")


if __name__ == "__main__":
    # Make sure to set your API keys in environment variables
    # os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    run_search_example()
