# LlamaIndex Tool Integration: BGPT Evidence Retrieval

This tool allows LlamaIndex agents to search for structured scientific literature using the [BGPT API](https://bgpt.pro/mcp/). 

Unlike standard search tools that only return titles or abstracts, BGPT provides deep, study-level evidence. It extracts and formats specific scientific fields from papers, making it highly effective for AI agents conducting rigorous academic or medical research.

## Features
The tool returns structured dictionaries containing:
* **Methods & Experimental Techniques**
* **Sample Size & Population Characteristics**
* **Paper Limitations & Biases**
* **Quality Scores** (Scientific quality, reproducibility, novelty)
* **Conflict of Interest Statements**
* **How to Falsify**

## Installation

```bash
pip install llama-index-tools-bgpt
```

## Usage

You can use the BGPT tool either as a standalone function or plug it into any LlamaIndex agent.

### 1. Standalone Usage

```python
from llama_index.tools.bgpt import BGPTToolSpec

# Initialize the tool
# Note: BGPT provides the first 50 results for free. 
# An API key is only required if you need higher limits.
tool_spec = BGPTToolSpec(api_key="your-optional-api-key")

# Call the search function directly
results = tool_spec.search_literature(
    query="effects of sleep deprivation on memory",
    num_results=2,
    days_back=365
)

for paper in results:
    print(f"Title: {paper.get('title')}")
    print(f"Limitations: {paper.get('paper_limitations_and_biases')}\n")
```

### 2. Usage with an Agent

```python
import os
from llama_index.tools.bgpt import BGPTToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-..."

# Initialize tool and convert to LlamaIndex format
tool_spec = BGPTToolSpec()
tools = tool_spec.to_tool_list()

# Create the agent
agent = ReActAgent.from_tools(
    tools, 
    llm=OpenAI(model="gpt-4o"), 
    verbose=True
)

# Query the agent
response = agent.chat(
    "Find a recent study on the effects of intermittent fasting on blood pressure "
    "using the BGPT tool. Summarize the sample size and the study's limitations."
)

print(response)
```

## Tool Arguments

The `search_literature` function accepts the following parameters:

* `query` (str): The specific research question or topic to search for.
* `num_results` (int, optional): The number of top results to return. Defaults to 5.
* `days_back` (int, optional): Filter results to only include papers published within the last N days.

## Testing

To run the tests for this tool locally, ensure you have `pytest` installed, and then run the following commands from the package root:

```bash
# Install the package in editable mode
pip install -e .

# Run the test suite
pytest tests/
```