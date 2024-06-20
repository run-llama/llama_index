# Memory

We've now made several additions and subtractions to our code. To make it clear what we're using, here's the current state of our agent. It's using OpenAI for the LLM and LlamaParse to enhance parsing:

```python
from dotenv import load_dotenv

load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_parse import LlamaParse
from llama_index.core.tools import QueryEngineTool

# settings
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)


# function tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

# rag pipeline
documents = LlamaParse(result_type="markdown").load_data(
    "./data/2023_canadian_budget.pdf"
)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

budget_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="canadian_budget_2023",
    description="A RAG engine with some basic facts about the 2023 Canadian federal budget. Ask natural-language questions about the budget.",
)

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, budget_tool], verbose=True
)

response = agent.chat(
    "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?"
)

print(response)

response = agent.chat(
    "How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?"
)

print(response)

response = agent.chat(
    "How much was the total of those two allocations added together? Use a tool to answer any questions."
)

print(response)
```

As you can see, we've asked the agent 3 questions in a row. This is demonstrating a powerful feature of agents in LlamaIndex: memory. Let's see what the output looks like:

```
Started parsing the file under job_id cac11eca-45e0-4ea9-968a-25f1ac9b8f99
Thought: The current language of the user is English. I need to use a tool to help me answer the question.
Action: canadian_budget_2023
Action Input: {'input': 'How much was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?'}
Observation: $20 billion was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget.
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: $20 billion was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget.
$20 billion was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget.
Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
Action: canadian_budget_2023
Action Input: {'input': 'How much was allocated to implement a means-tested dental care program in the 2023 Canadian federal budget?'}
Observation: $13 billion was allocated to implement a means-tested dental care program in the 2023 Canadian federal budget.
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: $13 billion was allocated to implement a means-tested dental care program in the 2023 Canadian federal budget.
$13 billion was allocated to implement a means-tested dental care program in the 2023 Canadian federal budget.
Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
Action: add
Action Input: {'a': 20, 'b': 13}
Observation: 33
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: The total of the allocations for the tax credit to promote investment in green technologies and the means-tested dental care program in the 2023 Canadian federal budget is $33 billion.
The total of the allocations for the tax credit to promote investment in green technologies and the means-tested dental care program in the 2023 Canadian federal budget is $33 billion.
```

The agent remembers that it already has the budget allocations from previous questions, and can answer a contextual question like "add those two allocations together" without needing to specify which allocations exactly. It even correctly uses the other addition tool to sum up the numbers.

Having demonstrated how memory helps, let's [add some more complex tools](./tools.md) to our agent.
