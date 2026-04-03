from llama_index.core.tools import FunctionTool
from llama_index.packs.agents_coa import CoAAgentPack
from llama_index.llms.openai import OpenAI


def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)

pack = CoAAgentPack(tools=[add_tool, multiply_tool], llm=OpenAI(model="gpt-4.1-mini"))

print(pack.run("what is 123.123*101.101 and what is its product with 12345"))
