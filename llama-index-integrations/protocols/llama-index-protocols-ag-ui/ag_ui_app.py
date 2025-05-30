# TODO: Delete this file before merging the PR

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.protocols.ag_ui.server import get_ag_ui_agent_router
from fastapi import FastAPI


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


def add(a: float, b: float) -> float:
    """Useful for adding two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Useful for subtracting two numbers."""
    return a - b


def divide(a: float, b: float) -> float:
    """Useful for dividing two numbers."""
    return a / b


agent = FunctionAgent(
    tools=[multiply, add, subtract, divide],
    llm=OpenAI(model="gpt-4.1"),
)


app = FastAPI()
app.include_router(get_ag_ui_agent_router(agent))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
