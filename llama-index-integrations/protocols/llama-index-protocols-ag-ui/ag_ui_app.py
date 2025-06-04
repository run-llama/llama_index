# TODO: Delete this file before merging the PR

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.protocols.ag_ui.server import get_ag_ui_agent_router
from fastapi import FastAPI
from typing import Annotated


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


def change_background(
    background: Annotated[str, "The background. Prefer gradients."],
) -> str:
    """ "Change the background color of the chat. Can be anything that the CSS background attribute accepts. Regular colors, linear of radial gradients etc."""
    return f"Changing background to {background}"


agent = FunctionAgent(
    tools=[change_background],
    llm=OpenAI(model="gpt-4.1"),
)


app = FastAPI()
app.include_router(get_ag_ui_agent_router(agent))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
