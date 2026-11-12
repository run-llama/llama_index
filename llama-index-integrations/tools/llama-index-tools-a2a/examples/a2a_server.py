"""
A2A Server Example

Run this to start an A2A-compatible agent server.
The agent will be discoverable at http://localhost:8000/.well-known/agent-card.json

Usage:
    python a2a_server.py
"""

import argparse
import uvicorn
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.tools.a2a import workflow_as_a2a


class ResearchStart(StartEvent):
    topic: str
    depth: int = 2


class ResearchWorkflow(Workflow):
    @step
    async def research(self, start_event: ResearchStart) -> StopEvent:
        """Perform research on the given topic."""
        result = f"Research on '{start_event.topic}' completed at depth {start_event.depth}. "
        result += "Key findings: A2A enables agent interoperability. "
        result += "MCP provides tool access. Both complement each other."
        return StopEvent(result=result)


def create_app():
    """Create the A2A-compatible FastAPI app."""
    return workflow_as_a2a(ResearchWorkflow(), url="http://localhost:8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2A Server Example")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )
    args = parser.parse_args()

    app = create_app()

    print(f"Starting A2A server at http://{args.host}:{args.port}")
    print(f"AgentCard available at http://{args.host}:{args.port}/.well-known/agent-card.json")
    print(f"JSON-RPC endpoint at http://{args.host}:{args.port}/")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )