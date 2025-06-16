# LlamaIndex Protocols AG UI Integration

```bash
pip install llama-index-protocols-ag-ui
```

The `llama-index-protocols-ag-ui` package provides a factory function for creating a FastAPI router that communicates using the [AG UI Protocol](https://github.com/ag-ui-protocol/ag-ui).

Using this package, you can quickly create a FastAPI app that can be used to communicate with AG-UI compatible frameworks like [CopilotKit](https://docs.copilotkit.ai/).

### Usage

The `get_ag_ui_workflow_router` function is a factory function that creates a FastAPI router that can be used to communicate with AG-UI compatible frameworks like [CopilotKit](https://docs.copilotkit.ai/).

The router is configured with the following parameters:

- `llm`: The LLM to use for the agent.
- `frontend_tools`: Tools that are available to execute on the frontend.
- `backend_tools`: Tools that are available to execute on the backend.
- `system_prompt`: The system prompt to use for the agent.
- `initial_state`: The initial state to use for the agent. Typically the state is then interacted with by the frontend.

```python
import uvicorn
from fastapi import FastAPI

from llama_index.llms.openai import OpenAI
from llama_index.protocols.ag_ui.server import get_ag_ui_workflow_router
from typing import Annotated


# This tool has a client-side version that is actually called to change the background
def change_background(
    background: Annotated[str, "The background. Prefer gradients."],
) -> str:
    """Change the background color of the chat. Can be anything that the CSS background attribute accepts. Regular colors, linear of radial gradients etc."""
    return f"Changing background to {background}"


agentic_chat_router = get_ag_ui_workflow_router(
    llm=OpenAI(model="gpt-4.1"),
    frontend_tools=[change_background],
    backend_tools=[],
    system_prompt="You are a helpful assistant that can change the background color of the chat.",
    initial_state=None,  # Unused in this example
)


app = FastAPI(title="AG-UI Llama-Index Endpoint")

app.include_router(agentic_chat_router, prefix="/agentic_chat")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
```

Then on the frontend, you might have setup a CopilotKit app like this:

```typescript
"use client";
import React, { useState } from "react";
import "@copilotkit/react-ui/styles.css";
import "./style.css";
import { useCopilotAction } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";

interface AgenticChatProps {
  params: Promise<{
    integrationId: string;
  }>;
}

const Chat = () => {
  const [background, setBackground] = useState<string>("--copilot-kit-background-color");

  useCopilotAction({
    name: "change_background",
    description:
      "Change the background color of the chat. Can be anything that the CSS background attribute accepts. Regular colors, linear of radial gradients etc.",
    parameters: [
      {
        name: "background",
        type: "string",
        description: "The background. Prefer gradients.",
      },
    ],
    handler: ({ background }) => {
      setBackground(background);
    },
  });

  return (
    <div className="flex justify-center items-center h-full w-full" style={{ background }}>
      <div className="w-8/10 h-8/10 rounded-lg">
        <CopilotChat
          className="h-full rounded-2xl"
          labels={{ initial: "Hi, I'm an agent. Want to chat?" }}
        />
      </div>
    </div>
  );
};
```

Check out the [CopilotKit Documentation]() for more details on using AG-UI with CopilotKit+LlamaIndex.
