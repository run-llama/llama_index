`Note`: Before running or testing the code in this notebook, ensure that you have set up the `Zep server`.

# 🧠 Zep Memory Integration with LlamaIndex Agents

This notebook demonstrates how to use [Zep Memory](https://docs.getzep.com/) with various agent types from [LlamaIndex](https://github.com/jerryjliu/llama_index), including:

- `SimpleChatEngine`
- `ReActAgent`
- `FunctionCallingAgent`
- `AgentWorkflow`

Both **synchronous** and **asynchronous** memory clients are supported and demonstrated.

---

## 📦 Install Dependencies

```bash
# pip install llama_index_memory_zep
# pip install llama-index zep-python openai
```

---

## 🔐 Environment Setup

```python
import os

os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your actual OpenAI key
```

---

## 📚 Import Required Packages

```python
import uuid
from zep_python.client import Zep, AsyncZep
from llamaindex.memory.zep import ZepMemory
from llama_index.llms.openai import OpenAI
```

---

## 🔁 Initialize Clients and IDs

```python
zep_client = Zep(api_key="mysupersecretkey", base_url="http://localhost:8000")
azep_client = AsyncZep(
    api_key="mysupersecretkey", base_url="http://localhost:8000"
)

user_id = uuid.uuid4().hex
session_id = uuid.uuid4().hex

# Register user
zep_client.user.add(user_id=user_id)

# Start memory session
zep_client.memory.add_session(session_id=session_id, user_id=user_id)
```

---

## 🧠 Initialize Zep Memory

```python
memory = ZepMemory.from_defaults(
    zep_client=zep_client, session_id=session_id, user_id=user_id
)
amemory = ZepMemory.from_defaults(
    zep_client=azep_client, session_id=session_id, user_id=user_id
)
```

---

## 🤖 LLM Setup

```python
llm = OpenAI(model="gpt-4o-mini")
```

---

## 💬 SimpleChatEngine

### ✅ Sync Example

```python
from llama_index.core.chat_engine.simple import SimpleChatEngine

agent = SimpleChatEngine.from_defaults(llm=llm, memory=memory)

agent.chat("Hi, my name is Younis")
agent.chat("What was my name?")
```

### 🌀 Async Example

```python
agent = SimpleChatEngine.from_defaults(llm=llm, memory=amemory)

agent.chat("Hi, my name is Younis")
agent.chat("What was my name?")
```

---

## 🔁 ReActAgent

### ✅ Sync Example

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(tools=[], llm=llm, memory=memory, verbose=True)

agent.chat("What's the capital of France?")
agent.chat("What was my previous question?")
```

### 🌀 Async Example

```python
agent = ReActAgent.from_tools(tools=[], llm=llm, memory=amemory, verbose=True)

agent.chat("What's the capital of France?")
agent.chat("What was my previous question?")
```

---

## ⚙️ FunctionCallingAgent

### ✅ Sync Example

```python
from llama_index.core.agent import FunctionCallingAgent

agent = FunctionCallingAgent.from_tools(
    [], llm=llm, memory=memory, verbose=True
)

agent.chat("Hi, my name is Younis")
agent.chat("What was my name?")
```

### 🌀 Async Example

```python
agent = FunctionCallingAgent.from_tools(
    [], llm=llm, memory=amemory, verbose=True
)

agent.chat("Hi, my name is Younis")
agent.chat("What was my name?")
```

---

## 🧩 AgentWorkflow

```python
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    AgentStream,
    FunctionAgent,
)

research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Responsible for synthesizing structured responses.",
    system_prompt="""
    You are the ResearchAgent. Your task is to compile and synthesize information based on context.
    Be systematic, transparent, and clear in your responses.
    """,
    llm=llm,
    tools=[],
    verbose=True,
)

agent_workflow = AgentWorkflow(
    agents=[research_agent],
    root_agent=research_agent.name,
    initial_state={"answer_content": ""},
)

# Run with sync memory
handler = agent_workflow.run(
    user_msg="Explain the heuristic function in detail.", memory=memory
)

# Stream response
current_agent = None
async for event in handler.stream_events():
    if (
        hasattr(event, "current_agent_name")
        and event.current_agent_name != current_agent
    ):
        current_agent = event.current_agent_name
        print(f"\\n{'='*50}\\n🤖 Agent: {current_agent}\\n{'='*50}\\n")
    if isinstance(event, AgentStream):
        print(event.delta, end="", flush=True)
```

### 🌀 Async Memory?

Just replace `memory=memory` with `memory=amemory`.

---

## ✅ Final Notes

- This example assumes your Zep server is running locally at `http://localhost:8000`
- All memory-aware agents should now retain previous conversation turns
- Use this setup as a base for tool-enhanced agents, longer workflows, or integrations
