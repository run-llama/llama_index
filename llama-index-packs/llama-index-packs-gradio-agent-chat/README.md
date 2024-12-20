# Gradio Chat With Your LlamaIndex Agent

Create a LlamaIndex Agent (i.e., `BaseAgent`) and quickly chat with it using
this pack's Gradio Chatbot interface.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack GradioAgentChatPack --download-dir ./gradio_agent_chat
```

You can then inspect the files at `./gradio_agent_chat` and use them as a template for your own project!

To run the app directly, use in your terminal:

```bash
export OPENAI_API_KEY="sk-...
python ./gradio_agent_chat/base.py
```
