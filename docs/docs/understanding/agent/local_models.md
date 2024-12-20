# Agents with local models

If you're happy using OpenAI or another remote model, you can skip this section, but many people are interested in using models they run themselves. The easiest way to do this is via the great work of our friends at [Ollama](https://ollama.com/), who provide a simple to use client that will download, install and run a [growing range of models](https://ollama.com/library) for you.

## Install Ollama

They provide a one-click installer for Mac, Linux and Windows on their [home page](https://ollama.com/).

## Pick and run a model

Since we're going to be doing agentic work, we'll need a very capable model, but the largest models are hard to run on a laptop. We think `mixtral 8x7b` is a good balance between power and resources, but `llama3` is another great option. You can run Mixtral by running

```bash
ollama run mixtral:8x7b
```

The first time you run, it will also automatically download and install the model for you, which can take a while.

## Switch to local agent

To switch to Mixtral, you'll need to bring in the Ollama integration:

```bash
pip install llama-index-llms-ollama
```

Then modify your dependencies to bring in Ollama instead of OpenAI:

```python
from llama_index.llms.ollama import Ollama
```

And finally initialize Mixtral as your LLM instead:

```python
llm = Ollama(model="mixtral:8x7b", request_timeout=120.0)
```

## Ask the question again

```python
response = agent.chat("What is 20+(2*4)? Calculate step by step.")
```

The exact output looks different from OpenAI (it makes a mistake the first time it tries), but Mixtral gets the right answer:

```
Thought: The current language of the user is: English. The user wants to calculate the value of 20+(2*4). I need to break down this task into subtasks and use appropriate tools to solve each subtask.
Action: multiply
Action Input: {'a': 2, 'b': 4}
Observation: 8
Thought: The user has calculated the multiplication part of the expression, which is (2*4), and got 8 as a result. Now I need to add this value to 20 by using the 'add' tool.
Action: add
Action Input: {'a': 20, 'b': 8}
Observation: 28
Thought: The user has calculated the sum of 20+(2*4) and got 28 as a result. Now I can answer without using any more tools.
Answer: The solution to the expression 20+(2*4) is 28.
The solution to the expression 20+(2*4) is 28.
```

Check the [repo](https://github.com/run-llama/python-agents-tutorial/blob/main/2_local_agent.py) to see what this final code looks like.

You can now continue the rest of the tutorial with a local model if you prefer. We'll keep using OpenAI as we move on to [adding RAG to your agent](./rag_agent.md).
