# 📨 Response Synthesizers

## Concept
A `Response Synthesizer` is what generates a response from an LLM, using a user query and a given set of text chunks. The method for doing this can take many forms, from as simple as iterating over text chunks, to as complex as building a tree. The main idea here is to simplify the process of generating a response using an LLM across your data.

In a LlamaIndex query engine, the response synthesizer is used once nodes are retrieved from a retriever, and any node-postprocessors are ran

There are serveral pre-made response synthesizers available:

- `Refine` - Send text chunks one at a time, and refining an answer after the first LLM response. This is done by getting an initial answer, and then asking the LLM to either update or repeat the answer after reading the next text chunk.
- `CompactAndRefine` (default) - The same as `Refine`, but puts as much text as possible into each LLM call, rather than running purely sequentially. 
- `Accumulate` - 
- `CompactAndAccumulate` -
- `TreeSummarize` - 
- `SimpleSummarize`

## Usage Pattern
Get started with:
```python
chat_engine = index.as_chat_engine()
response = chat_engine.chat("Tell me a joke.")
```

```{toctree}
---
maxdepth: 2
---
usage_pattern.md
```


## Modules
Below you can find corresponding tutorials to see the available chat engines in action. 

```{toctree}
---
maxdepth: 2
---
modules.md
```