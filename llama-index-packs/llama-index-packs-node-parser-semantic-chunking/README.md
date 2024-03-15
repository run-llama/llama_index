# Semantic Chunking Llama Pack

This LlamaPack implements the semantic chunking algorithm first proposed by Greg Kamradt in his [Five Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/5_Levels_Of_Text_Splitting.ipynb) tutorial.

How it works:

- Split text into sentences.
- For each sentence, generate an embedding.
- Measure cosine distance between each pair of consecutive sentences.
- Get the 95% percentile cosine distance, set that as the threshold.
- Create a new chunk if the cosine distance of a sentence compared to prev. exceeds that threshold.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack LLMCompilerAgentPack --download-dir ./llm_compiler_agent_pack
```

You can then inspect the files at `./llm_compiler_agent_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a directory. **NOTE**: You must specify `skip_load=True` - the pack contains multiple files,
which makes it hard to load directly.

We will show you how to import the agent from these files!
