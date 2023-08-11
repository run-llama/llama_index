# Principled Development Practices

In order to develop your application, it can help to implement some principled development practices.

We've boiled down the key practices into 3 pillars.

We will also try to do a compare and contrast with traditional software engineering and ML development, highlighting similarities and differences.

Here are the main pillars of principled development of LLM and RAG applications:

```{toctree}
---
maxdepth: 1
---
/end_to_end_tutorials/dev_practices/observability.md
/end_to_end_tutorials/dev_practices/evaluation.md
/end_to_end_tutorials/dev_practices/monitoring.md
```

## The Development Pathway

Start with a discovery phase of understanding your data and doing some identification of issues and corner cases as you interact with the system. 

Try to formalize processes and evaluation methodology and develop your system, setting up tools for observability, debugging and experiment tracking, and eventually production monitoring.

## Development Hurdles

Here are some potential problems you may encounter when developing your LLM application which may lead to unsatisfactory results.

### Retrieval

1. **Out of Domain:**
If your data is extremely specific (medical, legal, scientific, less common language), it may be worth:
    - trying out alternate embeddings 
      - Check the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
      - You may configure a local embedding model [with the steps here](local-embedding-models)
    - testing out fine-tuning of embeddings
        - Tools: [setfit](https://github.com/huggingface/setfit)
        - Anecdotally, we have seen retrieval accuracy improve by ~12% by curating a small dataset from production data
        - More detailed guides will come soon
    - testing out sparse retrieval methods (see ColBERT, SPLADE)
        - that are starting to be available in some enterprise systems (e.g. [Elastic Search's ELSeR](https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html))