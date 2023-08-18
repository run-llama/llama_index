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
/end_to_end_tutorials/dev_practices/UX_patterns.md
```

## The Development Pathway

It helps to start with a discovery phase of understanding your data and doing some identification of issues and corner cases as you interact with the system. 

Over time, try to formalize processes and evaluation methodology and develop your system, setting up tools for observability, debugging and experiment tracking, and eventually production monitoring.

## The Challenges of Building a Production-Ready LLM Application
Many who are interested in the LLM application space are not machine learning engineers but rather software developers or  non-technical folk. 

One of the biggest strides forward that LLMs and foundation models have made to the AI/ML application landscape is that it makes it really easy to go from idea to prototype without facing all of the hurdles and uncertainty of a traditional machine learning project - collecting, exploring and cleaning data, keeping up with latest research and exploring different methods, training models, adjusting hyperparameters, and dealing with unexpected issues in model quality. The huge infrastructure burden, long development cycle, and high risk to reward ratio have been blockers to successful applications.

At the same time, despite the fact that getting a prototype working quickly through a framework like LlamaIndex has become a lot more accessible, deploying a machine learning product in the real world is still rife with uncertainty and challenges.

### Quality and User Interaction
On the tamer side, one may face quality issues, and in the worse case, one may be liable to losing user trust if the application proves itself to be unreliable. 

We've already seen a bit of this with ChatGPT - despite its life-likeness and seeming ability to understand our conversations and requests, it often makes things up ("hallucinates"). It's not connected to the real world, data, or other digital applications. 

## Tradeoffs in LLM Application Development
There are a few tradeoffs in LLM application development:
1. **Cost** - more powerful models may be more expensive
2. **Latency** - more powerful models may be slower
3. **Simplicity** (one size fits all) - how powerful and flexible is the model / pipeline?
4. **Reliability / Useability** - is my application working at least in the general case? Is it ready for unstructured user interaction? Have I covered the major usage patterns?

LLM infra improvements are progressing quickly and we expect cost and latency to go down over time.
  
Here are some additional concerns:
1. **Evaluation** - Once I start diving deeper into improving quality, how can I evaluate individual components? How can I keep track of issues and track whether / how they are being improved over time as I change my application?
2. **Data-Driven** - How can I automate more of my evaluation and iteration process? How do I start small and improve over time? How can I manage the complexity while keeping track of my guiding light of providing the best user experience? 
3. **Customization / Complexity Tradeoff** - Can I add additional structure. How do I improve each stage of the pipeline - preprocessing and feature extraction, retrieval, generation? How can I break down this goal into more measurable and trackable sub-goals?

Differences between **Evaluation** and being **Data-Driven**:
1. **Evaluation** is not necessarily a rigorous or full data-driven process. It is more concerned with the initial *development* phase of the application - validating that the overall pipeline and starting to define possible signals and metrics which may be carried forward when shifting gears into production.
2. Being **Data-Driven** is closely tied to *automation*. After we've chosen our basic application structure, how can we improve the system over time? How can we ensure the quality of the system in a systematic way? How can we reduce the cost of monitoring and improving the system, and what are the pathways to adding and curating data points to add? How can we leverage ML systems to make this process easier?

Additional considerations:
1. **Privacy** - how can I ensure that my data is not leaked if I am feeding it into these models? What infrastructure am I using and what is the security guarantee / how is the access control structured?

## Development Hurdles

Here are some potential problems you may encounter when developing your LLM application which may lead to unsatisfactory results.

### Retrieval

1. **Out of Domain:**
If your data is extremely specific (medical, legal, scientific, financial, or other documents with technical lingo), it may be worth:
    - trying out alternate embeddings 
      - Check the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
      - You may configure a local embedding model [with the steps here](local-embedding-models)
    - testing out fine-tuning of embeddings
        - Tools: [setfit](https://github.com/huggingface/setfit)
        - Anecdotally, we have seen retrieval accuracy improve by ~12% by curating a small annotated dataset from production data
        - Even synthetic data generation without human labels has been shown to improve retrieval metrics across similar documents in train / val sets.
        - More detailed guides and case studies will come soon.
    - testing out sparse retrieval methods (see ColBERT, SPLADE)
        - these methods have been shown to generalize well to out of domain data
        - that are starting to be available in some enterprise systems (e.g. [Elastic Search's ELSeR](https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html))

## Case Studies and Resources
1. (Course) [Data-Centric AI (MIT), 2023](https://www.youtube.com/playlist?list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5)