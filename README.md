# GPT Tree Index

A tree-based index containing text data that is created using GPT-3 and can be traversed using GPT-3 in order to answer queries.

## Overview

GPT-3 is a phenomenonal piece of technology that at its core takes in text input and is able to generate text output. It is a very simple but general paradigm, and GPT-3 (especially the latest iterations) is able to execute this amazingly well. It is able to perform many tasks in a zero-shot setting, from sentiment analysis to categorization to question answering.

However, one fundamental limitation of GPT-3 is the context size. The most sophisticated model, Davinci, has a combined input+completion limit of 4096 tokens. This is large, but not infinite. As a result, the ability to feed "knowledge" to GPT-3 is restricted to this limited prompt size and model weights - these model weights by default encode world knowledge through the training process, but can also be finetuned for custom tasks (which can be very expensive).

But what if GPT-3 can have access to potentially a much larger database of knowledge for use in say, question-answering tasks? That's where the **GPT Tree Index** comes in. 

**How It Works**
The GPT Tree Index first takes in a large dataset of unprocessed text data as input. It then builds up a tree-index in a bottom-up fashion; each parent node is able to summarize the children nodes using a general **summarization prompt**; each intermediate node contains text summarizing the components below. Once the index is built, it can be saved to disk as a JSON and loaded for future use. 

Then, say the user wants to use GPT-3 to answer a question. Using a **query prompt template**, GPT-3 will be able to recursively perform tree traversal in a top-down fashion in order to answer a question. For example, in the very beginning GPT-3 is tasked with selecting between *n* top-level nodes which best answers a provided query, by outputting a number as a multiple-choice problem. The GPT Tree Index then uses the number to select the corresponding node, and the process repeats recursively among the children nodes until a leaf node is reached.

The high-level design exercise of this project is to test the capability of GPT-3 as a general-purpose processor. A somewhat handwavy anaology is that a CPU processor has limited memory of its own but is able to have access to a wider base of stored knowledge (e.g. in RAM, and then on dks) in order to achieve the broader goal. We are making one step in this direction with the GPT Index, by having GPT build its index and traverse its index through repeated processing in order to perform a question-answering task.


## Example Usage

An example is provided in `examples/test_wiki/TestNYC.ipynb`. To build the index do the following:
```python
from gpt_index.index import GPTIndex
GPTIndex.from_input_dir('data')
```

To save to disk and load from disk, do
```python
# save to disk
index.save_to_disk('index.json')
# load from disk
index = GPTIndex.load_from_disk('index.json')
```

To query,
```python
index.query("<question_text>?")
```


## Additional Thoughts / FAQ

**How is this better than an embeddings-based approach / other state-of-the-art QA and retrieval methods?**

The intent is not to compete against existing methods. A simpler embedding-based technique could be to just encode each chunk as an embedding and do a simple question-document embedding look-up to retrieve the result. Instead, this project is intended as a design exercise to test how GPT can organize information and lookup information purely through the text-in/text-out paradigm.

**Why build a tree? Why not just incrementally go through each chunk?**

Algorithmically speaking, $O(\log N)$ is better than $O(N)$.

More broadly, building a tree helps us to test GPT's capabilities in modeling information in a hierarchy. It seems to me that our brains organize information in a similar way (citation needed). We can use this design to test how GPT can use its own hierarchy to answer questions.

Practically speaking, it is much cheaper to do so and I want to limit my monthly spending (see below for costs).

**This work is very similar to X paper/project.**

Please let me know! I am not up-to-date on the latest NLP/LLM ArXiv papers or Github projects. I will give the appropriate references/credit below.

**Does this actually work?**

Kind of. It works for simple queries, such as the prompt provided for the NYC Wikipedia data above ("What are the three main airports?"). Sometimes it fails in frustrating ways, where the correct node to choose given the query is obvious but GPT stil picks another node for some unforseen reason (for instance, given a query prompt on "What are the main ethnicities within NYC?", GPT-3 somehow picks a node which summarizes the architecture styles within Brooklyn). Some of this can be fixed with prompt tuning; this is an active area of work! 

**How much does this cost to run?**

We currently use the Davinci model for good results. Unfortunately Davinci is quite expensive. The cost of building the tree is roughly 
$cN\log(N)\frac{p}{1000}$, where $p=4096$ is the prompt limit and $c$ is the cost per 1000 tokens ($0.02 as mentioned on the [pricing page](https://openai.com/api/pricing/)). The cost of querying the tree is roughly 
$c\log(N)\frac{p}{1000}$.

For the NYC example, this equates to \$~0.40 per query.

## Dependencies

The main third-party package requirements are `transformers`, `openai`, and `langchain`.

All requirements should be contained within the `setup.py` file. To run the package locally without building the wheel, simply do `pip install -r requirements.txt`. 

## Future Directions
- Add ability to insert/delete.
- Add ability to more easily customize summarization and query prompts.
- It would be super interesting to build different trees from the same pool of raw data with different summarization prompts in order to solve task-specific needs. For instance, perhaps one method of summarization is better suited for answering questions about specific numbers. Another method of summarization could be to answer cause-effect questions.
- Similarly, it would be very interesting to explore query prompts that allow more flexible querying traversals than purely a top-down linear approach!
- Add different index structures beyond trees.
- Add ability for GPT itself to reason about connections between nodes.
