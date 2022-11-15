# 🌲🗂️ ️GPT Tree Index

A tree-based index containing text data that is created using GPT-3 and can be traversed using GPT-3 in order to answer queries.

## 🚀 Overview

#### Context
- GPT-3 is a phenomenonal piece of technology for knowledge generation and reasoning.
- A big limitation of GPT-3 is context size (e.g. Davinci's limit is 4096 tokens. Large, but not infinite).
- The ability to feed "knowledge" to GPT-3 is restricted to this limited prompt size and model weights.
- **Thought**: What if GPT-3 can have access to potentially a much larger database of knowledge without retraining/finetuning? 

#### Proposed Solution [WIP]
That's where the **GPT Tree Index** comes in (if we can resolve some kinks!). Instead of relying on world knowledge encoded in the model weights, the GPT Tree Index does the following:
- Uses a pre-trained GPT-3 model primarily for *reasoning*/*summarization* instead of prior knowledge
- Takes as input a large corpus of text data, uses GPT-3 to build a tree-structured index over it
- Also use GPT-3 to traverse the tree index that it created in order to answer a query

The high-level design exercise of this project is to test the capability of GPT-3 as a general-purpose processor to organize and retrieve data. From our current understanding, related works have used GPT-3 to reason with external db sources (see below); this work links reasoning with knowledge building.

## 🔧 Dependencies

The main third-party package requirements are `transformers`, `openai`, and `langchain`.

All requirements should be contained within the `setup.py` file. To run the package locally without building the wheel, simply do `pip install -r requirements.txt`. 

## 💻 Example Usage

Examples are in the `examples` folder. We currently support the following indices:
- `GPTTreeIndex`
- `GPTKeywordTableIndex`

To build a tree index do the following:
```python
from gpt_index import GPTTreeIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = GPTTreeIndex(documents)
```

To save to disk and load from disk, do
```python
# save to disk
index.save_to_disk('index.json')
# load from disk
index = GPTTreeIndex.load_from_disk('index.json')
```

To query,
```python
index.query("<question_text>?", child_branch_factor=1)
```

## Does this actually work?

It works in varying degrees depending on the index struct (tree, keyword), the data,
and the question asked.

Check out this [Twitter thread](https://twitter.com/jerryjliu0/status/1590192529286520832?s=20&t=1Ss6eJJMZzFA6y-QmSU9lw) for instance describing the tree index.


## More Details on Each Index
- [`GPTTreeIndex`](indices/tree/README.md)
- [`GPTKeywordTableIndex`](indices/keyword_table/README.md)

## ❓🧠 Additional Thoughts / FAQ

**How is this better than an embeddings-based approach / other state-of-the-art QA and retrieval methods?**

The intent is not to compete against existing methods. An embedding-based technique could be to just encode each chunk as an embedding and do a simple question-document embedding look-up to retrieve the result. 

Instead, this project is focused on providing a set of data structures to test how GPT can organize information and lookup information purely through the text-in/text-out paradigm.

**This work is very similar to X paper/project.**

Please let me know! I am not up-to-date on the latest NLP/LLM ArXiv papers or Github projects. I am happy to give references/credit below.


## ⏭️ Future Directions
Please feel free to contribute with comments, issues, PR's! 
- `GPTTreeIndex`
    - Add ability to insert/delete.
    - Build different trees from the same pool of raw data with different summarization prompts in order to solve task-specific needs. For instance, perhaps one method of summarization is better suited for answering questions about specific numbers. Another method of summarization could be to answer cause-effect questions.
    - Similarly, continue exploring query prompts that allow more flexible querying traversals than purely a top-down linear approach!
- `GPTKeywordTableIndex`
    - Explore alternate methods of generating "keywords"
- Customization
    - Add ability to more easily customize summarization and query prompts.
- Other data structures: add different index structures beyond trees/hash tables.


## 🔬 Related Work [WIP]

[Measuring and Narrowing the Compositionality Gap in Language Models, by Press et al.](https://arxiv.org/abs/2210.03350)
- Introduces a *self-ask* paradigm, which forces the model to ask and answer followup questions before answering the original question. Similar to GPT Index in that it uses GPT to reason through subproblems; the difference is that the GPT Index also tries to organize the external information as opposed to being trained on it.
- [Example (from Langchain)](https://github.com/hwchase17/langchain/blob/master/examples/self_ask_with_search.ipynb)


[ReAct: Synergizing Reasoning and Acting in Language Models, by Yao et al.](https://arxiv.org/abs/2210.03629)
- Introduces a joint reasoning and action framework in an interleaved manner. This approach of connecting to external knowledge sources is similar to our approach of having GPT traverse an externally stored index of data. ReAct has much more fluid/sophisticated ways of traversal (e.g. search, lookup, finish), whereas this project just tries building an index with simple tree-based traversal.

