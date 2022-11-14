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

An example is provided in `examples/test_wiki/TestNYC.ipynb`. To build the index do the following:
```python
from gpt_index import GPTTreeIndex, SimpleDirectoryReader
GPTTreeIndex.from_input_dir('data')
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

Kind of, it's very much a WIP! It works for simple queries, such as the prompt provided for the Gatsby data in `examples/gatsby` ("What did the narrator do after getting back to Chicago?"). 

#### Where it breaks
In many cases it doesn't reason down the correct chain, and oftentimes it can fail in very frustrating ways. For instance, in the Paul Graham example `examples/paul_graham_essay`, when we ask "What did the author do during his time *after* Y Combinator?" and run with `verbose=True`, we find that the reasoning is oftentimes correct ("the author decided to try painting"), but the selected multiple choice answer is completely wrong, leading GPT Index down the wrong path. This is open to future work! 

Interestingly in the case of the NYC wiki dataset `examples/test_wiki`, we find that GPT oftentimes relies on its own world knowledge (it leads GPT Index down the wrong path but still surfaces the correct answer in the end e.g. "What are the airports within New York City?").


## ❓🧠 Additional Thoughts / FAQ

**More details on how it works**

The GPT Index first takes in a large dataset of unprocessed text data as input. It then builds up a tree-index in a bottom-up fashion; each parent node is able to summarize the children nodes using a general **summarization prompt**; each intermediate node contains text summarizing the components below. Once the index is built, it can be saved to disk as a JSON and loaded for future use. 

Then, say the user wants to use GPT-3 to answer a question. Using a **query prompt template**, the GPT Index will be able to recursively perform tree traversal in a top-down fashion in order to answer a question. For example, in the very beginning GPT-3 is tasked with selecting between *n* top-level nodes which best answers a provided query, by outputting a number as a multiple-choice problem. The GPT Tree Index then uses the number to select the corresponding node, and the process repeats recursively among the children nodes until a leaf node is reached.


**How is this better than an embeddings-based approach / other state-of-the-art QA and retrieval methods?**

The intent is not to compete against existing methods. A simpler embedding-based technique could be to just encode each chunk as an embedding and do a simple question-document embedding look-up to retrieve the result. Instead, this project is intended as a design exercise to test how GPT can organize information and lookup information purely through the text-in/text-out paradigm.

**Why build a tree? Why not just incrementally go through each chunk?**

Algorithmically speaking, $O(\log N)$ is better than $O(N)$.

More broadly, building a tree helps us to test GPT's capabilities in modeling information in a hierarchy. It seems to me that our brains organize information in a similar way (citation needed). We can use this design to test how GPT can use its own hierarchy to answer questions.

Practically speaking, it is much cheaper to do so and I want to limit my monthly spending (see below for costs).

**This work is very similar to X paper/project.**

Please let me know! I am not up-to-date on the latest NLP/LLM ArXiv papers or Github projects. I am happy to give references/credit below.

**How much does this cost to run?**

We currently use the Davinci model for good results. Unfortunately Davinci is quite expensive. The cost of building the tree is roughly 
$cN\log(N)\frac{p}{1000}$, where $p=4096$ is the prompt limit and $c$ is the cost per 1000 tokens ($0.02 as mentioned on the [pricing page](https://openai.com/api/pricing/)). The cost of querying the tree is roughly 
$c\log(N)\frac{p}{1000}$.

For the NYC example, this equates to \$~0.40 per query.

## ⏭️ Future Directions
Please feel free to contribute with comments, issues, PR's! 
- Add ability to insert/delete.
- Add ability to more easily customize summarization and query prompts.
- Build different trees from the same pool of raw data with different summarization prompts in order to solve task-specific needs. For instance, perhaps one method of summarization is better suited for answering questions about specific numbers. Another method of summarization could be to answer cause-effect questions.
- Similarly, would also be interesting to explore query prompts that allow more flexible querying traversals than purely a top-down linear approach!
- Add different index structures beyond trees.
- Add ability for GPT itself to reason about connections between nodes.


## 🔬 Related Work [WIP]

[Measuring and Narrowing the Compositionality Gap in Language Models, by Press et al.](https://arxiv.org/abs/2210.03350)
- Introduces a *self-ask* paradigm, which forces the model to ask and answer followup questions before answering the original question. Similar to GPT Index in that it uses GPT to reason through subproblems; the difference is that the GPT Index also tries to organize the external information as opposed to being trained on it.
- [Example (from Langchain)](https://github.com/hwchase17/langchain/blob/master/examples/self_ask_with_search.ipynb)


[ReAct: Synergizing Reasoning and Acting in Language Models, by Yao et al.](https://arxiv.org/abs/2210.03629)
- Introduces a joint reasoning and action framework in an interleaved manner. This approach of connecting to external knowledge sources is similar to our approach of having GPT traverse an externally stored index of data. ReAct has much more fluid/sophisticated ways of traversal (e.g. search, lookup, finish), whereas this project just tries building an index with simple tree-based traversal.

