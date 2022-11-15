## 🌲 GPTTreeIndex

The GPTTreeIndex first takes in a set of text documents as input. It then builds up a tree-index in a bottom-up fashion; each parent node is able to summarize the children nodes using a general **summarization prompt**; each intermediate node contains text summarizing the components below. Once the index is built, it can be saved to disk as a JSON and loaded for future use. 

Then, say the user wants to use GPT-3 to answer a question. Using a **query prompt template**, the GPTTreeIndex will be able to recursively perform tree traversal in a top-down fashion in order to answer a question. For example, in the very beginning GPT-3 is tasked with selecting between *n* top-level nodes which best answers a provided query, by outputting a number as a multiple-choice problem. The GPTTreeIndex then uses the number to select the corresponding node, and the process repeats recursively among the children nodes until a leaf node is reached.

**Why build a tree? Why not just incrementally go through each chunk?**

Algorithmically speaking, $O(\log N)$ is better than $O(N)$.

More broadly, building a tree helps us to test GPT's capabilities in modeling information in a hierarchy. It seems to me that our brains organize information in a similar way (citation needed). We can use this design to test how GPT can use its own hierarchy to answer questions.

Practically speaking, it is much cheaper to do so and I want to limit my monthly spending (see below for costs).


**How much does this cost to run?**

We currently use the Davinci model for good results. Unfortunately Davinci is quite expensive. The cost of building the tree is roughly 
$cN\log(N)\frac{p}{1000}$, where $p=4096$ is the prompt limit and $c$ is the cost per 1000 tokens ($0.02 as mentioned on the [pricing page](https://openai.com/api/pricing/)). The cost of querying the tree is roughly 
$c\log(N)\frac{p}{1000}$.

For the NYC example, this equates to \$~0.40 per query.

