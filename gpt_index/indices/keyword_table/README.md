## 🔑 GPTKeywordIndex


GPTKeywordIndex is a keyword-based table data structure (inspired by "hash tables"). During index construction, GPTKeywordIndex first takes in a dataset of text documents as input, and chunks them up into smaller document chunks. For each text chunk, GPTKeywordIndex uses GPT to extract a set of relevant keywords with a **keyword extraction prompt**. (keywords can include short phrases, like "new york city"). These keywords are then stored in a table, referencing the same text chunk.

During query-time, the GPTKeywordIndex extracts a set of relevant keywords from the query using a customized variant of the same **keyword extraction prompt**. These keywords are then used to fetch the set of candidate text chunk ID's. The text chunk ID's are ordered by number of matching keywords (from highest to lowest), and truncated after a cutoff $d$, which represents the maximum number of text chunks to consider. 

We construct an answer using the *create and refine* paradigm. An initial answer to the query is constructed using the first text chunk. The answer is then *refined* through feeding in subsequent text chunks as context. Refinement could mean keeping the original answer, making small edits to the original answer, or rewriting the original answer completely.

**Runtime**

Worst-case runtime to execute a query should be $O(k*c)$, where $k$ is the number of extracted keywords, and $c$ is the number of text chunks per query.

However the number of queries to GPT is limited by $O(d)$, where $d$ is a 
user-specified parameter indicating the maximum number of text chunks to query.

**How much does this cost to run?**

Assuming `num_chunks_per_query=10`, then this equates to \$~0.40 per query.

