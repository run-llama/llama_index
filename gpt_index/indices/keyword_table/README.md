## GPTKeywordIndex



**Runtime**

Worst-case runtime to execute a query should be $O(k*c)$, where $k$ is the number of extracted keywords, and $c$ is the number of text chunks per query.

However the number of queries to GPT is limited by $O(d)$, where $d$ is a 
user-specified parameter indicating the maximum number of text chunks to query.

**How much does this cost to run?**

Assuming `num_chunks_per_query=10`, then this equates to \$~0.40 per query.

