# Response Modes

Right now, we support the following options:
- `default`: "create and refine" an answer by sequentially going through each retrieved `Node`; 
    This makes a separate LLM call per Node. Good for more detailed answers.
- `compact`: "compact" the prompt during each LLM call by stuffing as 
    many `Node` text chunks that can fit within the maximum prompt size. If there are 
    too many chunks to stuff in one prompt, "create and refine" an answer by going through
    multiple prompts.
- `tree_summarize`: Given a set of `Node` objects and the query, recursively construct a tree 
    and return the root node as the response. Good for summarization purposes.
- `no_text`: Only runs the retriever to fetch the nodes that would have been sent to the LLM, 
    without actually sending them. Then can be inspected by checking `response.source_nodes`.
    The response object is covered in more detail in Section 5.
- `accumulate`: Given a set of `Node` objects and the query, apply the query to each `Node` text
    chunk while accumulating the responses into an array. Returns a concatenated string of all
    responses. Good for when you need to run the same query separately against each text
    chunk.

See [Response Synthesizer](/core_modules/query_modules/response_synthesizers/root.md) to learn more.