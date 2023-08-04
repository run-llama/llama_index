# Response Modes

Right now, we support the following options:

- `refine`: ***create and refine*** an answer by sequentially going through each retrieved text chunk. 
    This makes a separate LLM call per Node/retrieved chunk. The first chunk is used in a query using the 
    `text_qa_template` prompt. Then the answer and the next chunk (and the original question) are used 
    in another query with the `refine_template` prompt. And so on until all chunks have been parsed.
    Good for more detailed answers.
- `compact` (default): ***compact*** the prompt during each LLM call by stuffing as 
    many text (concatenated from the retrieved chunks) that can fit within the maximum prompt size. 
    If the text is too long to fit in one prompt, it is splitted in as many parts as needed 
    (using a `TokenTextSplitter`). Each text part is considered a "chunk" and is sent to the 
    ***create and refine*** synthesizer. In short, it is like `refine`, but with less LLM calls.
- `tree_summarize`: Given a set of text chunks and the query, recursively construct a tree 
    and return the root node as the response. All retrieved chunks/nodes are concatenated and then
    splitted to fit the context window using the `text_qa_template` prompt, resulting in as many new "chunks".
    Each of these chunks are queried against the `text_qa_template` prompt, giving as many answers. If there is
    only one answer, then it's the final answer. If there are more than one answer, these themselves are 
    considered as chunks and sent recursively to the `tree_summarize` process (concatenated/splitted-to-fit/queried).
    Good for summarization purposes.
- `simple_summarize`: Truncates all text chunks to fit into a single LLM prompt. Good for quick
    summarization purposes, but may lose detail due to truncation.
- `no_text`: Only runs the retriever to fetch the nodes that would have been sent to the LLM, 
    without actually sending them. Then can be inspected by checking `response.source_nodes`.
- `accumulate`: Given a set of text chunks and the query, apply the query to each text
    chunk while accumulating the responses into an array. Returns a concatenated string of all
    responses. Good for when you need to run the same query separately against each text
    chunk.
- `compact_accumulate`: The same as accumulate, but will "compact" each LLM prompt similar to
    `compact`, and run the same query against each text chunk.

See [Response Synthesizer](/core_modules/query_modules/response_synthesizers/root.md) to learn more.
