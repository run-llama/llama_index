# Q&A

One of the most common use-cases for an LLM application is to answer questions about a set of documents. LlamaIndex has rich support for many forms of question & answering.

## Types of question answering use cases

Q&A has all sorts of sub-types, such as:

### What to do

- **Semantic search**: finding data that matches not just your query terms, but your intent and the meaning behind your question. This is sometimes known as "top k" search.
  - [Example of semantic search](semantic-search)
- **Summarization**: condensing a large amount of data into a short summary relevant to your current question
  - [Example of summarization](summarization)

### Where to search

- **Over documents**: LlamaIndex can pull in unstructured text, PDFs, Notion and Slack documents and more and index the data within them.
  - [Example of search over documents](combine-multiple-sources)
  - [Building a multi-document agent over the LlamaIndex docs](/examples/agent/multi_document_agents-v1.ipynb)
- **Over structured data**: if your data already exists in a SQL database, as JSON or as any number of other structured formats, LlamaIndex can query the data in these sources.
  - [Searching Pandas tables](/examples/query_engine/pandas_query_engine.md)
  - [Text to SQL](/examples/index_structs/struct_indices/SQLIndexDemo.md)

### How to search

- **Combine multiple sources**: is some of your data in Slack, some in PDFs, some in unstructured text? LlamaIndex can combine queries across an arbitrary number of sources and combine them.
  - [Example of combining multiple sources](combine-multiple-sources)
- **Route across multiple sources**: given multiple data sources, your application can first pick the best source and then "route" the question to that source.
  - [Example of routing across multiple sources](route-across-multiple-sources)
- **Multi-document queries**: some questions have partial answers in multiple data sources which need to be questioned separately before they can be combined
  - [Example of multi-document queries](multi-document-queries)

## Further examples

For further examples of Q&A use cases, see our [Q&A section in Putting it All Together](/understanding/putting_it_all_together/q_and_a.html).
