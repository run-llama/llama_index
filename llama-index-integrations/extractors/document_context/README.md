# Document Context Retrieval

## Summary

This repository contains a llama_index implementation of "contextual retrieval" (https://www.anthropic.com/news/contextual-retrieval)

It implements a custom llama_index Extractor class, which can then be used in a llama index pipeline. It requires you to initialize it using a Document Store and an LLM to provide the context. It also requires you keep the documentstore up to date. 

## Demo

See hybridsearchdemo.py for a demo of the extractor in action with Qdrant hybrid search, effectively re-implementing the blog post.

## Usage

```python
docstore = SimpleDocumentStore()

llm = OpenRouter(model="openai/gpt-4o-mini")

# initialize the extractor
extractor = DocumentContextExtractor(document_store, llm)

storage_context = StorageContext.from_defaults(vector_store=self.vector_store,
                                                            docstore=docstore,
                                                            index_store=index_store)
index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            storage_context=storage_context,
            transformations=[text_splitter, self.document_context_extractor]
        )

reader = SimpleDirectoryReader(directory)
documents = reader.load_data()

# have to keep this updated for the DocumentContextExtractor to function.
storagecontext.docstore.add_documents(documents)
for doc in documents:
    self.index.insert(doc)
```

### custom keys and prompts

by default, the extractor adds a key called "context" to each node, using a reasonable default prompt taken from the blog post cookbook, but you can pass in a list of keys and prompts like so:

```python
extractor = DocumentContextExtractor(document_store, llm, keys=["context", "title"], prompts=["Give the document context", "Provide a chunk title"])
```

## model selection
The recommended model is gpt-4o-mini, whose low cost, intelligence, high rate limits, and automatic prompt caching make it absolutely perfect.
Make sure to keep document size below the context window of your model. pre-split the documents yourself if necessary. For 4o-mini this is 128000. The extractor will warn you if a doc is too big.
Throw $50 at openai and wait 7 days, they'll give you 2,000,000 tokens/minute on 4o-mini

You're going to pay (doc_size * doc_size//chunk_size) tokens for each document in input costs, and then (num_chunks * 100) or so for output tokens.
This means 10-50 million tokens to process Pride and Prejudice, if you dont split it into chapters first.

### prompt caching with other models
I have not been able to get anthropic/other prompt caching to work with llama_index for some reason, and can only trigger prompt caching via the Anthropic python library. GPT caching is automatic.
Anthropic also has pretty harsh rate limits. 

### alternatives to 4o-mini
Gemini flash 2.0 or any other fast cheap model with high rate limits would work as well.
Keep in mind input costs add up really fast with large documents.

## TODO

- detect rate limits and retry with exponential backoff
- add a TransformComponent that splits documents into smaller documents and then adds them to the docstore
    - or better yet, a TransformComponent that simply adds the nodes to the docstore and does nothing else
    - then you can build a pipeline like this: ChapterSplitter -> DocstoreCatcher -> SentenceSplitter -> DocumentContextExtractor
- make this an installable package
- make this into an MCP server
- make a pull request to llama_index?